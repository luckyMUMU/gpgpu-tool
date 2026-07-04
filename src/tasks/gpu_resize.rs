use std::sync::Arc;

use crate::batch::GpuBatchSubmitter;
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor, wgsl_push_constant_to_uniform};

/// 将目标宽高打包为 u32（高 16 位 = 宽，低 16 位 = 高）。
/// 当宽或高超过 65535 时返回错误，避免移位溢出导致着色器解包出错误尺寸。
fn pack_dst_dimensions(width: u32, height: u32) -> Result<u32, GpuError> {
    if width > 0xFFFF || height > 0xFFFF {
        return Err(GpuError::InvalidInput(format!(
            "目标尺寸超出限制: {}x{}, 最大支持 65535x65535",
            width, height
        )));
    }
    Ok((width << 16) | height)
}

const RESIZE_WGSL: &str = include_str!("resize.wgsl");

/// GPU 缩放器的默认 workgroup 大小。
pub const DEFAULT_RESIZE_WORKGROUP_SIZE: [u32; 3] = [8, 8, 1];
/// GPU 缩放器的默认最大缓冲区大小（256 MB）。
pub const DEFAULT_MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

/// GPU 图像缩放器配置参数。
#[derive(Debug, Clone)]
pub struct GpuResizeConfig {
    /// 计算 workgroup 大小，默认 [8, 8, 1]。
    pub workgroup_size: [u32; 3],
    /// 单次 dispatch 允许的最大缓冲区字节数，超出会拆分为多批次。
    /// 默认 256 MB。
    pub max_buffer_size: u64,
}

impl Default for GpuResizeConfig {
    fn default() -> Self {
        Self {
            workgroup_size: DEFAULT_RESIZE_WORKGROUP_SIZE,
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
        }
    }
}

impl GpuResizeConfig {
    pub fn new() -> Self { Self::default() }
    /// 设置 workgroup 大小。
    pub fn workgroup_size(mut self, v: [u32; 3]) -> Self { self.workgroup_size = v; self }
    /// 设置单批次最大缓冲大小（字节）。
    pub fn max_buffer_size(mut self, v: u64) -> Self { self.max_buffer_size = v; self }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ResizeParams {
    pub image_count: u32,
    pub src_w: u32,
    pub src_h: u32,
    pub packed_dst: u32,
}

/// ResizeParams 的 Push Constant 大小（字节数）。
const RESIZE_PUSH_CONSTANT_SIZE: u32 = std::mem::size_of::<ResizeParams>() as u32;

/// 根据设备 Push Constant 支持情况创建缩放管线描述符。
fn resize_pipeline_descriptor(
    wgsl: &'static str,
    workgroup_size: [u32; 3],
    push_constants_supported: bool,
) -> PipelineDescriptor {
    if push_constants_supported {
        PipelineDescriptor::push_constant_2_binding(wgsl, workgroup_size, RESIZE_PUSH_CONSTANT_SIZE)
    } else {
        let uniform_wgsl = wgsl_push_constant_to_uniform(wgsl);
        PipelineDescriptor::default_3_binding(uniform_wgsl, workgroup_size)
    }
}

pub struct GpuResize {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    max_buffer_size: u64,
}

impl GpuResize {
    /// 创建 GPU 缩放器（默认配置）。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_config(ctx, GpuResizeConfig::default())
    }

    /// 返回缩放管线引用（用于编码到共享 encoder）。
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// 返回 workgroup 大小。
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// 创建 GPU 缩放器，指定 workgroup_size。
    pub fn with_workgroup_size(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        Self::with_config(ctx, GpuResizeConfig::default().workgroup_size(workgroup_size))
    }

    /// 创建 GPU 缩放器，完整配置。
    pub fn with_config(
        ctx: &mut GpuContext,
        config: GpuResizeConfig,
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &resize_pipeline_descriptor(RESIZE_WGSL, config.workgroup_size, ctx.push_constants_supported()),
        )?;
        Ok(Self {
            pipeline,
            workgroup_size: config.workgroup_size,
            max_buffer_size: config.max_buffer_size,
        })
    }

    pub fn resize_batch(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        target_width: u32,
        target_height: u32,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        let (src_w, src_h) = Self::validate_batch_input(images, dimensions)?;

        let src_pixels = (src_w as u64 * src_h as u64) as usize;

        let src_u32_per_image = src_pixels as u64;
        let dst_pixels = (target_width as u64 * target_height as u64) as usize;
        let dst_u32_per_image = dst_pixels as u64;
        let u32_per_image = src_u32_per_image + dst_u32_per_image;

        let max_batch = if u32_per_image > 0 {
            ((self.max_buffer_size / 2) / (u32_per_image * 4)).max(1) as usize
        } else {
            images.len()
        };

        let mut all_resized = Vec::with_capacity(images.len());

        for chunk_start in (0..images.len()).step_by(max_batch) {
            let chunk_end = (chunk_start + max_batch).min(images.len());
            let chunk = &images[chunk_start..chunk_end];
            let resized = self.resize_batch_inner(
                ctx, chunk, src_w, src_h, target_width, target_height,
            )?;
            all_resized.extend(resized);
        }

        Ok(all_resized)
    }

    pub fn resize_batch_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        if images.is_empty() {
            let device = ctx.device()?;
            let empty_buffer = GpuBuffer::empty(device, 4, BufferUsage::Storage);
            return Ok((empty_buffer, 0));
        }

        let (src_w, src_h) = Self::validate_batch_input(images, dimensions)?;

        self.resize_batch_gpu_inner(
            ctx, images, src_w, src_h, target_width, target_height,
        )
    }

    /// 执行 resize dispatch，创建输出缓冲区并提交 GPU 计算。
    ///
    /// 封装了输出缓冲区创建 → 参数编码 → dispatch_with_params 的公共流程，
    /// 消除 `resize_batch_gpu_inner`、`resize_batch_gpu_from_buffer` 和
    /// `resize_batch_inner` 之间的重复逻辑。
    #[allow(clippy::too_many_arguments)]
    fn dispatch_resize(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        image_count: usize,
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let dst_pixels = (target_width as u64 * target_height as u64) as usize;
        let output_u32_count = image_count * dst_pixels;
        let output_size = (output_u32_count * 4) as u64;

        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, output_size, BufferUsage::Storage)?;
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = ResizeParams {
            image_count: image_count as u32,
            src_w,
            src_h,
            packed_dst: pack_dst_dimensions(target_width, target_height)?,
        };
        let params_arr = [params];
        let params_bytes = bytemuck::cast_slice::<ResizeParams, u8>(&params_arr);

        let dispatch_x = target_width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = target_height.div_ceil(self.workgroup_size[1]).max(1);
        let dispatch_z = image_count as u32;

        self.pipeline.dispatch_with_params(
            device,
            queue,
            params_bytes,
            &[input_buffer, &output_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            ctx.compute_units(),
        );

        Ok((output_buffer, output_u32_count))
    }

    /// 执行 resize dispatch（批量编码模式），将命令编码到共享 encoder。
    ///
    /// 与 `dispatch_resize()` 功能相同，但将 dispatch 命令编码到
    /// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
    #[allow(clippy::too_many_arguments)]
    fn dispatch_resize_batch(
        &self,
        ctx: &GpuContext,
        batch: &mut GpuBatchSubmitter,
        input_buffer: &GpuBuffer,
        image_count: usize,
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let device = ctx.device()?;
        let dst_pixels = (target_width as u64 * target_height as u64) as usize;
        let output_u32_count = image_count * dst_pixels;
        let output_size = (output_u32_count * 4) as u64;

        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, output_size, BufferUsage::Storage)?;
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = ResizeParams {
            image_count: image_count as u32,
            src_w,
            src_h,
            packed_dst: pack_dst_dimensions(target_width, target_height)?,
        };
        let params_arr = [params];
        let params_bytes = bytemuck::cast_slice::<ResizeParams, u8>(&params_arr);

        let dispatch_x = target_width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = target_height.div_ceil(self.workgroup_size[1]).max(1);
        let dispatch_z = image_count as u32;

        // 根据管线配置选择 Push Constant 或 Uniform buffer
        if self.pipeline.push_constant_size().is_some() {
            batch.encode_dispatch(
                ctx,
                &self.pipeline,
                &[input_buffer, &output_buffer],
                [dispatch_x, dispatch_y, dispatch_z],
                Some(params_bytes),
                ctx.compute_units(),
            )?;
        } else {
            let params_buffer = GpuBuffer::from_bytes(device, params_bytes, BufferUsage::Uniform);
            batch.encode_dispatch(
                ctx,
                &self.pipeline,
                &[input_buffer, &output_buffer, &params_buffer],
                [dispatch_x, dispatch_y, dispatch_z],
                None,
                ctx.compute_units(),
            )?;
        }

        Ok((output_buffer, output_u32_count))
    }

    fn resize_batch_gpu_inner(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let image_count = images.len();
        let src_pixels = (src_w as u64 * src_h as u64) as usize;

        let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * src_pixels);
        crate::pixel_pack::pack_u8_batch_to_u32(images, &mut all_pixels);

        let input_size = (all_pixels.len() * 4) as u64;
        let input_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, input_size, BufferUsage::Storage)?;
        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));
        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);

        let result = self.dispatch_resize(
            ctx, &input_buffer, image_count, src_w, src_h, target_width, target_height,
        )?;

        ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);

        Ok(result)
    }

    /// 从 GPU buffer 输入执行批量缩放，返回 GPU buffer（零拷贝流水线）。
    ///
    /// 输入 buffer 包含 `image_count` 张尺寸为 `src_w × src_h` 的已打包 u32 像素数据，
    /// 输出 buffer 包含缩放后的 u32 像素数据。
    #[allow(clippy::too_many_arguments)]
    pub fn resize_batch_gpu_from_buffer(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        src_w: u32,
        src_h: u32,
        image_count: usize,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        if image_count == 0 {
            return Err(GpuError::InvalidInput("图像数量为 0".to_string()));
        }

        if target_width == 0 || target_height == 0 {
            return Err(GpuError::InvalidInput("目标尺寸不能为零".to_string()));
        }

        self.dispatch_resize(
            ctx, input_buffer, image_count, src_w, src_h, target_width, target_height,
        )
    }

    /// 从 GPU buffer 输入执行批量缩放（批量编码模式），返回 GPU buffer。
    ///
    /// 与 `resize_batch_gpu_from_buffer()` 功能相同，但将 dispatch 命令编码到
    /// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
    #[allow(clippy::too_many_arguments)]
    pub fn resize_batch_gpu_from_buffer_batch(
        &self,
        ctx: &GpuContext,
        batch: &mut GpuBatchSubmitter,
        input_buffer: &GpuBuffer,
        src_w: u32,
        src_h: u32,
        image_count: usize,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        if image_count == 0 {
            return Err(GpuError::InvalidInput("图像数量为 0".to_string()));
        }

        if target_width == 0 || target_height == 0 {
            return Err(GpuError::InvalidInput("目标尺寸不能为零".to_string()));
        }

        self.dispatch_resize_batch(
            ctx, batch, input_buffer, image_count, src_w, src_h, target_width, target_height,
        )
    }

    fn validate_batch_input(
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<(u32, u32), GpuError> {
        if images.len() != dimensions.len() {
            return Err(GpuError::InvalidInput(
                "图像数量与尺寸数量不匹配".to_string(),
            ));
        }
        let (src_w, src_h) = dimensions[0];
        let src_pixels = (src_w as u64 * src_h as u64) as usize;
        for (i, img) in images.iter().enumerate() {
            if dimensions[i] != (src_w, src_h) {
                return Err(GpuError::InvalidInput(
                    "批量缩放要求所有图像尺寸相同".to_string(),
                ));
            }
            if img.len() != src_pixels {
                return Err(GpuError::InvalidInput(format!(
                    "第 {} 张图像像素数量与声明尺寸不符",
                    i
                )));
            }
        }
        Ok((src_w, src_h))
    }

    fn resize_batch_inner(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let image_count = images.len();
        let dst_pixels = (target_width as u64 * target_height as u64) as usize;

        let (output_buffer, _) = self.resize_batch_gpu_inner(
            ctx, images, src_w, src_h, target_width, target_height,
        )?;

        let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;
        ctx.buffer_pool().release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        let mut resized_images = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let offset = i * dst_pixels;
            let pixels = crate::pixel_pack::unpack_u32_to_u8(&raw_u32[offset..offset + dst_pixels], dst_pixels);
            resized_images.push(pixels);
        }

        Ok(resized_images)
    }
}
