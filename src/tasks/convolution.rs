use std::sync::Arc;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor};

const CONVOLUTION_WGSL: &str = include_str!("convolution.wgsl");

/// GPU 卷积器的默认 workgroup 大小。
pub const DEFAULT_CONVOLUTION_WORKGROUP_SIZE: [u32; 3] = [8, 8, 1];

/// 卷积核最大边长（11×11 = 121 个 f32）。
const MAX_KERNEL_SIZE: u32 = 11;

/// uniform buffer 中卷积核数组的固定长度（124 个 f32，对齐到 vec4 边界）。
const KERNEL_ARRAY_LEN: usize = 124;

const MAX_RADIUS: u32 = (MAX_KERNEL_SIZE - 1) / 2;
const WG_X: u32 = DEFAULT_CONVOLUTION_WORKGROUP_SIZE[0];
const WG_Y: u32 = DEFAULT_CONVOLUTION_WORKGROUP_SIZE[1];
const LDS_H_SIZE: u32 = (WG_Y + 2 * MAX_RADIUS) * WG_X;

const _: () = assert!(
    LDS_H_SIZE == 144,
    "LDS 大小与 WGSL 不一致，请同步更新 convolution.wgsl 中的 lds_h 声明"
);

/// pass_mode 常量：融合可分离卷积（LDS 优化）。
const PASS_MODE_FUSED_SEPARABLE: u32 = 3;

/// 边界处理模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BorderMode {
    /// 越界像素值为 0。
    Zero,
    /// 钳制到最近的边缘像素。
    Clamp,
    /// 镜像反射。
    Reflect,
}

impl BorderMode {
    fn as_u32(&self) -> u32 {
        match self {
            BorderMode::Zero => 0,
            BorderMode::Clamp => 1,
            BorderMode::Reflect => 2,
        }
    }
}

/// 卷积模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvMode {
    /// 不可分离 2D 卷积（单趟）。
    Full2D,
    /// 可分离卷积（水平 + 垂直两趟 1D）。
    Separable,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    kernel_radius: u32,
    border_mode: u32,
    pass_mode: u32,
    _pad1: u32,
    _pad2: u32,
    kernel: [f32; KERNEL_ARRAY_LEN],
}

/// GPU 2D 卷积计算器。
///
/// 支持两种可分离卷积路径：
/// - **LDS 优化路径**（默认）：水平 + 垂直 pass 融合在单次 dispatch 中，
///   水平结果存入 LDS 共享内存，垂直 pass 从 LDS 读取，避免一次全局内存往返。
/// - **全局内存路径**（回退）：水平 pass 结果写回全局内存，垂直 pass 再从全局内存读取。
pub struct GpuConvolution {
    pipeline: Arc<ComputePipeline>,
    fused_pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    use_lds: bool,
}

impl GpuConvolution {
    /// 创建 GPU 卷积计算器，默认启用 LDS 优化。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        let bindings = vec![
            crate::pipeline::BindingType::StorageReadOnly,
            crate::pipeline::BindingType::StorageReadWrite,
            crate::pipeline::BindingType::StorageReadOnly,
        ];

        let pipeline = ctx.get_or_create_pipeline(&PipelineDescriptor {
            bindings: bindings.clone(),
            wgsl: CONVOLUTION_WGSL.to_string(),
            workgroup_size: DEFAULT_CONVOLUTION_WORKGROUP_SIZE,
            entry_point: "main",
            push_constant_size: None,
        })?;

        let fused_pipeline = ctx.get_or_create_pipeline(&PipelineDescriptor {
            bindings,
            wgsl: CONVOLUTION_WGSL.to_string(),
            workgroup_size: DEFAULT_CONVOLUTION_WORKGROUP_SIZE,
            entry_point: "separable_fused",
            push_constant_size: None,
        })?;

        Ok(Self {
            pipeline,
            fused_pipeline,
            workgroup_size: DEFAULT_CONVOLUTION_WORKGROUP_SIZE,
            use_lds: true,
        })
    }

    /// 设置是否启用 LDS 优化，返回 `self` 支持链式调用。
    pub fn with_lds(mut self, use_lds: bool) -> Self {
        self.use_lds = use_lds;
        self
    }

    /// 返回是否启用 LDS 优化。
    pub fn use_lds(&self) -> bool {
        self.use_lds
    }

    /// 动态切换 LDS 优化开关。
    pub fn set_use_lds(&mut self, use_lds: bool) {
        self.use_lds = use_lds;
    }

    /// 执行 2D 卷积（不可分离）。
    #[allow(clippy::too_many_arguments)]
    pub fn convolve_2d(
        &self,
        ctx: &GpuContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        kernel: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<Vec<u8>, GpuError> {
        validate_2d_kernel(kernel, kernel_size)?;
        validate_dimensions(pixels, width, height)?;

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let pixel_count = (width as u64 * height as u64) as usize;

        let packed = crate::pixel_pack::pack_u8_to_u32(pixels);

        let input_size = (packed.len() * 4) as u64;
        let output_size = input_size;

        let input_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = build_params(width, height, kernel_size, border_mode, 0, kernel)?;
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Storage);

        let dispatch_x = width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = height.div_ceil(self.workgroup_size[1]).max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;

        ctx.buffer_pool()
            .release(input_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool()
            .release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        Ok(crate::pixel_pack::unpack_u32_to_u8(raw_u32, pixel_count))
    }

    /// 执行可分离卷积（水平 + 垂直两趟 1D）。
    ///
    /// 启用 LDS 优化时使用融合管线（单次 dispatch），否则使用全局内存双 pass。
    #[allow(clippy::too_many_arguments)]
    pub fn convolve_separable(
        &self,
        ctx: &GpuContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<Vec<u8>, GpuError> {
        validate_1d_kernel(kernel_1d, kernel_size)?;
        validate_dimensions(pixels, width, height)?;

        if self.use_lds {
            self.convolve_separable_lds(ctx, pixels, width, height, kernel_1d, kernel_size, border_mode)
        } else {
            self.convolve_separable_global(ctx, pixels, width, height, kernel_1d, kernel_size, border_mode)
        }
    }

    /// 执行可分离卷积，返回 GPU buffer（零拷贝流水线支持）。
    ///
    /// 启用 LDS 优化时使用融合管线（单次 dispatch），否则使用全局内存双 pass。
    #[allow(clippy::too_many_arguments)]
    pub fn convolve_separable_gpu(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        input_u32_count: usize,
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<GpuBuffer, GpuError> {
        validate_1d_kernel(kernel_1d, kernel_size)?;

        let pixel_count = (width as u64 * height as u64) as usize;
        if input_u32_count != pixel_count {
            return Err(GpuError::InvalidInput(
                "输入 u32 数量与声明尺寸不符".to_string(),
            ));
        }

        if self.use_lds {
            self.convolve_separable_gpu_lds(ctx, input_buffer, width, height, kernel_1d, kernel_size, border_mode)
        } else {
            self.convolve_separable_gpu_global(ctx, input_buffer, width, height, kernel_1d, kernel_size, border_mode)
        }
    }

    /// LDS 优化的可分离卷积：融合水平 + 垂直 pass 为单次 dispatch。
    ///
    /// 水平结果存入 LDS 共享内存，垂直 pass 从 LDS 读取，
    /// 无需中间缓冲区，减少一次全局内存往返。
    #[allow(clippy::too_many_arguments)]
    fn convolve_separable_lds(
        &self,
        ctx: &GpuContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<Vec<u8>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let pixel_count = (width * height) as usize;

        let packed = crate::pixel_pack::pack_u8_to_u32(pixels);

        let input_size = (packed.len() * 4) as u64;
        let output_size = input_size;

        let input_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = build_params(width, height, kernel_size, border_mode, PASS_MODE_FUSED_SEPARABLE, kernel_1d)?;
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Storage);

        let dispatch_x = width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = height.div_ceil(self.workgroup_size[1]).max(1);
        self.fused_pipeline.dispatch(
            device,
            queue,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;

        ctx.buffer_pool()
            .release(input_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool()
            .release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        Ok(crate::pixel_pack::unpack_u32_to_u8(raw_u32, pixel_count))
    }

    /// LDS 优化的可分离卷积（GPU buffer 版）：融合水平 + 垂直 pass 为单次 dispatch。
    #[allow(clippy::too_many_arguments)]
    fn convolve_separable_gpu_lds(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<GpuBuffer, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let pixel_count = (width as u64 * height as u64) as usize;
        let buffer_size = (pixel_count * 4) as u64;

        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, buffer_size);

        let params = build_params(width, height, kernel_size, border_mode, PASS_MODE_FUSED_SEPARABLE, kernel_1d)?;
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Storage);

        let dispatch_x = width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = height.div_ceil(self.workgroup_size[1]).max(1);
        self.fused_pipeline.dispatch(
            device,
            queue,
            &[input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        Ok(output_buffer)
    }

    /// 全局内存路径的可分离卷积：水平 pass 结果写回全局内存，垂直 pass 再从全局内存读取。
    #[allow(clippy::too_many_arguments)]
    fn convolve_separable_global(
        &self,
        ctx: &GpuContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<Vec<u8>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let pixel_count = (width as u64 * height as u64) as usize;

        let packed = crate::pixel_pack::pack_u8_to_u32(pixels);

        let buffer_size = (packed.len() * 4) as u64;

        let input_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);
        let intermediate_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);
        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, buffer_size);
        let intermediate_buffer = GpuBuffer::from_raw(intermediate_buffer_raw, buffer_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, buffer_size);

        let dispatch_x = width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = height.div_ceil(self.workgroup_size[1]).max(1);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("separable_convolution_encoder"),
        });

        let params_h = build_params(width, height, kernel_size, border_mode, 1, kernel_1d)?;
        let params_buffer_h = GpuBuffer::from_data(device, &[params_h], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&input_buffer, &intermediate_buffer, &params_buffer_h],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let params_v = build_params(width, height, kernel_size, border_mode, 2, kernel_1d)?;
        let params_buffer_v = GpuBuffer::from_data(device, &[params_v], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&intermediate_buffer, &output_buffer, &params_buffer_v],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        queue.submit(std::iter::once(encoder.finish()));

        let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;

        ctx.buffer_pool()
            .release(input_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool()
            .release(intermediate_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool()
            .release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        Ok(crate::pixel_pack::unpack_u32_to_u8(raw_u32, pixel_count))
    }

    /// 全局内存路径的可分离卷积（GPU buffer 版）。
    #[allow(clippy::too_many_arguments)]
    fn convolve_separable_gpu_global(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        width: u32,
        height: u32,
        kernel_1d: &[f32],
        kernel_size: u32,
        border_mode: BorderMode,
    ) -> Result<GpuBuffer, GpuError> {
        let pixel_count = (width as u64 * height as u64) as usize;
        let buffer_size = (pixel_count * 4) as u64;

        let device = ctx.device()?;
        let queue = ctx.queue()?;

        let intermediate_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);
        let output_buffer_raw = ctx
            .buffer_pool()
            .acquire(device, buffer_size, BufferUsage::Storage);

        let intermediate_buffer = GpuBuffer::from_raw(intermediate_buffer_raw, buffer_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, buffer_size);

        let dispatch_x = width.div_ceil(self.workgroup_size[0]).max(1);
        let dispatch_y = height.div_ceil(self.workgroup_size[1]).max(1);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("separable_convolution_gpu_encoder"),
        });

        let params_h = build_params(width, height, kernel_size, border_mode, 1, kernel_1d)?;
        let params_buffer_h = GpuBuffer::from_data(device, &[params_h], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[input_buffer, &intermediate_buffer, &params_buffer_h],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        let params_v = build_params(width, height, kernel_size, border_mode, 2, kernel_1d)?;
        let params_buffer_v = GpuBuffer::from_data(device, &[params_v], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&intermediate_buffer, &output_buffer, &params_buffer_v],
            [dispatch_x, dispatch_y, 1],
            None,
            ctx.compute_units(),
        );

        queue.submit(std::iter::once(encoder.finish()));

        ctx.buffer_pool()
            .release(intermediate_buffer.into_raw(), BufferUsage::Storage);

        Ok(output_buffer)
    }
}

fn validate_dimensions(pixels: &[u8], width: u32, height: u32) -> Result<(), GpuError> {
    if width == 0 || height == 0 {
        return Err(GpuError::InvalidInput(
            "图像宽高必须大于 0".to_string(),
        ));
    }
    let pixel_count = (width as u64 * height as u64) as usize;
    if pixels.len() != pixel_count {
        return Err(GpuError::InvalidInput(
            "像素数量与声明尺寸不符".to_string(),
        ));
    }
    Ok(())
}

fn validate_2d_kernel(kernel: &[f32], kernel_size: u32) -> Result<(), GpuError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(GpuError::InvalidInput(
            "卷积核边长必须为正奇数".to_string(),
        ));
    }
    if kernel_size > MAX_KERNEL_SIZE {
        return Err(GpuError::InvalidInput(format!(
            "卷积核边长不能超过 {}",
            MAX_KERNEL_SIZE
        )));
    }
    let expected = (kernel_size * kernel_size) as usize;
    if kernel.len() != expected {
        return Err(GpuError::InvalidInput(format!(
            "2D 卷积核长度不匹配: 期望 {}, 实际 {}",
            expected,
            kernel.len()
        )));
    }
    Ok(())
}

fn validate_1d_kernel(kernel: &[f32], kernel_size: u32) -> Result<(), GpuError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(GpuError::InvalidInput(
            "卷积核长度必须为正奇数".to_string(),
        ));
    }
    if kernel_size > MAX_KERNEL_SIZE {
        return Err(GpuError::InvalidInput(format!(
            "卷积核长度不能超过 {}",
            MAX_KERNEL_SIZE
        )));
    }
    if kernel.len() != kernel_size as usize {
        return Err(GpuError::InvalidInput(format!(
            "1D 卷积核长度不匹配: 期望 {}, 实际 {}",
            kernel_size,
            kernel.len()
        )));
    }
    Ok(())
}

fn build_params(
    width: u32,
    height: u32,
    kernel_size: u32,
    border_mode: BorderMode,
    pass_mode: u32,
    kernel: &[f32],
) -> Result<ConvParams, GpuError> {
    let kernel_radius = kernel_size / 2;

    let mut kernel_arr = [0.0f32; KERNEL_ARRAY_LEN];
    let copy_len = kernel.len().min(KERNEL_ARRAY_LEN);
    kernel_arr[..copy_len].copy_from_slice(&kernel[..copy_len]);

    Ok(ConvParams {
        width,
        height,
        kernel_size,
        kernel_radius,
        border_mode: border_mode.as_u32(),
        pass_mode,
        _pad1: 0,
        _pad2: 0,
        kernel: kernel_arr,
    })
}
