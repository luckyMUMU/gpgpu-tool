use std::sync::Arc;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::buffer_pool::BufferPool;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor};

const CONVOLUTION_WGSL: &str = include_str!("convolution.wgsl");

/// GPU 卷积器的默认 workgroup 大小。
pub const DEFAULT_CONVOLUTION_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// 卷积核最大边长（11×11 = 121 个 f32）。
const MAX_KERNEL_SIZE: u32 = 11;

/// uniform buffer 中卷积核数组的固定长度（124 个 f32，对齐到 vec4 边界）。
const KERNEL_ARRAY_LEN: usize = 124;

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
pub struct GpuConvolution {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    buffer_pool: BufferPool,
}

impl GpuConvolution {
    /// 创建 GPU 卷积计算器。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor {
                bindings: vec![
                    crate::pipeline::BindingType::StorageReadOnly,
                    crate::pipeline::BindingType::StorageReadWrite,
                    crate::pipeline::BindingType::StorageReadOnly,
                ],
                wgsl: CONVOLUTION_WGSL,
                workgroup_size: DEFAULT_CONVOLUTION_WORKGROUP_SIZE,
            },
        )?;
        Ok(Self {
            pipeline,
            workgroup_size: DEFAULT_CONVOLUTION_WORKGROUP_SIZE,
            buffer_pool: BufferPool::new(),
        })
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

        let device = ctx.device();
        let queue = ctx.queue();
        let pixel_count = (width * height) as usize;

        let packed = crate::pixel_pack::pack_u8_to_u32(pixels);

        let input_size = (packed.len() * 4) as u64;
        let output_size = input_size;

        let input_buffer_raw = self
            .buffer_pool
            .acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .buffer_pool
            .acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = build_params(width, height, kernel_size, border_mode, 0, kernel)?;
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Storage);

        let dispatch_x = (pixel_count as u32)
            .div_ceil(self.workgroup_size[0])
            .max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, 1, 1],
        );

        let result = output_buffer.download_with_pool(device, queue, &self.buffer_pool)?;

        self.buffer_pool
            .release(input_buffer.into_raw(), BufferUsage::Storage);
        self.buffer_pool
            .release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        Ok(crate::pixel_pack::unpack_u32_to_u8(raw_u32, pixel_count))
    }

    /// 执行可分离卷积（水平 + 垂直两趟 1D）。
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

        let device = ctx.device();
        let queue = ctx.queue();
        let pixel_count = (width * height) as usize;

        let packed = crate::pixel_pack::pack_u8_to_u32(pixels);

        let buffer_size = (packed.len() * 4) as u64;

        let input_buffer_raw = self
            .buffer_pool
            .acquire(device, buffer_size, BufferUsage::Storage);
        let intermediate_buffer_raw = self
            .buffer_pool
            .acquire(device, buffer_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .buffer_pool
            .acquire(device, buffer_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, buffer_size);
        let intermediate_buffer = GpuBuffer::from_raw(intermediate_buffer_raw, buffer_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, buffer_size);

        let dispatch_x = (pixel_count as u32)
            .div_ceil(self.workgroup_size[0])
            .max(1);

        // 单 encoder 编码两趟 dispatch，减少同步开销
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("separable_convolution_encoder"),
        });

        // 第一趟：水平 1D 卷积
        let params_h = build_params(width, height, kernel_size, border_mode, 1, kernel_1d)?;
        let params_buffer_h = GpuBuffer::from_data(device, &[params_h], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&input_buffer, &intermediate_buffer, &params_buffer_h],
            [dispatch_x, 1, 1],
        );

        // 第二趟：垂直 1D 卷积
        let params_v = build_params(width, height, kernel_size, border_mode, 2, kernel_1d)?;
        let params_buffer_v = GpuBuffer::from_data(device, &[params_v], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&intermediate_buffer, &output_buffer, &params_buffer_v],
            [dispatch_x, 1, 1],
        );

        // 单次提交两趟 dispatch
        queue.submit(std::iter::once(encoder.finish()));

        let result = output_buffer.download_with_pool(device, queue, &self.buffer_pool)?;

        self.buffer_pool
            .release(input_buffer.into_raw(), BufferUsage::Storage);
        self.buffer_pool
            .release(intermediate_buffer.into_raw(), BufferUsage::Storage);
        self.buffer_pool
            .release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        Ok(crate::pixel_pack::unpack_u32_to_u8(raw_u32, pixel_count))
    }

    /// 执行可分离卷积，返回 GPU buffer（零拷贝流水线支持）。
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

        let pixel_count = (width * height) as usize;
        if input_u32_count != pixel_count {
            return Err(GpuError::InvalidInput(
                "输入 u32 数量与声明尺寸不符".to_string(),
            ));
        }

        let device = ctx.device();
        let queue = ctx.queue();
        let buffer_size = (pixel_count * 4) as u64;

        let intermediate_buffer_raw = self
            .buffer_pool
            .acquire(device, buffer_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .buffer_pool
            .acquire(device, buffer_size, BufferUsage::Storage);

        let intermediate_buffer = GpuBuffer::from_raw(intermediate_buffer_raw, buffer_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, buffer_size);

        let dispatch_x = (pixel_count as u32)
            .div_ceil(self.workgroup_size[0])
            .max(1);

        // 单 encoder 编码两趟 dispatch，减少同步开销
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("separable_convolution_gpu_encoder"),
        });

        // 第一趟：水平 1D 卷积
        let params_h = build_params(width, height, kernel_size, border_mode, 1, kernel_1d)?;
        let params_buffer_h = GpuBuffer::from_data(device, &[params_h], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[input_buffer, &intermediate_buffer, &params_buffer_h],
            [dispatch_x, 1, 1],
        );

        // 第二趟：垂直 1D 卷积
        let params_v = build_params(width, height, kernel_size, border_mode, 2, kernel_1d)?;
        let params_buffer_v = GpuBuffer::from_data(device, &[params_v], BufferUsage::Storage);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&intermediate_buffer, &output_buffer, &params_buffer_v],
            [dispatch_x, 1, 1],
        );

        // 单次提交两趟 dispatch
        queue.submit(std::iter::once(encoder.finish()));

        self.buffer_pool
            .release(intermediate_buffer.into_raw(), BufferUsage::Storage);

        Ok(output_buffer)
    }

    /// 返回内部缓冲区复用池的引用。
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }
}

fn validate_dimensions(pixels: &[u8], width: u32, height: u32) -> Result<(), GpuError> {
    if width == 0 || height == 0 {
        return Err(GpuError::InvalidInput(
            "图像宽高必须大于 0".to_string(),
        ));
    }
    let pixel_count = (width * height) as usize;
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
