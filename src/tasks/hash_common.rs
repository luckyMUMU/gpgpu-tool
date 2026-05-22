use crate::buffer::{BufferUsage, GpuBuffer};
use crate::buffer_pool::BufferPool;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

/// 感知哈希公共参数，通过 uniform buffer 传递给 WGSL。
///
/// 字段顺序必须与 WGSL 着色器中 `params: vec4<u32>` 一致：
/// `params.x = image_count, params.y = width, params.z = height`
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhashParams {
    /// 图像数量（对应 WGSL params.x）
    pub image_count: u32,
    /// 图像宽度（对应 WGSL params.y）
    pub width: u32,
    /// 图像高度（对应 WGSL params.z）
    pub height: u32,
    /// 保留对齐（对应 WGSL params.w）
    pub _padding: u32,
}

/// 感知哈希计算器的统一接口。
///
/// 所有感知哈希算法（Mean、Gradient、Block 等）均实现此 trait，
/// 便于上层通过多态方式调用不同算法。
pub trait PerceptualHashComputer {
    fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError>;

    fn pipeline(&self) -> &ComputePipeline;
    fn workgroup_size(&self) -> [u32; 3];
}

/// 通用感知哈希 GPU 计算流程。
///
/// 封装了像素打包 → 缓冲区创建 → dispatch → 下载 → 解析的完整流水线，
/// 所有感知哈希算法均可复用此实现。
pub fn compute_phash(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    buffer_pool: &BufferPool,
) -> Result<Vec<u64>, GpuError> {
    if images.is_empty() {
        return Ok(vec![]);
    }

    let device = ctx.device();
    let queue = ctx.queue();

    let image_count = images.len();
    let pixels_per_image = (width * height) as usize;

    let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * pixels_per_image);
    for img in images {
        if img.len() != pixels_per_image {
            return Err(GpuError::InvalidInput(
                "所有图像尺寸必须一致".to_string(),
            ));
        }
        let start = all_pixels.len();
        all_pixels.resize(start + img.len(), 0);
        for (i, &pixel) in img.iter().enumerate() {
            all_pixels[start + i] = pixel as u32;
        }
    }

    let input_size = (all_pixels.len() * 4) as u64;
    let output_size = (image_count * 2 * 4) as u64;

    let input_buffer_raw = buffer_pool.acquire(device, input_size, BufferUsage::Storage);
    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage);

    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));

    let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        _padding: 0,
    };
    let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

    let dispatch_x = (image_count as u32).div_ceil(workgroup_size[0]).max(1);
    pipeline.dispatch(
        device,
        queue,
        &input_buffer,
        &output_buffer,
        &params_buffer,
        [dispatch_x, 1, 1],
    );

    let result = output_buffer.download_with_pool(device, queue, buffer_pool)?;

    buffer_pool.release(input_buffer.into_raw());
    buffer_pool.release(output_buffer.into_raw());

    let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);

    let mut hashes = Vec::with_capacity(image_count);
    for i in 0..image_count {
        let low = raw_u32[i * 2] as u64;
        let high = raw_u32[i * 2 + 1] as u64;
        hashes.push(low | (high << 32));
    }

    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（从 GPU buffer 输入）。
///
/// 与 `compute_phash` 相同的计算逻辑，但输入数据已在 GPU buffer 中，
/// 避免了 CPU→GPU 的数据传输。适用于 GPU 缩放→GPU 哈希的零拷贝流水线。
///
/// `input_buffer` 应包含打包后的 u32 像素数据，
/// `input_u32_count` 为 u32 元素总数。
#[allow(clippy::too_many_arguments)]
pub fn compute_phash_from_gpu_buffer(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    input_buffer: &GpuBuffer,
    _input_u32_count: usize,
    image_count: usize,
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    buffer_pool: &BufferPool,
) -> Result<Vec<u64>, GpuError> {
    if image_count == 0 {
        return Ok(vec![]);
    }

    let device = ctx.device();
    let queue = ctx.queue();
    let output_size = (image_count * 2 * 4) as u64;

    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        _padding: 0,
    };
    let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

    let dispatch_x = (image_count as u32).div_ceil(workgroup_size[0]).max(1);
    pipeline.dispatch(
        device,
        queue,
        input_buffer,
        &output_buffer,
        &params_buffer,
        [dispatch_x, 1, 1],
    );

    let result = output_buffer.download_with_pool(device, queue, buffer_pool)?;

    buffer_pool.release(output_buffer.into_raw());

    let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);

    let mut hashes = Vec::with_capacity(image_count);
    for i in 0..image_count {
        let low = raw_u32[i * 2] as u64;
        let high = raw_u32[i * 2 + 1] as u64;
        hashes.push(low | (high << 32));
    }

    Ok(hashes)
}

/// 创建默认感知哈希计算器结构体。
/// 封装 pipeline + workgroup_size 字段和 new / with_workgroup_size 构造函数。
#[macro_export]
macro_rules! declare_phash_computer {
    ($name:ident, $doc:expr, $wgsl:ident, $wg_size:expr) => {
        #[doc = $doc]
        pub struct $name {
            pipeline: ::std::sync::Arc<$crate::pipeline::ComputePipeline>,
            workgroup_size: [u32; 3],
            buffer_pool: $crate::buffer_pool::BufferPool,
        }

        impl $name {
            #[doc = concat!("创建 ", stringify!($name), "，使用默认 workgroup_size")]
            pub fn new(ctx: &mut $crate::context::GpuContext) -> Result<Self, $crate::error::GpuError> {
                Self::with_workgroup_size(ctx, $wg_size)
            }

            #[doc = concat!("创建 ", stringify!($name), "，指定 workgroup_size")]
            pub fn with_workgroup_size(
                ctx: &mut $crate::context::GpuContext,
                workgroup_size: [u32; 3],
            ) -> Result<Self, $crate::error::GpuError> {
                let pipeline = ctx.get_or_create_pipeline($wgsl, workgroup_size)?;
                Ok(Self {
                    pipeline: ::std::sync::Arc::clone(&pipeline),
                    workgroup_size,
                    buffer_pool: $crate::buffer_pool::BufferPool::new(),
                })
            }

            pub fn pipeline(&self) -> &$crate::pipeline::ComputePipeline {
                &self.pipeline
            }

            pub fn workgroup_size_val(&self) -> [u32; 3] {
                self.workgroup_size
            }
        }
    };
}

/// 简化：为感知哈希算法生成 PerceptualHashComputer trait 实现。
#[macro_export]
macro_rules! impl_phash_computer_simple {
    ($name:ident) => {
        impl $crate::tasks::hash_common::PerceptualHashComputer for $name {
            fn compute(
                &self,
                ctx: &$crate::context::GpuContext,
                images: &[Vec<u8>],
            ) -> Result<Vec<u64>, $crate::error::GpuError> {
                if images.is_empty() {
                    return Ok(vec![]);
                }
                let pixels_per_image = images[0].len();
                let width = (pixels_per_image as u32).isqrt();
                let height = width;
                $crate::tasks::hash_common::compute_phash(
                    &self.pipeline, ctx, images, width, height, self.workgroup_size, &self.buffer_pool,
                )
            }

            fn pipeline(&self) -> &$crate::pipeline::ComputePipeline {
                &self.pipeline
            }

            fn workgroup_size(&self) -> [u32; 3] {
                self.workgroup_size
            }
        }
    };
}

/// 为需要自定义 width/height 推断的感知哈希算法生成 PerceptualHashComputer 实现。
#[macro_export]
macro_rules! impl_phash_computer_custom_dims {
    ($name:ident, $mode:expr) => {
        impl $crate::tasks::hash_common::PerceptualHashComputer for $name {
            fn compute(
                &self,
                ctx: &$crate::context::GpuContext,
                images: &[Vec<u8>],
            ) -> Result<Vec<u64>, $crate::error::GpuError> {
                if images.is_empty() {
                    return Ok(vec![]);
                }
                let pixels_per_image = images[0].len() as u32;
                let (width, height) = match $mode {
                    0 => {
                        let w = pixels_per_image.div_ceil(9);
                        (w, pixels_per_image / w)
                    }
                    1 => {
                        let h = pixels_per_image.div_ceil(9);
                        (pixels_per_image / h, h)
                    }
                    _ => {
                        let w = (pixels_per_image as f64).sqrt() as u32;
                        (w, w)
                    }
                };
                $crate::tasks::hash_common::compute_phash(
                    &self.pipeline, ctx, images, width, height, self.workgroup_size, &self.buffer_pool,
                )
            }

            fn pipeline(&self) -> &$crate::pipeline::ComputePipeline {
                &self.pipeline
            }

            fn workgroup_size(&self) -> [u32; 3] {
                self.workgroup_size
            }
        }
    };
}
