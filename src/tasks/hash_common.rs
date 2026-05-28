use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor};

/// 感知哈希公共参数，通过 Push Constant 传递给 WGSL。
///
/// 字段顺序必须与 WGSL 着色器中 `Params` struct 一致。
/// 4 个 u32 = 16 字节，满足 Push Constant 4 字节对齐和 Uniform 16 字节对齐。
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PhashParams {
    pub image_count: u32,
    pub width: u32,
    pub height: u32,
    pub hash_size: u32,
}

/// Push Constant 参数大小（字节数）。
const PHASH_PUSH_CONSTANT_SIZE: u32 = std::mem::size_of::<PhashParams>() as u32;

/// 将 Push Constant 版 WGSL 转换为 Uniform buffer 回退版 WGSL。
///
/// 替换 `var<push_constant> params: Params;` 为
/// `@group(0) @binding(2) var<uniform> params: Params;`。
fn wgsl_push_constant_to_uniform(wgsl: &str) -> String {
    wgsl.replace(
        "var<push_constant> params: Params;",
        "@group(0) @binding(2) var<uniform> params: Params;",
    )
}

/// 根据设备 Push Constant 支持情况创建感知哈希管线描述符。
///
/// 支持时使用 2-binding + Push Constant 布局（参数通过 Push Constant 传递），
/// 不支持时回退到 3-binding + Uniform buffer 布局（WGSL 自动转换）。
pub fn phash_pipeline_descriptor(
    wgsl: &'static str,
    workgroup_size: [u32; 3],
    push_constants_supported: bool,
) -> PipelineDescriptor {
    if push_constants_supported {
        PipelineDescriptor::push_constant_2_binding(wgsl, workgroup_size, PHASH_PUSH_CONSTANT_SIZE)
    } else {
        let uniform_wgsl = wgsl_push_constant_to_uniform(wgsl);
        PipelineDescriptor::default_3_binding(uniform_wgsl, workgroup_size)
    }
}

/// 感知哈希网格尺寸配置，控制哈希的精细度。
///
/// `hash_size` 为网格边长（像素数），最终哈希位宽 = hash_size × hash_size。
///
/// | hash_size | 网格尺寸 | 哈希位宽 |
/// |-----------|---------|---------|
/// | 8  | 8×8   | 64 bit   |
/// | 16 | 16×16 | 256 bit  |
/// | 32 | 32×32 | 1024 bit |
/// | 64 | 64×64 | 4096 bit |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HashSize(u32);

impl HashSize {
    pub fn new(size: u32) -> Self {
        Self(size)
    }

    pub fn size(self) -> u32 {
        self.0
    }

    pub const fn bits(self) -> u32 {
        self.0 * self.0
    }

    pub const fn u32s_per_image(self) -> u32 {
        self.bits().div_ceil(32)
    }

    pub const fn u64s_per_image(self) -> u32 {
        self.u32s_per_image().div_ceil(2)
    }

    pub const fn default() -> Self {
        Self(8)
    }
}

impl Default for HashSize {
    fn default() -> Self {
        Self::default()
    }
}

impl From<u32> for HashSize {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<HashSize> for u32 {
    fn from(v: HashSize) -> Self {
        v.0
    }
}

#[deprecated(since = "0.2.0", note = "使用 HashSize 替代，HashBits 将在未来版本移除")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashBits {
    B64 = 64,
    B128 = 128,
    B256 = 256,
}

#[allow(deprecated)]
impl HashBits {
    pub const fn default() -> Self { Self::B64 }
    pub const fn bits(self) -> u32 { self as u32 }
    pub const fn u32s_per_image(self) -> u32 { self.bits().div_ceil(32) }
}

#[allow(deprecated)]
impl From<HashBits> for HashSize {
    fn from(bits: HashBits) -> Self {
        match bits {
            HashBits::B64 | HashBits::B128 => HashSize::new(8),
            HashBits::B256 => HashSize::new(16),
        }
    }
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

    fn compute_sized(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        hash_size: HashSize,
    ) -> Result<Vec<u64>, GpuError>;

    fn pipeline(&self) -> &ComputePipeline;
    fn workgroup_size(&self) -> [u32; 3];
    fn hash_size(&self) -> HashSize;
}

/// 通用感知哈希 GPU 计算流程。
///
/// 封装了像素打包 → 缓冲区创建 → dispatch → 下载 → 解析的完整流水线，
/// 所有感知哈希算法均可复用此实现。
///
/// 根据管线的 Push Constant 支持情况自动选择参数传递方式：
/// - Push Constant 模式：参数通过 `set_push_constants` 传递，仅使用 2 个 binding
/// - Uniform 回退模式：参数通过 Uniform buffer 传递，使用 3 个 binding
#[allow(clippy::too_many_arguments)]
pub fn compute_phash(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
) -> Result<Vec<u64>, GpuError> {
    if images.is_empty() { return Ok(vec![]); }

    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let buffer_pool = ctx.buffer_pool();
    let image_count = images.len();
    let pixels_per_image = images[0].len();
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let u64s_per_image = u32s_per_image.div_ceil(2);

    for img in images {
        if img.len() != pixels_per_image {
            return Err(GpuError::InvalidInput("所有图像尺寸必须一致".to_string()));
        }
    }

    let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * pixels_per_image);
    for img in images {
        all_pixels.extend(crate::pixel_pack::pack_u8_to_u32(img));
    }

    let input_size = (all_pixels.len() * 4) as u64;
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let input_buffer_raw = buffer_pool.acquire(device, input_size, BufferUsage::Storage);
    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage);
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));
    queue.write_buffer(&output_buffer_raw, 0, &vec![0u8; output_size as usize]);

    let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };
    let params_arr = [params];
    let params_bytes = bytemuck::cast_slice::<PhashParams, u8>(&params_arr);

    let dispatch_x = width.div_ceil(workgroup_size[0]).max(1);
    let dispatch_y = height.div_ceil(workgroup_size[1]).max(1);
    let dispatch_z = image_count as u32;

    if pipeline.push_constant_size().is_some() {
        pipeline.dispatch(
            device, queue,
            &[&input_buffer, &output_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            Some(params_bytes),
            ctx.compute_units(),
        );
    } else {
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);
        pipeline.dispatch(
            device, queue,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            None,
            ctx.compute_units(),
        );
    }

    let result = output_buffer.download_with_pool(device, queue, buffer_pool)?;
    buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
    let mut hashes = Vec::with_capacity(image_count * u64s_per_image);
    for i in 0..image_count {
        let base = i * u32s_per_image;
        for chunk in 0..u64s_per_image {
            let lo_idx = base + chunk * 2;
            let low = raw_u32[lo_idx] as u64;
            let high = if lo_idx + 1 < base + u32s_per_image {
                raw_u32[lo_idx + 1] as u64
            } else { 0 };
            hashes.push(low | (high << 32));
        }
    }
    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（从 GPU buffer 输入，支持可变 hash_size）。
///
/// 根据管线的 Push Constant 支持情况自动选择参数传递方式。
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
    hash_size: HashSize,
) -> Result<Vec<u64>, GpuError> {
    if image_count == 0 { return Ok(vec![]); }

    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let buffer_pool = ctx.buffer_pool();
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let u64s_per_image = u32s_per_image.div_ceil(2);
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage);
    queue.write_buffer(&output_buffer_raw, 0, &vec![0u8; output_size as usize]);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };
    let params_arr = [params];
    let params_bytes = bytemuck::cast_slice::<PhashParams, u8>(&params_arr);

    let dispatch_x = width.div_ceil(workgroup_size[0]).max(1);
    let dispatch_y = height.div_ceil(workgroup_size[1]).max(1);
    let dispatch_z = image_count as u32;

    if pipeline.push_constant_size().is_some() {
        pipeline.dispatch(
            device, queue,
            &[input_buffer, &output_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            Some(params_bytes),
            ctx.compute_units(),
        );
    } else {
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);
        pipeline.dispatch(
            device, queue,
            &[input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            None,
            ctx.compute_units(),
        );
    }

    let result = output_buffer.download_with_pool(device, queue, buffer_pool)?;
    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
    let mut hashes = Vec::with_capacity(image_count * u64s_per_image);
    for i in 0..image_count {
        let base = i * u32s_per_image;
        for chunk in 0..u64s_per_image {
            let lo_idx = base + chunk * 2;
            let low = raw_u32[lo_idx] as u64;
            let high = if lo_idx + 1 < base + u32s_per_image {
                raw_u32[lo_idx + 1] as u64
            } else { 0 };
            hashes.push(low | (high << 32));
        }
    }
    Ok(hashes)
}

/// 创建默认感知哈希计算器结构体。
/// 封装 pipeline + workgroup_size + hash_size 字段和构造。
#[macro_export]
macro_rules! declare_phash_computer {
    ($name:ident, $doc:expr, $wgsl:ident, $wg_size:expr) => {
        #[doc = $doc]
        pub struct $name {
            pipeline: ::std::sync::Arc<$crate::pipeline::ComputePipeline>,
            workgroup_size: [u32; 3],
            hash_size: $crate::tasks::hash_common::HashSize,
        }

        impl $name {
            #[doc = concat!("创建 ", stringify!($name), "，默认 hash_size=8 (64 bit)")]
            pub fn new(ctx: &mut $crate::context::GpuContext) -> Result<Self, $crate::error::GpuError> {
                Self::with_config(ctx, $wg_size, $crate::tasks::hash_common::HashSize::default())
            }

            #[doc = concat!("创建 ", stringify!($name), "，指定 workgroup_size（默认 hash_size）")]
            pub fn with_workgroup_size(
                ctx: &mut $crate::context::GpuContext,
                workgroup_size: [u32; 3],
            ) -> Result<Self, $crate::error::GpuError> {
                Self::with_config(ctx, workgroup_size, $crate::tasks::hash_common::HashSize::default())
            }

            #[doc = concat!("创建 ", stringify!($name), "，指定 workgroup_size 和 hash_size")]
            pub fn with_config(
                ctx: &mut $crate::context::GpuContext,
                workgroup_size: [u32; 3],
                hash_size: $crate::tasks::hash_common::HashSize,
            ) -> Result<Self, $crate::error::GpuError> {
                let pipeline = ctx.get_or_create_pipeline(
                    &$crate::tasks::hash_common::phash_pipeline_descriptor(
                        $wgsl, workgroup_size, ctx.push_constants_supported(),
                    ),
                )?;
                Ok(Self {
                    pipeline: ::std::sync::Arc::clone(&pipeline),
                    workgroup_size,
                    hash_size,
                })
            }

            pub fn pipeline(&self) -> &$crate::pipeline::ComputePipeline { &self.pipeline }
            pub fn workgroup_size_val(&self) -> [u32; 3] { self.workgroup_size }
            pub fn hash_size_val(&self) -> $crate::tasks::hash_common::HashSize { self.hash_size }
        }
    };
}

/// 简化：为感知哈希算法生成 PerceptualHashComputer trait 实现。
#[macro_export]
macro_rules! impl_phash_computer_simple {
    ($name:ident) => {
        impl $crate::tasks::hash_common::PerceptualHashComputer for $name {
            fn compute(&self, ctx: &$crate::context::GpuContext, images: &[Vec<u8>]) -> Result<Vec<u64>, $crate::error::GpuError> {
                self.compute_sized(ctx, images, self.hash_size)
            }

            fn compute_sized(&self, ctx: &$crate::context::GpuContext, images: &[Vec<u8>], hash_size: $crate::tasks::hash_common::HashSize) -> Result<Vec<u64>, $crate::error::GpuError> {
                if images.is_empty() { return Ok(vec![]); }
                let pixels_per_image = images[0].len();
                let width = (pixels_per_image as f64).sqrt() as u32;
                let height = width;
                $crate::tasks::hash_common::compute_phash(
                    &self.pipeline, ctx, images, width, height, self.workgroup_size, hash_size,
                )
            }

            fn pipeline(&self) -> &$crate::pipeline::ComputePipeline { &self.pipeline }
            fn workgroup_size(&self) -> [u32; 3] { self.workgroup_size }
            fn hash_size(&self) -> $crate::tasks::hash_common::HashSize { self.hash_size }
        }
    };
}

/// 为需要自定义 width/height 推断的感知哈希算法生成 PerceptualHashComputer 实现。
#[macro_export]
macro_rules! impl_phash_computer_custom_dims {
    ($name:ident, $mode:expr) => {
        impl $crate::tasks::hash_common::PerceptualHashComputer for $name {
            fn compute(&self, ctx: &$crate::context::GpuContext, images: &[Vec<u8>]) -> Result<Vec<u64>, $crate::error::GpuError> {
                self.compute_sized(ctx, images, self.hash_size)
            }

            fn compute_sized(&self, ctx: &$crate::context::GpuContext, images: &[Vec<u8>], hash_size: $crate::tasks::hash_common::HashSize) -> Result<Vec<u64>, $crate::error::GpuError> {
                if images.is_empty() { return Ok(vec![]); }
                let s = hash_size.size();
                let (width, height) = match $mode {
                    0 => (s, s + 1),
                    1 => (s + 1, s),
                    _ => (s, s),
                };
                $crate::tasks::hash_common::compute_phash(
                    &self.pipeline, ctx, images, width, height, self.workgroup_size, hash_size,
                )
            }

            fn pipeline(&self) -> &$crate::pipeline::ComputePipeline { &self.pipeline }
            fn workgroup_size(&self) -> [u32; 3] { self.workgroup_size }
            fn hash_size(&self) -> $crate::tasks::hash_common::HashSize { self.hash_size }
        }
    };
}
