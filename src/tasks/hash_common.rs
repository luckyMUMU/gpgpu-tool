use crate::batch::GpuBatchSubmitter;
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor, wgsl_push_constant_to_uniform};
// 引入默认批量大小常量，用于 compute_phash 内部按显存预算自动分批
use crate::tasks::phasher::DEFAULT_MAX_BATCH_SIZE;

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

    /// 从 czkawka 的 `hash_size` 参数（u8）创建 `HashSize`。
    ///
    /// czkawka 使用 `hash_size ∈ {8, 16, 32, 64}` 表示网格边长，
    /// 对应哈希位宽 64/256/1024/4096 bit。
    ///
    /// # 错误
    ///
    /// `hash_size` 不在 `{8, 16, 32, 64}` 范围内时返回 `Err(GpuError::InvalidInput)`。
    pub fn from_czkawka(hash_size: u8) -> Result<Self, GpuError> {
        match hash_size {
            8 | 16 | 32 | 64 => Ok(Self(hash_size as u32)),
            _ => Err(GpuError::InvalidInput(format!(
                "czkawka hash_size 必须为 8/16/32/64，收到 {}",
                hash_size
            ))),
        }
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

/// 将感知哈希 dispatch 编码到共享 encoder 中（不提交）。
///
/// 用于融合管线场景，将 hash dispatch 编码到已有的 encoder 中，
/// 避免创建额外的 encoder 和 submit。
#[allow(clippy::too_many_arguments)]
pub fn encode_hash_dispatch(
    pipeline: &ComputePipeline,
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    params: &PhashParams,
    input_buffer: &GpuBuffer,
    output_buffer: &GpuBuffer,
    workgroup_size: [u32; 3],
    compute_units: u32,
) {
    let params_arr = [*params];
    let params_bytes = bytemuck::cast_slice::<PhashParams, u8>(&params_arr);

    let dispatch_x = params.width.div_ceil(workgroup_size[0]).max(1);
    let dispatch_y = params.height.div_ceil(workgroup_size[1]).max(1);
    let dispatch_z = params.image_count;

    pipeline.encode_dispatch_with_params_into(
        device,
        encoder,
        params_bytes,
        &[input_buffer, output_buffer],
        [dispatch_x, dispatch_y, dispatch_z],
        compute_units,
    );
}

/// 将 GPU 输出的原始 u32 字节解析为 u64 哈希值列表。
///
/// 每个图像的哈希由 `u32s_per_image` 个 u32 组成，两两配对合并为 u64
/// （低 32 位在前，高 32 位在后）。当 u32 数量为奇数时，最高位补零。
pub fn parse_hash_results(raw: &[u8], image_count: usize, hash_size: HashSize) -> Vec<u64> {
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let u64s_per_image = u32s_per_image.div_ceil(2);
    let raw_u32 = bytemuck::cast_slice::<u8, u32>(raw);
    let mut hashes = Vec::with_capacity(image_count * u64s_per_image);
    for i in 0..image_count {
        let base = i * u32s_per_image;
        for chunk in 0..u64s_per_image {
            let lo_idx = base + chunk * 2;
            let low = raw_u32[lo_idx] as u64;
            let high = if lo_idx + 1 < base + u32s_per_image {
                raw_u32[lo_idx + 1] as u64
            } else {
                0
            };
            hashes.push(low | (high << 32));
        }
    }
    hashes
}

/// 执行感知哈希 dispatch 并下载解析结果。
///
/// 封装了参数编码 → dispatch_with_params → 下载 → u32→u64 解析的公共流程，
/// 消除 `compute_phash` 和 `compute_phash_from_gpu_buffer` 之间的重复逻辑。
///
/// `output_zeroed` 为 true 时，在 dispatch 前通过 GPU fill_buffer 清零输出缓冲区
/// （替代 CPU 端 `vec![0u8; N]` + `queue.write_buffer` 的方式）。
#[allow(clippy::too_many_arguments)]
fn dispatch_and_parse_hash(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    params: &PhashParams,
    input_buffer: &GpuBuffer,
    output_buffer: &GpuBuffer,
    workgroup_size: [u32; 3],
    image_count: usize,
    hash_size: HashSize,
    output_zeroed: bool,
) -> Result<Vec<u64>, GpuError> {
    let device = ctx.device()?;
    let queue = ctx.queue()?;

    // GPU 端清零输出缓冲区（替代 CPU 端分配 + 上传零向量）
    // hash 着色器使用 atomicOr（只置位不清零），缓冲区池复用的缓冲区可能残留旧数据
    if output_zeroed {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hash_zero_encoder"),
        });
        encoder.clear_buffer(
            output_buffer.raw(),
            0,
            Some(output_buffer.size()),
        );
        queue.submit(std::iter::once(encoder.finish()));
    }

    let params_arr = [*params];
    let params_bytes = bytemuck::cast_slice::<PhashParams, u8>(&params_arr);

    let dispatch_x = params.width.div_ceil(workgroup_size[0]).max(1);
    let dispatch_y = params.height.div_ceil(workgroup_size[1]).max(1);
    let dispatch_z = image_count as u32;

    pipeline.dispatch_with_params(
        device,
        queue,
        params_bytes,
        &[input_buffer, output_buffer],
        [dispatch_x, dispatch_y, dispatch_z],
        ctx.compute_units(),
    );

    let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;
    Ok(parse_hash_results(&result, image_count, hash_size))
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
    if images.is_empty() {
        return Ok(vec![]);
    }

    let pixels_per_image = images[0].len();
    // u32 对齐后每张图的字节数（每像素 1 字节 u8 → 打包为 1 个 u32 = 4 字节）
    let per_image_bytes = (pixels_per_image * 4) as u64;
    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

    // 单图超限直接报错（无法通过分批解决）
    if per_image_bytes > max_binding {
        return Err(GpuError::InvalidInput(format!(
            "单图尺寸 {} 字节超过 max_storage_buffer_binding_size {} 字节",
            per_image_bytes, max_binding
        )));
    }

    // 校验所有图像尺寸一致
    for img in images {
        if img.len() != pixels_per_image {
            return Err(GpuError::InvalidInput("所有图像尺寸必须一致".to_string()));
        }
    }

    // 按显存预算计算单批最大图像数（安全系数：DEFAULT_MAX_BATCH_SIZE 已含余量）
    let max_batch = if per_image_bytes > 0 {
        ((DEFAULT_MAX_BATCH_SIZE / per_image_bytes) as usize).max(1)
    } else {
        images.len()
    };

    // 单批可容纳全部图像：直接走原路径（无分批开销）
    if max_batch >= images.len() {
        return compute_phash_single_batch(
            pipeline, ctx, images, width, height, workgroup_size, hash_size,
        );
    }

    // 分批处理：循环 chunks，合并结果
    let mut all_hashes = Vec::with_capacity(images.len());
    for chunk in images.chunks(max_batch) {
        let chunk_hashes = compute_phash_single_batch(
            pipeline, ctx, chunk, width, height, workgroup_size, hash_size,
        )?;
        all_hashes.extend(chunk_hashes);
    }
    Ok(all_hashes)
}

/// 单批感知哈希 GPU 计算流程（原 compute_phash 的核心逻辑）。
///
/// 不含分批逻辑，调用方需确保 `images` 总大小不超过显存预算。
/// 由 [`compute_phash`] 外层按显存预算分批后调用。
#[allow(clippy::too_many_arguments)]
fn compute_phash_single_batch(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
) -> Result<Vec<u64>, GpuError> {
    if images.is_empty() {
        return Ok(vec![]);
    }

    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let buffer_pool = ctx.buffer_pool();
    let image_count = images.len();
    let pixels_per_image = images[0].len();
    let u32s_per_image = hash_size.u32s_per_image() as usize;

    let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * pixels_per_image);
    crate::pixel_pack::pack_u8_batch_to_u32(images, &mut all_pixels);

    let input_size = (all_pixels.len() * 4) as u64;
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let input_buffer_raw = buffer_pool.acquire(device, input_size, BufferUsage::Storage)?;
    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage)?;
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));

    let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };

    let hashes = dispatch_and_parse_hash(
        pipeline, ctx, &params, &input_buffer, &output_buffer,
        workgroup_size, image_count, hash_size, true,
    )?;

    buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（带预计算阈值）。
///
/// 与 [`compute_phash`] 功能相同，但在每张图的像素数据末尾追加一个 u32 阈值。
/// 着色器从 `pixels[base + pixels_per_image]` 读取阈值，避免 GPU 端 O(N²) 循环
/// 计算全局统计量（均值/中位数），解决大 hash_size 下 DX12 着色器编译失败的问题。
///
/// 输入缓冲区布局（每张图）：`[pixels_per_image 个 u32 像素, 1 个 u32 阈值]`
/// 步长 stride = pixels_per_image + 1。
///
/// # 参数
///
/// - `thresholds` — 每张图的预计算阈值（长度必须等于 `images.len()`）
/// - 其余参数同 [`compute_phash`]
#[allow(clippy::too_many_arguments)]
pub fn compute_phash_with_thresholds(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    images: &[Vec<u8>],
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
    thresholds: &[u32],
) -> Result<Vec<u64>, GpuError> {
    if images.is_empty() {
        return Ok(vec![]);
    }
    if thresholds.len() != images.len() {
        return Err(GpuError::InvalidInput(
            "阈值数量必须等于图像数量".to_string(),
        ));
    }

    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let buffer_pool = ctx.buffer_pool();
    let image_count = images.len();
    let pixels_per_image = images[0].len();
    let u32s_per_image = hash_size.u32s_per_image() as usize;

    for img in images {
        if img.len() != pixels_per_image {
            return Err(GpuError::InvalidInput("所有图像尺寸必须一致".to_string()));
        }
    }

    // 像素打包 + 每张图追加 1 个 u32 阈值
    let stride = pixels_per_image + 1; // 每张图的 u32 步长（像素 + 阈值）
    let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * stride);
    crate::pixel_pack::pack_u8_batch_to_u32(images, &mut all_pixels);

    // 在每张图的像素数据后插入阈值
    let mut extended_pixels = Vec::with_capacity(image_count * stride);
    for i in 0..image_count {
        let start = i * pixels_per_image;
        let end = start + pixels_per_image;
        extended_pixels.extend_from_slice(&all_pixels[start..end]);
        extended_pixels.push(thresholds[i]);
    }

    let input_size = (extended_pixels.len() * 4) as u64;
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let input_buffer_raw = buffer_pool.acquire(device, input_size, BufferUsage::Storage)?;
    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage)?;
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&extended_pixels));

    let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };

    let hashes = dispatch_and_parse_hash(
        pipeline, ctx, &params, &input_buffer, &output_buffer,
        workgroup_size, image_count, hash_size, true,
    )?;

    buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（从 GPU buffer 输入，带预计算阈值）。
///
/// 与 [`compute_phash_from_gpu_buffer`] 功能相同，但在每张图的像素数据末尾追加一个 u32 阈值。
/// 用于 Mean Hash 和 Median Hash 的零拷贝管线场景：先从 GPU buffer 下载像素数据到 CPU，
/// 计算阈值，再创建扩展输入缓冲区（像素 + 阈值）进行 dispatch。
///
/// # 参数
///
/// - `thresholds` — 每张图的预计算阈值（长度必须等于 `image_count`）
/// - 其余参数同 [`compute_phash_from_gpu_buffer`]
#[allow(clippy::too_many_arguments)]
pub fn compute_phash_from_gpu_buffer_with_thresholds(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    input_buffer: &GpuBuffer,
    _u32_count: usize,
    image_count: usize,
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
    thresholds: &[u32],
) -> Result<Vec<u64>, GpuError> {
    if image_count == 0 {
        return Ok(vec![]);
    }
    if thresholds.len() != image_count {
        return Err(GpuError::InvalidInput(
            "阈值数量必须等于图像数量".to_string(),
        ));
    }

    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let buffer_pool = ctx.buffer_pool();

    // 从 GPU buffer 下载像素数据
    let raw_pixels = input_buffer.download_with_pool(device, queue, buffer_pool)?;
    let raw_u32: &[u32] = bytemuck::cast_slice(&raw_pixels);

    let pixels_per_image = (width * height) as usize;
    let stride = pixels_per_image + 1; // 像素 + 阈值

    // 构建扩展输入缓冲区：每张图追加 1 个 u32 阈值
    let mut extended_pixels = Vec::with_capacity(image_count * stride);
    for i in 0..image_count {
        let base = i * pixels_per_image;
        let end = (base + pixels_per_image).min(raw_u32.len());
        extended_pixels.extend_from_slice(&raw_u32[base..end]);
        extended_pixels.push(thresholds[i]);
    }

    let input_size = (extended_pixels.len() * 4) as u64;
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let input_buffer_raw = buffer_pool.acquire(device, input_size, BufferUsage::Storage)?;
    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage)?;
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&extended_pixels));

    let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };

    let hashes = dispatch_and_parse_hash(
        pipeline, ctx, &params, &input_buffer, &output_buffer,
        workgroup_size, image_count, hash_size, true,
    )?;

    buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（从 GPU buffer 输入，支持可变 hash_size）。
///
/// 根据管线的 Push Constant 支持情况自动选择参数传递方式。
///
/// 注：输入 buffer 由调用方提供，调用方需确保其大小不超过 `max_storage_buffer_binding_size`。
/// 输出 buffer 大小 = `image_count * u32s_per_image * 4`，本函数会预检查是否超限。
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
    if image_count == 0 {
        return Ok(vec![]);
    }

    let device = ctx.device()?;
    let buffer_pool = ctx.buffer_pool();
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let output_size = (image_count * u32s_per_image * 4) as u64;
    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;

    // 输出 buffer 超限预检查（输入 buffer 由调用方负责）
    if output_size > max_binding {
        return Err(GpuError::InvalidInput(format!(
            "输出缓冲区大小 {} 字节超过 max_storage_buffer_binding_size {} 字节（image_count={}, u32s_per_image={}）",
            output_size, max_binding, image_count, u32s_per_image
        )));
    }

    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage)?;
    let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

    let params = PhashParams {
        image_count: image_count as u32,
        width,
        height,
        hash_size: hash_size.size(),
    };

    let hashes = dispatch_and_parse_hash(
        pipeline, ctx, &params, input_buffer, &output_buffer,
        workgroup_size, image_count, hash_size, true,
    )?;

    buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

    Ok(hashes)
}

/// 通用感知哈希 GPU 计算流程（批量编码模式）。
///
/// 将 hash dispatch 命令编码到 `GpuBatchSubmitter` 的共享 CommandEncoder 中，
/// 并设置 staging buffer 用于结果收集。
/// 调用方需在 `batch.wait_all()` 后手动释放 output_buffer 并解析结果。
///
/// # 返回
///
/// `(output_buffer, image_count, hash_size)` — 输出缓冲区、图像数量和哈希尺寸，
/// 用于后续结果解析和缓冲区释放。
#[allow(clippy::too_many_arguments)]
pub fn compute_phash_from_gpu_buffer_batch(
    pipeline: &ComputePipeline,
    ctx: &GpuContext,
    batch: &mut GpuBatchSubmitter,
    input_buffer: &GpuBuffer,
    image_count: usize,
    width: u32,
    height: u32,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
) -> Result<(GpuBuffer, usize, HashSize), GpuError> {
    if image_count == 0 {
        return Err(GpuError::InvalidInput("图像数量为 0".to_string()));
    }

    let device = ctx.device()?;
    let buffer_pool = ctx.buffer_pool();
    let u32s_per_image = hash_size.u32s_per_image() as usize;
    let output_size = (image_count * u32s_per_image * 4) as u64;

    let output_buffer_raw = buffer_pool.acquire(device, output_size, BufferUsage::Storage)?;
    // 清零输出缓冲区：hash 着色器使用 atomicOr（只置位不清零），
    // 缓冲区池复用的缓冲区可能残留旧数据导致哈希结果错误。
    ctx.queue()?.write_buffer(&output_buffer_raw, 0, &vec![0u8; output_size as usize]);
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

    // 根据管线配置选择 Push Constant 或 Uniform buffer
    if pipeline.push_constant_size().is_some() {
        batch.encode_dispatch(
            ctx,
            pipeline,
            &[input_buffer, &output_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            Some(params_bytes),
            ctx.compute_units(),
        )?;
    } else {
        let params_buffer = GpuBuffer::from_bytes(device, params_bytes, BufferUsage::Uniform);
        batch.encode_dispatch(
            ctx,
            pipeline,
            &[input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, dispatch_y, dispatch_z],
            None,
            ctx.compute_units(),
        )?;
    }

    // 设置 staging buffer 用于结果收集
    let staging = buffer_pool.acquire_staging(device, output_size);
    batch.encode_copy(
        ctx,
        output_buffer.raw(),
        0,
        &staging,
        0,
        output_size,
    )?;
    batch.add_staging_buffer(staging);

    Ok((output_buffer, image_count, hash_size))
}

/// 解析批量哈希结果。
///
/// 从 `wait_all()` 返回的原始字节中解析 u64 哈希值。
pub fn parse_batch_hash_results(
    raw_data: &[u8],
    image_count: usize,
    hash_size: HashSize,
) -> Vec<u64> {
    parse_hash_results(raw_data, image_count, hash_size)
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
