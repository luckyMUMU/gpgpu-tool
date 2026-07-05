use crate::backend_dispatcher::{BackendDispatcher, DefaultBackendDispatcher};
use crate::batch::GpuBatchSubmitter;
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::gaussian_blur::GpuGaussianBlur;
use crate::tasks::hash_bytes::HashBytes;
use crate::tasks::hash_common::{HashSize, PerceptualHashComputer, compute_phash_from_gpu_buffer, compute_phash_from_gpu_buffer_with_thresholds};
use crate::tasks::mean_hash::MeanHashComputer;
use crate::tasks::median_hash::MedianHashComputer;
use crate::tasks::gradient_hash::GradientHashComputer;
use crate::tasks::block_hash::BlockHashComputer;
use crate::tasks::vert_gradient_hash::VertGradientHashComputer;
use crate::tasks::double_gradient_hash::DoubleGradientHashComputer;
use crate::tasks::gpu_resize::{GpuResize, GpuResizeConfig, ResizeFilter};
use crate::tasks::phasher_util::{merge_gpu_buffers, merge_gpu_buffers_batch, resize_grayscale, upload_image_to_gpu, upload_packed_to_gpu};
#[cfg(feature = "cpu-fallback")]
use crate::tasks::phasher_cpu::{PHasherCpu, cpu_gaussian_blur};
#[cfg(feature = "pdq")]
use crate::tasks::pdq_hash::PdqHashGpu;

/// 感知哈希算法类型。
///
/// # 算法选择指南
///
/// | 算法 | 最佳用途 | 速度 | 亮度鲁棒性 | 缩放鲁棒性 |
/// |------|----------|------|-----------|-----------|
/// | [`Mean`](HashAlgorithm::Mean) | 精确去重 | 最快 | ❌ 差 | ✅ 好 |
/// | [`Median`](HashAlgorithm::Median) | 去重（抗异常值） | 快 | ❌ 差 | ✅ 好 |
/// | [`Gradient`](HashAlgorithm::Gradient) | 相似图像搜索 | 快 | ✅ 好 | ✅ 好 |
/// | [`VertGradient`](HashAlgorithm::VertGradient) | 竖屏/文字图像 | 快 | ✅ 好 | ✅ 好 |
/// | [`DoubleGradient`](HashAlgorithm::DoubleGradient) | 通用相似搜索 | 中 | ✅ 好 | ✅ 好 |
/// | [`Block`](HashAlgorithm::Block) | 精细匹配 | 中 | ❌ 差 | ✅ 好 |
/// | [`Pdq`](HashAlgorithm::Pdq) | 专业取证 | 慢 | ✅ 最好 | ✅ 最好 |
///
/// # 推荐用例
///
/// - **`"deduplication"`**（精确去重）：[`Mean`](HashAlgorithm::Mean) 或 [`Median`](HashAlgorithm::Median)
/// - **`"near-duplicate"`**（近似重复检测）：[`Gradient`](HashAlgorithm::Gradient) 或 [`DoubleGradient`](HashAlgorithm::DoubleGradient)
/// - **`"general"`**（通用相似图像搜索）：[`Gradient`](HashAlgorithm::Gradient)（推荐默认）
/// - **`"forensic"`**（取证/精确匹配）：[`Pdq`](HashAlgorithm::Pdq)（需 `pdq` feature）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    Mean,
    Median,
    Gradient,
    Block,
    VertGradient,
    DoubleGradient,
    #[cfg(feature = "pdq")]
    Pdq,
}

impl HashAlgorithm {
    /// 从 czkawka 字符串名称创建算法枚举。
    ///
    /// 兼容 `"Blockhash"` 和 `"Block"` 两种命名（czkawka 使用 `Blockhash`，
    /// 本工具包使用 `Block`）。
    ///
    /// # 示例
    ///
    /// ```
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    ///
    /// assert_eq!(HashAlgorithm::from_czkawka("Blockhash"), Some(HashAlgorithm::Block));
    /// assert_eq!(HashAlgorithm::from_czkawka("Block"), Some(HashAlgorithm::Block));
    /// assert_eq!(HashAlgorithm::from_czkawka("Gradient"), Some(HashAlgorithm::Gradient));
    /// assert_eq!(HashAlgorithm::from_czkawka("Unknown"), None);
    /// ```
    pub fn from_czkawka(name: &str) -> Option<Self> {
        match name {
            "Mean" => Some(HashAlgorithm::Mean),
            "Median" => Some(HashAlgorithm::Median),
            "Gradient" => Some(HashAlgorithm::Gradient),
            "Block" | "Blockhash" => Some(HashAlgorithm::Block),
            "VertGradient" => Some(HashAlgorithm::VertGradient),
            "DoubleGradient" => Some(HashAlgorithm::DoubleGradient),
            _ => None,
        }
    }

    /// 转换为 czkawka 字符串名称。
    ///
    /// `Block` → `"Blockhash"`，其余与枚举变体名一致。
    ///
    /// 可用于与 `img_hash::HashAlg` 互转：
    /// ```ignore
    /// // img_hash -> 本工具包
    /// let alg = HashAlgorithm::from_czkawka(&format!("{:?}", img_hash_alg));
    /// // 本工具包 -> img_hash
    /// let img_hash_alg: img_hash::HashAlg =
    ///     format!("{:?}", hasher_alg.to_czkawka_name()).parse().unwrap();
    /// ```
    pub fn to_czkawka_name(&self) -> &'static str {
        match self {
            HashAlgorithm::Mean => "Mean",
            HashAlgorithm::Median => "Median",
            HashAlgorithm::Gradient => "Gradient",
            HashAlgorithm::Block => "Blockhash",
            HashAlgorithm::VertGradient => "VertGradient",
            HashAlgorithm::DoubleGradient => "DoubleGradient",
            #[cfg(feature = "pdq")]
            HashAlgorithm::Pdq => "Pdq",
        }
    }

    /// 根据用例推荐合适的哈希算法。
    ///
    /// # 参数
    ///
    /// - `"deduplication"` — 精确去重，推荐 Mean/Median
    /// - `"near-duplicate"` — 近似重复检测，推荐 Gradient/DoubleGradient
    /// - `"general"` — 通用相似图像搜索，推荐 Gradient（默认）
    /// - `"forensic"` — 取证/精确匹配，推荐 PDQ（需 `pdq` feature）
    pub fn recommended_for(use_case: &str) -> Vec<HashAlgorithm> {
        match use_case {
            "deduplication" => vec![HashAlgorithm::Mean, HashAlgorithm::Median],
            "near-duplicate" => vec![HashAlgorithm::Gradient, HashAlgorithm::DoubleGradient],
            "forensic" => {
                #[cfg(feature = "pdq")]
                { vec![HashAlgorithm::Pdq] }
                #[cfg(not(feature = "pdq"))]
                { vec![HashAlgorithm::DoubleGradient] }
            }
            _ => vec![HashAlgorithm::Gradient],
        }
    }
}

impl HashAlgorithm {
    pub fn target_size_for(&self, hash_size: HashSize) -> (u32, u32) {
        let s = hash_size.size();
        match self {
            HashAlgorithm::Mean | HashAlgorithm::Median | HashAlgorithm::Block => (s, s),
            HashAlgorithm::Gradient => (s, s + 1),
            HashAlgorithm::VertGradient => (s + 1, s),
            HashAlgorithm::DoubleGradient => (s + 1, s + 1),
            #[cfg(feature = "pdq")]
            HashAlgorithm::Pdq => (64, 64),
        }
    }
}

/// 感知哈希计算器，支持任意尺寸图像输入。
///
/// 内部自动将图像缩放到算法所需的目标尺寸后计算哈希。
/// 持有 GPU 计算器实例，避免重复创建管线。
///
/// # 三阶段流水线
///
/// 感知哈希计算分为三个独立阶段，可分别调用：
///
/// 1. **预处理**（[`preprocess_gpu`](Self::preprocess_gpu) / [`preprocess_cpu`](Self::preprocess_cpu)）：
///    上传图像到 GPU，可选执行高斯模糊降噪
/// 2. **缩放**（[`resize_gpu`](Self::resize_gpu) / [`resize_cpu`](Self::resize_cpu)）：
///    将图像缩放到算法所需的目标尺寸
/// 3. **哈希计算**（[`compute_hash_gpu`](Self::compute_hash_gpu) / [`compute_hash_cpu`](Self::compute_hash_cpu)）：
///    从缩放后的数据计算感知哈希值
///
/// [`compute()`](Self::compute) 方法自动编排三阶段流水线，
/// 也可独立调用各阶段实现自定义流水线。
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::GpuContext;
/// use gpgpu_tool::tasks::phasher::{PerceptualHasher, HashAlgorithm};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
///
/// // 传入任意尺寸的灰度图像
/// let images = vec![vec![128u8; 256 * 256]]; // 一张 256x256 灰度图
/// let dimensions = vec![(256u32, 256u32)];
/// let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
/// ```
pub struct PerceptualHasher {
    pub(crate) algorithm: HashAlgorithm,
    pub(crate) target_width: u32,
    pub(crate) target_height: u32,
    pub(crate) computer: Box<dyn PerceptualHashComputer>,
    pub(crate) gpu_resize: Option<GpuResize>,
    pub(crate) gpu_blur: Option<GpuGaussianBlur>,
    pub(crate) blur_sigma: Option<f32>,
    pub(crate) blur_kernel_size: Option<u32>,
    pub(crate) hash_size: HashSize,
    pub(crate) max_batch_size: u64,
    #[cfg(feature = "cpu-fallback")]
    pub(crate) cpu_hasher: PHasherCpu,
}

/// 感知哈希计算器的默认 workgroup 大小。
pub const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [8, 8, 1];
/// 感知哈希批次处理默认最大缓冲区字节数（128 MB）。
pub const DEFAULT_MAX_BATCH_SIZE: u64 = 128 * 1024 * 1024;

impl PerceptualHasher {
    // ==================== 构造方法 ====================

    /// 创建感知哈希计算器（默认 64bit，CPU 缩放，默认 workgroup_size）。
    pub fn new(ctx: &mut GpuContext, algorithm: HashAlgorithm) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, false, HashSize::default(), DEFAULT_WORKGROUP_SIZE, None, None)
    }

    /// 创建感知哈希计算器，指定哈希位长和是否 GPU 缩放。
    pub fn with_resize_mode(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, use_gpu_resize, HashSize::default(), DEFAULT_WORKGROUP_SIZE, None, None)
    }

    /// 创建感知哈希计算器，指定 hash_size（CPU 缩放，默认 workgroup_size）。
    pub fn with_hash_size(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, false, hash_size, DEFAULT_WORKGROUP_SIZE, None, None)
    }

    /// 创建感知哈希计算器，完整配置（所有参数均可定制）。
    pub fn with_config(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, use_gpu_resize, hash_size, DEFAULT_WORKGROUP_SIZE, None, None)
    }

    /// 创建感知哈希计算器，启用 GPU Lanczos3 高质量缩放。
    ///
    /// 启用后图像缩放使用 GPU Lanczos3（2-pass 可分离卷积），
    /// 在保持锐度的同时利用 GPU 并行加速。
    /// 与 CPU Lanczos3 相比，GPU 版本在大批量场景下有 2-5x 加速。
    ///
    /// `compute_images()` 方法会自动使用 GPU Lanczos3 路径。
    /// `compute()` 方法在 GPU 可用时也使用 GPU Lanczos3 缩放。
    pub fn with_lanczos3(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
    ) -> Result<Self, GpuError> {
        Self::with_full_config_and_gpu_resize(
            ctx, algorithm, true, HashSize::default(), DEFAULT_WORKGROUP_SIZE,
            GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
            None, None,
        )
    }

    /// 创建感知哈希计算器，启用 GPU Lanczos3 缩放和自定义哈希位长。
    ///
    /// 与 [`with_lanczos3`](Self::with_lanczos3) 相同，但允许指定 `hash_size`。
    pub fn with_lanczos3_and_hash_size(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        Self::with_full_config_and_gpu_resize(
            ctx, algorithm, true, hash_size, DEFAULT_WORKGROUP_SIZE,
            GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
            None, None,
        )
    }

    /// 创建感知哈希计算器，启用高斯模糊预处理。
    ///
    /// 启用模糊后自动启用 GPU 缩放，实现 blur → resize → hash 零拷贝 GPU 流水线。
    /// `sigma` 为高斯核标准差，`kernel_size` 为核大小（必须为正奇数，最大 11）。
    pub fn with_blur(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        sigma: f32,
        kernel_size: u32,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(
            ctx, algorithm, true, HashSize::default(), DEFAULT_WORKGROUP_SIZE,
            Some(sigma), Some(kernel_size),
        )
    }

    /// 创建感知哈希计算器，完整配置（包含 workgroup_size 和可选高斯模糊）。
    ///
    /// `blur_sigma` 和 `blur_kernel_size` 同时为 `Some` 时启用高斯模糊预处理，
    /// 此时自动启用 GPU 缩放以实现零拷贝流水线。
    pub fn with_full_config(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
        workgroup_size: [u32; 3],
        blur_sigma: Option<f32>,
        blur_kernel_size: Option<u32>,
    ) -> Result<Self, GpuError> {
        Self::with_full_config_and_gpu_resize(
            ctx, algorithm, use_gpu_resize, hash_size, workgroup_size,
            GpuResizeConfig::default(), blur_sigma, blur_kernel_size,
        )
    }

    /// 创建感知哈希计算器，完全自定义配置。
    ///
    /// 允许同时指定 workgroup_size、hash_size、GpuResizeConfig 和可选高斯模糊。
    /// 如果 `use_gpu_resize` 为 true，`gpu_resize_config` 决定 GPU 缩放器的行为。
    /// `blur_sigma` 和 `blur_kernel_size` 同时为 `Some` 时启用高斯模糊预处理，
    /// 此时自动启用 GPU 缩放以实现零拷贝流水线。
    #[allow(clippy::too_many_arguments)]
    pub fn with_full_config_and_gpu_resize(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
        workgroup_size: [u32; 3],
        gpu_resize_config: GpuResizeConfig,
        blur_sigma: Option<f32>,
        blur_kernel_size: Option<u32>,
    ) -> Result<Self, GpuError> {
        let blur_enabled = blur_sigma.is_some() && blur_kernel_size.is_some();
        let effective_use_gpu_resize = use_gpu_resize || blur_enabled;

        let (w, h) = algorithm.target_size_for(hash_size);
        let computer: Box<dyn PerceptualHashComputer> = match algorithm {
            HashAlgorithm::Mean => Box::new(MeanHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Median => Box::new(MedianHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Gradient => Box::new(GradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Block => Box::new(BlockHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::VertGradient => Box::new(VertGradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::DoubleGradient => Box::new(DoubleGradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            #[cfg(feature = "pdq")]
            HashAlgorithm::Pdq => {
                Box::new(PdqHashGpu::new(ctx)?)
            }
        };
        let max_batch_size = gpu_resize_config.max_buffer_size / 2;
        let gpu_resize = if effective_use_gpu_resize {
            Some(GpuResize::with_config(ctx, gpu_resize_config)?)
        } else {
            None
        };
        let gpu_blur = if blur_enabled {
            Some(GpuGaussianBlur::new(ctx)?)
        } else {
            None
        };
        Ok(Self {
            algorithm,
            target_width: w,
            target_height: h,
            computer,
            gpu_resize,
            gpu_blur,
            blur_sigma,
            blur_kernel_size,
            hash_size,
            max_batch_size,
            #[cfg(feature = "cpu-fallback")]
            cpu_hasher: PHasherCpu::with_hash_size(algorithm, hash_size),
        })
    }

    // ==================== 访问器 ====================

    /// 返回哈希位长配置。
    pub fn hash_size(&self) -> HashSize { self.hash_size }
    /// 返回算法类型。
    pub fn algorithm(&self) -> HashAlgorithm { self.algorithm }
    /// 返回目标缩放尺寸。
    pub fn target_size(&self) -> (u32, u32) { (self.target_width, self.target_height) }

    // ==================== 阶段一：预处理 ====================

    /// GPU 预处理：上传图像到 GPU 并可选执行高斯模糊。
    ///
    /// 将每张图像的灰度像素数据上传到 GPU 存储缓冲区。
    /// 如果构造时启用了高斯模糊（[`with_blur`](Self::with_blur)），
    /// 会对每张图像执行模糊降噪处理。
    ///
    /// 返回每个图像对应的 `GpuBuffer`（已打包为 u32 像素）。
    /// 调用方负责在使用完毕后释放返回的缓冲区：
    /// ```ignore
    /// for buf in gpu_images {
    ///     ctx.buffer_pool().release(buf.into_raw(), BufferUsage::Storage);
    /// }
    /// ```
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `images` — 灰度像素数据，每个元素为一张图像的宽×高字节
    /// - `dimensions` — 对应图像的 (width, height)
    pub fn preprocess_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<GpuBuffer>, GpuError> {
        if let (Some(ref gpu_blur), Some(sigma), Some(kernel_size)) =
            (&self.gpu_blur, self.blur_sigma, self.blur_kernel_size)
        {
            let mut buffers = Vec::with_capacity(images.len());
            for (i, image) in images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let src_pixels = (w as u64 * h as u64) as usize;
                let input_buffer = upload_image_to_gpu(ctx, image, w, h)?;
                let blurred = gpu_blur.blur_gpu(
                    ctx, &input_buffer, src_pixels, w, h, kernel_size, sigma,
                )?;
                ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);
                buffers.push(blurred);
            }
            Ok(buffers)
        } else {
            let mut buffers = Vec::with_capacity(images.len());
            for (i, image) in images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let buffer = upload_image_to_gpu(ctx, image, w, h)?;
                buffers.push(buffer);
            }
            Ok(buffers)
        }
    }

    /// GPU 预处理（批量编码模式）：上传图像到 GPU 并可选执行高斯模糊。
    ///
    /// 与 `preprocess_gpu()` 功能相同，但将模糊 dispatch 命令编码到
    /// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
    pub fn preprocess_gpu_batch(
        &self,
        ctx: &GpuContext,
        batch: &mut GpuBatchSubmitter,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<GpuBuffer>, GpuError> {
        if let (Some(ref gpu_blur), Some(sigma), Some(kernel_size)) =
            (&self.gpu_blur, self.blur_sigma, self.blur_kernel_size)
        {
            let mut buffers = Vec::with_capacity(images.len());
            for (i, image) in images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let src_pixels = (w as u64 * h as u64) as usize;
                let input_buffer = upload_image_to_gpu(ctx, image, w, h)?;
                let blurred = gpu_blur.blur_gpu_batch(
                    ctx, batch, &input_buffer, src_pixels, w, h, kernel_size, sigma,
                )?;
                ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);
                buffers.push(blurred);
            }
            Ok(buffers)
        } else {
            let mut buffers = Vec::with_capacity(images.len());
            for (i, image) in images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let buffer = upload_image_to_gpu(ctx, image, w, h)?;
                buffers.push(buffer);
            }
            Ok(buffers)
        }
    }

    /// GPU 预处理（从预打包数据）：上传已打包的 u32 像素数据并可选执行高斯模糊。
    ///
    /// 与 [`preprocess_gpu`](Self::preprocess_gpu) 功能相同，但跳过 u8→u32 像素打包步骤。
    /// 用于双缓冲流水线中 CPU 线程已预先完成像素打包的场景。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `packed_images` — 已打包为 u32 的像素数据（每张图像一个 Vec）
    /// - `dimensions` — 对应图像的 (width, height)
    #[allow(dead_code)]
    pub(super) fn preprocess_gpu_from_packed(
        &self,
        ctx: &GpuContext,
        packed_images: &[Vec<u32>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<GpuBuffer>, GpuError> {
        if let (Some(ref gpu_blur), Some(sigma), Some(kernel_size)) =
            (&self.gpu_blur, self.blur_sigma, self.blur_kernel_size)
        {
            let mut buffers = Vec::with_capacity(packed_images.len());
            for (i, packed) in packed_images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let src_pixels = (w as u64 * h as u64) as usize;
                let input_buffer = upload_packed_to_gpu(ctx, packed)?;
                let blurred = gpu_blur.blur_gpu(
                    ctx, &input_buffer, src_pixels, w, h, kernel_size, sigma,
                )?;
                ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);
                buffers.push(blurred);
            }
            Ok(buffers)
        } else {
            let mut buffers = Vec::with_capacity(packed_images.len());
            for packed in packed_images {
                let buffer = upload_packed_to_gpu(ctx, packed)?;
                buffers.push(buffer);
            }
            Ok(buffers)
        }
    }

    /// GPU 预处理（批量编码模式）：上传已打包数据并可选执行高斯模糊。
    ///
    /// 与 `preprocess_gpu_from_packed()` 功能相同，但将模糊 dispatch 命令编码到
    /// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
    pub(super) fn preprocess_gpu_from_packed_batch(
        &self,
        ctx: &GpuContext,
        batch: &mut GpuBatchSubmitter,
        packed_images: &[Vec<u32>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<GpuBuffer>, GpuError> {
        if let (Some(ref gpu_blur), Some(sigma), Some(kernel_size)) =
            (&self.gpu_blur, self.blur_sigma, self.blur_kernel_size)
        {
            let mut buffers = Vec::with_capacity(packed_images.len());
            for (i, packed) in packed_images.iter().enumerate() {
                let (w, h) = dimensions[i];
                let src_pixels = (w as u64 * h as u64) as usize;
                let input_buffer = upload_packed_to_gpu(ctx, packed)?;
                let blurred = gpu_blur.blur_gpu_batch(
                    ctx, batch, &input_buffer, src_pixels, w, h, kernel_size, sigma,
                )?;
                ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);
                buffers.push(blurred);
            }
            Ok(buffers)
        } else {
            let mut buffers = Vec::with_capacity(packed_images.len());
            for packed in packed_images {
                let buffer = upload_packed_to_gpu(ctx, packed)?;
                buffers.push(buffer);
            }
            Ok(buffers)
        }
    }

    /// CPU 预处理：当前为直通（高斯模糊在 compute_cpu 中统一处理）。
    #[cfg(feature = "cpu-fallback")]
    pub fn preprocess_cpu(
        &self,
        images: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        Ok(images.to_vec())
    }

    // ==================== 阶段二：缩放 ====================

    /// GPU 缩放：将预处理后的 GPU 缓冲区缩放到目标尺寸。
    ///
    /// 内部合并输入缓冲区后执行批量 GPU 缩放。
    /// 所有图像必须具有相同的源尺寸（由 `src_width` 和 `src_height` 指定）。
    ///
    /// 需要构造时启用 GPU 缩放（`use_gpu_resize = true` 或启用高斯模糊），
    /// 否则返回 `GpuError::InvalidInput`。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `gpu_images` — 预处理阶段返回的 GPU 缓冲区
    /// - `src_width` — 源图像宽度（所有图像必须相同）
    /// - `src_height` — 源图像高度（所有图像必须相同）
    ///
    /// # 返回
    ///
    /// `(GpuBuffer, usize)` — 缩放后的合并缓冲区及其 u32 元素总数。
    /// 调用方负责在使用完毕后释放返回的缓冲区：
    /// ```ignore
    /// ctx.buffer_pool().release(resized.0.into_raw(), BufferUsage::Storage);
    /// ```
    pub fn resize_gpu(
        &self,
        ctx: &GpuContext,
        gpu_images: &[GpuBuffer],
        src_width: u32,
        src_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let gpu_resize = self.gpu_resize.as_ref().ok_or_else(|| {
            GpuError::InvalidInput("GPU 缩放器未启用，请使用 with_resize_mode(true) 或 with_blur() 创建".to_string())
        })?;

        let image_count = gpu_images.len();
        if image_count == 0 {
            return Err(GpuError::InvalidInput("图像列表为空".to_string()));
        }

        let src_pixels = (src_width as u64 * src_height as u64) as usize;

        if image_count == 1 {
            return gpu_resize.resize_batch_gpu_from_buffer(
                ctx, &gpu_images[0], src_width, src_height, 1,
                self.target_width, self.target_height,
            );
        }

        let merged = merge_gpu_buffers(ctx, gpu_images, src_pixels)?;
        let result = gpu_resize.resize_batch_gpu_from_buffer(
            ctx, &merged, src_width, src_height, image_count,
            self.target_width, self.target_height,
        )?;
        ctx.buffer_pool().release(merged.into_raw(), BufferUsage::Storage);
        Ok(result)
    }

    /// GPU 缩放（批量编码模式）：将预处理后的 GPU 缓冲区缩放到目标尺寸。
    ///
    /// 与 `resize_gpu()` 功能相同，但将 resize dispatch 命令编码到
    /// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
    pub fn resize_gpu_batch(
        &self,
        ctx: &GpuContext,
        batch: &mut GpuBatchSubmitter,
        gpu_images: &[GpuBuffer],
        src_width: u32,
        src_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let gpu_resize = self.gpu_resize.as_ref().ok_or_else(|| {
            GpuError::InvalidInput("GPU 缩放器未启用，请使用 with_resize_mode(true) 或 with_blur() 创建".to_string())
        })?;

        let image_count = gpu_images.len();
        if image_count == 0 {
            return Err(GpuError::InvalidInput("图像列表为空".to_string()));
        }

        let src_pixels = (src_width as u64 * src_height as u64) as usize;

        if image_count == 1 {
            return gpu_resize.resize_batch_gpu_from_buffer_batch(
                ctx, batch, &gpu_images[0], src_width, src_height, 1,
                self.target_width, self.target_height,
            );
        }

        let merged = merge_gpu_buffers_batch(ctx, batch, gpu_images, src_pixels)?;
        let result = gpu_resize.resize_batch_gpu_from_buffer_batch(
            ctx, batch, &merged, src_width, src_height, image_count,
            self.target_width, self.target_height,
        )?;
        ctx.buffer_pool().release(merged.into_raw(), BufferUsage::Storage);
        Ok(result)
    }

    /// CPU 缩放：将图像缩放到目标尺寸。
    ///
    /// 使用盒式滤波下采样，适合感知哈希场景。
    ///
    /// # 参数
    ///
    /// - `images` — 灰度像素数据
    /// - `dimensions` — 对应图像的 (width, height)
    pub fn resize_cpu(
        &self,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        let resized: Vec<Vec<u8>> = images
            .iter()
            .zip(dimensions.iter())
            .map(|(pixels, &(w, h))| {
                resize_grayscale(pixels, w, h, self.target_width, self.target_height)
            })
            .collect();
        Ok(resized)
    }

    // ==================== 阶段三：哈希计算 ====================

    /// GPU 哈希计算：从 GPU 缓冲区计算感知哈希。
    ///
    /// 输入缓冲区应包含已缩放到目标尺寸的 u32 打包像素数据。
    ///
    /// 对于 Mean Hash 和 Median Hash，需要从 GPU buffer 下载像素数据到 CPU
    /// 计算阈值（均值/中位数），然后创建扩展输入缓冲区进行 dispatch。
    /// 其他算法直接使用输入缓冲区，保持零拷贝。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `gpu_buffer` — 包含已缩放像素数据的 GPU 缓冲区
    /// - `u32_count` — 缓冲区中 u32 元素总数
    /// - `image_count` — 缓冲区中的图像数量
    pub fn compute_hash_gpu(
        &self,
        ctx: &GpuContext,
        gpu_buffer: &GpuBuffer,
        u32_count: usize,
        image_count: usize,
    ) -> Result<Vec<u64>, GpuError> {
        // Mean/Median Hash 需要预计算阈值
        if matches!(self.algorithm, HashAlgorithm::Mean | HashAlgorithm::Median) {
            let thresholds = self.compute_thresholds_from_gpu_buffer(
                ctx, gpu_buffer, u32_count, image_count,
            )?;
            compute_phash_from_gpu_buffer_with_thresholds(
                self.computer.pipeline(),
                ctx,
                gpu_buffer,
                u32_count,
                image_count,
                self.target_width,
                self.target_height,
                self.computer.workgroup_size(),
                self.hash_size,
                &thresholds,
            )
        } else {
            compute_phash_from_gpu_buffer(
                self.computer.pipeline(),
                ctx,
                gpu_buffer,
                u32_count,
                image_count,
                self.target_width,
                self.target_height,
                self.computer.workgroup_size(),
                self.hash_size,
            )
        }
    }

    /// CPU 哈希计算：从已缩放的图像数据计算感知哈希。
    ///
    /// `images` 中每个元素应为已缩放到目标尺寸的灰度像素数据。
    #[cfg(feature = "cpu-fallback")]
    pub fn compute_hash_cpu(
        &self,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        self.cpu_hasher.compute(images, self.target_width, self.target_height)
    }

    // ==================== 编排方法 ====================

    /// RGBA→灰度 CPU 转换（czkawka 兼容公式）。
    ///
    /// 使用整数运算 `(R*77 + G*150 + B*29) >> 8`，与 czkawka CPU 路径一致。
    /// Alpha 通道被忽略。
    ///
    /// 当 `simd` feature 启用时，自动使用 SIMD 向量化路径加速。
    pub fn rgba_to_grayscale_cpu(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        #[cfg(feature = "simd")]
        {
            return Self::rgba_to_grayscale_simd(rgba, width, height);
        }
        #[cfg(not(feature = "simd"))]
        {
            Self::rgba_to_grayscale_scalar(rgba, width, height)
        }
    }

    /// RGBA→灰度标量实现（czkawka 兼容公式）。
    pub fn rgba_to_grayscale_scalar(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width as usize) * (height as usize);
        let mut gray = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let offset = i * 4;
            let r = rgba[offset] as u32;
            let g = rgba[offset + 1] as u32;
            let b = rgba[offset + 2] as u32;
            gray.push(((r * 77 + g * 150 + b * 29) >> 8) as u8);
        }
        gray
    }

    /// RGBA→灰度 SIMD 实现（czkawka 兼容公式）。
    ///
    /// 使用 `wide` crate 的 `u16x8` SIMD 类型，一次处理 8 个像素。
    /// 尾部不足 8 像素时回退到标量实现。
    #[cfg(feature = "simd")]
    fn rgba_to_grayscale_simd(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
        use wide::u16x8;

        let pixel_count = (width as usize) * (height as usize);
        let mut gray = Vec::with_capacity(pixel_count);

        // 每次处理 8 个像素 = 32 字节 RGBA
        const CHUNK_PIXELS: usize = 8;
        const CHUNK_BYTES: usize = CHUNK_PIXELS * 4;

        let full_chunks = pixel_count / CHUNK_PIXELS;
        let remainder_start = full_chunks * CHUNK_PIXELS;

        let coeff_r = u16x8::splat(77);
        let coeff_g = u16x8::splat(150);
        let coeff_b = u16x8::splat(29);

        for chunk_idx in 0..full_chunks {
            let base = chunk_idx * CHUNK_BYTES;

            // 提取 8 个像素的 R、G、B 通道，扩展为 u16
            let mut r_vals = [0u16; 8];
            let mut g_vals = [0u16; 8];
            let mut b_vals = [0u16; 8];
            for j in 0..CHUNK_PIXELS {
                r_vals[j] = rgba[base + j * 4] as u16;
                g_vals[j] = rgba[base + j * 4 + 1] as u16;
                b_vals[j] = rgba[base + j * 4 + 2] as u16;
            }

            // SIMD 运算：gray = (R*77 + G*150 + B*29) >> 8
            let r = u16x8::from(r_vals);
            let g = u16x8::from(g_vals);
            let b = u16x8::from(b_vals);
            let gray_simd = (r * coeff_r + g * coeff_g + b * coeff_b) >> 8u16;

            // 写入输出
            let gray_arr: [u16; 8] = gray_simd.to_array();
            for v in &gray_arr {
                gray.push(*v as u8);
            }
        }

        // 处理尾部剩余像素（标量回退）
        for i in remainder_start..pixel_count {
            let offset = i * 4;
            let r = rgba[offset] as u32;
            let g = rgba[offset + 1] as u32;
            let b = rgba[offset + 2] as u32;
            gray.push(((r * 77 + g * 150 + b * 29) >> 8) as u8);
        }

        gray
    }

    /// 对任意尺寸的 RGBA 像素数据计算感知哈希。
    ///
    /// 接受 RGBA8888 格式数据（每像素 4 字节），内部使用 czkawka 兼容公式
    /// `(R*77 + G*150 + B*29) >> 8` 转换为灰度后计算哈希。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `rgba_images` — RGBA 像素数据，每个元素为一张图像的宽×高×4 字节
    /// - `dimensions` — 对应图像的 (width, height)
    pub fn compute_from_rgba(
        &self,
        ctx: &GpuContext,
        rgba_images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        if rgba_images.len() != dimensions.len() {
            return Err(GpuError::InvalidInput(
                "图像数量与尺寸数量不匹配".to_string(),
            ));
        }
        if rgba_images.is_empty() {
            return Ok(vec![]);
        }

        // RGBA→灰度转换（czkawka 兼容公式）
        let gray_images: Vec<Vec<u8>> = rgba_images
            .iter()
            .zip(dimensions.iter())
            .map(|(rgba, &(w, h))| Self::rgba_to_grayscale_cpu(rgba, w, h))
            .collect();

        // 委托到现有 compute 方法
        self.compute(ctx, &gray_images, dimensions)
    }

    /// 对任意尺寸的 RGBA 像素数据计算感知哈希，返回 `HashBytes`。
    ///
    /// 与 [`compute_from_rgba()`](Self::compute_from_rgba) 功能相同，
    /// 但返回 `Vec<HashBytes>`（对应 czkawka `ImHash = Vec<u8>`）。
    pub fn compute_from_rgba_to_hash_bytes(
        &self,
        ctx: &GpuContext,
        rgba_images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<HashBytes>, GpuError> {
        let hashes = self.compute_from_rgba(ctx, rgba_images, dimensions)?;
        // 将 u64 哈希转换为 HashBytes
        // 每个图像的 u64 数量取决于 hash_size
        let u64s_per_image = self.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(hashes.len() / u64s_per_image.max(1));
        for chunk in hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 对任意尺寸的灰度像素数据计算感知哈希，返回 `HashBytes`。
    ///
    /// 与 [`compute()`](Self::compute) 功能相同，
    /// 但返回 `Vec<HashBytes>`（对应 czkawka `ImHash = Vec<u8>`）。
    pub fn compute_to_hash_bytes(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<HashBytes>, GpuError> {
        let hashes = self.compute(ctx, images, dimensions)?;
        let u64s_per_image = self.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(hashes.len() / u64s_per_image.max(1));
        for chunk in hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 对任意尺寸的灰度像素数据计算感知哈希。
    ///
    /// 编排预处理 → 缩放 → 哈希三阶段流水线，
    /// 根据 GPU/CPU 后端和配置自动选择最优路径。
    ///
    /// `images` 中每个元素为一张图像的灰度像素（宽×高字节），
    /// `dimensions` 为对应图像的 (width, height)。
    /// 内部自动缩放到目标尺寸后计算。
    ///
    /// 如果构造时启用了 GPU 缩放（`use_gpu_resize = true`），
    /// 则所有输入图像必须具有相同的源尺寸。
    pub fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        if images.len() != dimensions.len() {
            return Err(GpuError::InvalidInput(
                "图像数量与尺寸数量不匹配".to_string(),
            ));
        }

        if images.is_empty() {
            return Ok(vec![]);
        }

        let dispatcher = DefaultBackendDispatcher;
        dispatcher.dispatch_gpu(
            ctx,
            |ctx| self.compute_gpu(ctx, images, dimensions),
            || {
                #[cfg(feature = "cpu-fallback")]
                {
                    self.compute_cpu(images, dimensions)
                }
                #[cfg(not(feature = "cpu-fallback"))]
                {
                    Err(GpuError::CpuFallback("CPU 降级未启用".to_string()))
                }
            },
        )
    }

    /// 对已缩放到目标尺寸的灰度像素数据计算感知哈希（跳过缩放步骤）。
    ///
    /// 当 `GpuContext` 处于 CPU 降级模式时，自动委托到 [`PHasherCpu`] 计算。
    pub fn compute_resized(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        let dispatcher = DefaultBackendDispatcher;
        dispatcher.dispatch_gpu(
            ctx,
            |ctx| self.computer.compute(ctx, images),
            || {
                #[cfg(feature = "cpu-fallback")]
                {
                    self.cpu_hasher.compute(images, self.target_width, self.target_height)
                }
                #[cfg(not(feature = "cpu-fallback"))]
                {
                    Err(GpuError::CpuFallback("CPU 降级未启用".to_string()))
                }
            },
        )
    }

    // ==================== 内部编排实现 ====================

    /// GPU 路径编排：预处理 → 缩放 → 哈希。
    fn compute_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        let all_target_size = dimensions.iter().all(|&(w, h)| {
            w == self.target_width && h == self.target_height
        });

        // 快速路径：已缩放且无需模糊，直接计算哈希
        if all_target_size && self.gpu_blur.is_none() {
            return self.computer.compute(ctx, images);
        }

        // PDQ 特殊路径：使用 CPU 缩放 + GPU 哈希
        #[cfg(feature = "pdq")]
        if self.algorithm == HashAlgorithm::Pdq {
            let resized = self.resize_cpu(images, dimensions)?;
            return self.computer.compute(ctx, &resized);
        }

        // 无 GPU 缩放且无需模糊：CPU 缩放 + GPU 哈希
        if self.gpu_resize.is_none() && self.gpu_blur.is_none() {
            let resized = self.resize_cpu(images, dimensions)?;
            return self.computer.compute(ctx, &resized);
        }

        // 需要 GPU 管线（模糊和/或缩放）
        let all_same_size = dimensions.windows(2).all(|w| w[0] == w[1]);

        if all_same_size {
            self.compute_gpu_batch_pipeline(ctx, images, dimensions, all_target_size)
        } else {
            self.compute_gpu_per_image_pipeline(ctx, images, dimensions)
        }
    }

    /// CPU 路径编排：预处理 → 高斯模糊 → 缩放 → 哈希。
    #[cfg(feature = "cpu-fallback")]
    fn compute_cpu(
        &self,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        // 阶段一：CPU 预处理
        let preprocessed = self.preprocess_cpu(images)?;

        // 阶段二：CPU 高斯模糊（如果启用）
        let blurred: Vec<Vec<u8>> = if let Some(sigma) = self.blur_sigma {
            preprocessed
                .iter()
                .zip(dimensions.iter())
                .map(|(img, &(w, h))| cpu_gaussian_blur(img, w, h, sigma))
                .collect()
        } else {
            preprocessed
        };

        // 阶段三：CPU 缩放
        let all_target_size = dimensions.iter().all(|&(w, h)| {
            w == self.target_width && h == self.target_height
        });
        let resized = if all_target_size {
            blurred
        } else {
            self.resize_cpu(&blurred, dimensions)?
        };

        // 阶段四：CPU 哈希
        self.compute_hash_cpu(&resized)
    }
}

#[cfg(feature = "image")]
mod image_support {
    use super::*;
    use crate::backend_dispatcher::{BackendDispatcher, DefaultBackendDispatcher};
    use image::{DynamicImage, GenericImageView, imageops};

    impl PerceptualHasher {
        /// 从 `image::DynamicImage` 计算感知哈希（需要启用 `image` feature）。
        ///
        /// 内部使用 Lanczos3 高质量缩放。
        /// 当 `GpuContext` 处于 CPU 降级模式时，自动委托到 [`PHasherCpu`] 计算。
        pub fn compute_images(
            &self,
            ctx: &GpuContext,
            images: &[DynamicImage],
        ) -> Result<Vec<u64>, GpuError> {
            // 检查是否启用 GPU Lanczos3 缩放
            let use_gpu_lanczos3 = self.gpu_resize.as_ref()
                .map(|r| r.filter() == ResizeFilter::Lanczos3)
                .unwrap_or(false);

            if use_gpu_lanczos3 {
                // GPU Lanczos3 路径：灰度转换 → GPU 上传 → GPU Lanczos3 缩放 → GPU 哈希
                // 先转灰度（CPU），再通过 compute() 走 GPU 流水线
                let gray_images: Vec<Vec<u8>> = images
                    .iter()
                    .map(|img| {
                        let luma = img.grayscale();
                        luma.pixels().map(|(_, _, p)| p.0[0]).collect()
                    })
                    .collect();
                let dimensions: Vec<(u32, u32)> = images
                    .iter()
                    .map(|img| img.dimensions())
                    .collect();

                self.compute(ctx, &gray_images, &dimensions)
            } else {
                // CPU Lanczos3 路径（原逻辑）：CPU 缩放 → 灰度 → GPU/CPU 哈希
                let resized: Vec<Vec<u8>> = images
                    .iter()
                    .map(|img| {
                        let resized = img.resize_exact(
                            self.target_width,
                            self.target_height,
                            imageops::FilterType::Lanczos3,
                        );
                        let luma = resized.grayscale();
                        luma.pixels().map(|(_, _, luma)| luma.0[0]).collect()
                    })
                    .collect();

                let dispatcher = DefaultBackendDispatcher;
                dispatcher.dispatch_gpu(
                    ctx,
                    |ctx| self.computer.compute(ctx, &resized),
                    || {
                        #[cfg(feature = "cpu-fallback")]
                        {
                            self.cpu_hasher.compute(&resized, self.target_width, self.target_height)
                        }
                        #[cfg(not(feature = "cpu-fallback"))]
                        {
                            Err(GpuError::CpuFallback("CPU 降级未启用".to_string()))
                        }
                    },
                )
            }
        }
    }
}
