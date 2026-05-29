use crate::backend_dispatcher::{BackendDispatcher, DefaultBackendDispatcher};
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::gaussian_blur::GpuGaussianBlur;
use crate::tasks::hash_common::{HashSize, PerceptualHashComputer, compute_phash_from_gpu_buffer};
use crate::tasks::mean_hash::MeanHashComputer;
use crate::tasks::median_hash::MedianHashComputer;
use crate::tasks::gradient_hash::GradientHashComputer;
use crate::tasks::block_hash::BlockHashComputer;
use crate::tasks::vert_gradient_hash::VertGradientHashComputer;
use crate::tasks::double_gradient_hash::DoubleGradientHashComputer;
use crate::tasks::gpu_resize::{GpuResize, GpuResizeConfig};
#[cfg(feature = "cpu-fallback")]
use crate::tasks::phasher_cpu::PHasherCpu;
#[cfg(feature = "pdq")]
use crate::tasks::pdq_hash::PdqHashGpu;

/// 感知哈希算法类型。
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
    algorithm: HashAlgorithm,
    target_width: u32,
    target_height: u32,
    computer: Box<dyn PerceptualHashComputer>,
    gpu_resize: Option<GpuResize>,
    gpu_blur: Option<GpuGaussianBlur>,
    blur_sigma: Option<f32>,
    blur_kernel_size: Option<u32>,
    hash_size: HashSize,
    max_batch_size: u64,
    #[cfg(feature = "cpu-fallback")]
    cpu_hasher: PHasherCpu,
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

    /// CPU 预处理：当前为直通（CPU 降级路径暂不支持高斯模糊）。
    ///
    /// 高斯模糊是可选增强步骤，跳过不影响哈希结果的基本正确性。
    /// 启用模糊时会在日志中输出警告。
    #[cfg(feature = "cpu-fallback")]
    pub fn preprocess_cpu(
        &self,
        images: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        if self.blur_sigma.is_some() {
            log::warn!("CPU 降级模式下暂不支持高斯模糊预处理，已跳过模糊步骤");
        }
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

    /// GPU 批量管线：所有图像同尺寸时的 预处理 → 缩放 → 哈希。
    ///
    /// 含分块逻辑，避免单次 GPU dispatch 超出缓冲区限制。
    fn compute_gpu_batch_pipeline(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        all_target_size: bool,
    ) -> Result<Vec<u64>, GpuError> {
        let (src_w, src_h) = dimensions[0];
        let src_pixels = (src_w as u64 * src_h as u64) as usize;
        let dst_pixels = (self.target_width as u64 * self.target_height as u64) as usize;
        let src_u32_per_image = src_pixels as u64;
        let dst_u32_per_image = dst_pixels as u64;
        let u32_per_image = if all_target_size {
            src_u32_per_image
        } else {
            src_u32_per_image + dst_u32_per_image
        };
        let max_batch = if u32_per_image > 0 {
            (self.max_batch_size / (u32_per_image * 4)).max(1) as usize
        } else {
            images.len()
        };

        let mut all_hashes = Vec::with_capacity(images.len());
        for chunk_start in (0..images.len()).step_by(max_batch) {
            let chunk_end = (chunk_start + max_batch).min(images.len());
            let chunk_images = &images[chunk_start..chunk_end];
            let chunk_dims = &dimensions[chunk_start..chunk_end];
            let chunk_len = chunk_images.len();

            // 阶段一：预处理（上传 + 可选模糊）
            let gpu_images = self.preprocess_gpu(ctx, chunk_images, chunk_dims)?;

            if all_target_size {
                // 已缩放（有模糊），合并后直接计算哈希
                let merged = merge_gpu_buffers(ctx, &gpu_images, src_pixels)?;
                release_buffers(ctx, gpu_images);
                let u32_count = src_pixels * chunk_len;
                let hashes = self.compute_hash_gpu(ctx, &merged, u32_count, chunk_len)?;
                ctx.buffer_pool().release(merged.into_raw(), BufferUsage::Storage);
                all_hashes.extend(hashes);
            } else {
                // 阶段二：缩放
                let (resized, u32_count) = self.resize_gpu(ctx, &gpu_images, src_w, src_h)?;
                release_buffers(ctx, gpu_images);
                // 阶段三：哈希
                let hashes = self.compute_hash_gpu(ctx, &resized, u32_count, chunk_len)?;
                ctx.buffer_pool().release(resized.into_raw(), BufferUsage::Storage);
                all_hashes.extend(hashes);
            }
        }
        Ok(all_hashes)
    }

    /// GPU 逐图管线：图像尺寸不同时的 预处理 → 缩放 → 哈希。
    fn compute_gpu_per_image_pipeline(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        let mut all_hashes = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let (w, h) = dimensions[i];
            let dims = [(w, h)];

            // 阶段一：预处理
            let gpu_images = self.preprocess_gpu(ctx, std::slice::from_ref(image), &dims)?;

            if w == self.target_width && h == self.target_height {
                // 已缩放（有模糊），直接计算哈希
                let src_pixels = (w as u64 * h as u64) as usize;
                let hashes = self.compute_hash_gpu(ctx, &gpu_images[0], src_pixels, 1)?;
                release_buffers(ctx, gpu_images);
                all_hashes.extend(hashes);
            } else {
                // 阶段二：缩放
                let (resized, u32_count) = self.resize_gpu(ctx, &gpu_images, w, h)?;
                release_buffers(ctx, gpu_images);
                // 阶段三：哈希
                let hashes = self.compute_hash_gpu(ctx, &resized, u32_count, 1)?;
                ctx.buffer_pool().release(resized.into_raw(), BufferUsage::Storage);
                all_hashes.extend(hashes);
            }
        }
        Ok(all_hashes)
    }

    /// CPU 路径编排：预处理 → 缩放 → 哈希。
    #[cfg(feature = "cpu-fallback")]
    fn compute_cpu(
        &self,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        // 阶段一：CPU 预处理
        let preprocessed = self.preprocess_cpu(images)?;

        // 阶段二：CPU 缩放
        let all_target_size = dimensions.iter().all(|&(w, h)| {
            w == self.target_width && h == self.target_height
        });
        let resized = if all_target_size {
            preprocessed
        } else {
            self.resize_cpu(&preprocessed, dimensions)?
        };

        // 阶段三：CPU 哈希
        self.compute_hash_cpu(&resized)
    }
}

/// 释放 GPU 缓冲区列表到缓冲池。
fn release_buffers(ctx: &GpuContext, buffers: Vec<GpuBuffer>) {
    for buf in buffers {
        ctx.buffer_pool().release(buf.into_raw(), BufferUsage::Storage);
    }
}

/// 将灰度图像数据上传到 GPU 存储缓冲区。
///
/// 使用 pixel_pack 将 u8 像素扩展为 u32，再通过 buffer_pool 分配并写入 GPU。
fn upload_image_to_gpu(
    ctx: &GpuContext,
    image: &[u8],
    width: u32,
    height: u32,
) -> Result<GpuBuffer, GpuError> {
    let expected_len = (width as usize) * (height as usize);
    if image.len() != expected_len {
        return Err(GpuError::InvalidInput(format!(
            "图像数据长度 ({}) 与声明的尺寸 ({}x{}={}) 不匹配",
            image.len(), width, height, expected_len
        )));
    }
    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let packed = crate::pixel_pack::pack_u8_to_u32(image);
    let input_size = (packed.len() * 4) as u64;
    let input_buffer_raw = ctx.buffer_pool().acquire(device, input_size, BufferUsage::Storage);
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));
    Ok(GpuBuffer::from_raw(input_buffer_raw, input_size))
}

/// 将多个 GPU 缓冲区合并为一个连续存储缓冲区。
///
/// 使用 GPU 端 copy_buffer_to_buffer 实现零拷贝合并，
/// 避免将数据下载到 CPU 再重新上传。
fn merge_gpu_buffers(
    ctx: &GpuContext,
    buffers: &[GpuBuffer],
    pixels_per_image: usize,
) -> Result<GpuBuffer, GpuError> {
    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let byte_size_per_image = (pixels_per_image * 4) as u64;
    let total_bytes = byte_size_per_image * buffers.len() as u64;

    let merged_raw = ctx.buffer_pool().acquire(device, total_bytes, BufferUsage::Storage);
    let merged = GpuBuffer::from_raw(merged_raw, total_bytes);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("merge_buffers_encoder"),
    });
    for (i, buf) in buffers.iter().enumerate() {
        let offset = (i as u64) * byte_size_per_image;
        encoder.copy_buffer_to_buffer(buf.raw(), 0, merged.raw(), offset, byte_size_per_image);
    }
    queue.submit(std::iter::once(encoder.finish()));

    Ok(merged)
}

/// CPU 侧灰度图像盒式滤波下采样（零依赖，适合感知哈希场景）。
fn resize_grayscale(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return pixels.to_vec();
    }

    let sw = src_w as usize;
    let sh = src_h as usize;
    let stride = sw + 1;

    let mut sat = vec![0u64; (sh + 1) * stride];
    for y in 0..sh {
        let mut row_sum = 0u64;
        for x in 0..sw {
            row_sum += pixels[y * sw + x] as u64;
            sat[(y + 1) * stride + (x + 1)] = row_sum + sat[y * stride + (x + 1)];
        }
    }

    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;
    let mut output = Vec::with_capacity((dst_w as u64 * dst_h as u64) as usize);

    for dy in 0..dst_h {
        let y0 = (dy as f64 * y_ratio) as usize;
        let y1 = ((dy as f64 + 1.0) * y_ratio).min(src_h as f64) as usize;
        for dx in 0..dst_w {
            let x0 = (dx as f64 * x_ratio) as usize;
            let x1 = ((dx as f64 + 1.0) * x_ratio).min(src_w as f64) as usize;
            let top_right = sat[y1 * stride + x1] - sat[y0 * stride + x1];
            let bottom_right = sat[y1 * stride + x0] - sat[y0 * stride + x0];
            let sum = top_right - bottom_right;
            let area = ((x1 - x0) * (y1 - y0)).max(1);
            output.push((sum / area as u64) as u8);
        }
    }

    output
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
