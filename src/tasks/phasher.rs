use crate::buffer::BufferUsage;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::hash_common::{HashSize, PerceptualHashComputer, compute_phash_from_gpu_buffer};
use crate::tasks::mean_hash::MeanHashComputer;
use crate::tasks::median_hash::MedianHashComputer;
use crate::tasks::gradient_hash::GradientHashComputer;
use crate::tasks::block_hash::BlockHashComputer;
use crate::tasks::vert_gradient_hash::VertGradientHashComputer;
use crate::tasks::double_gradient_hash::DoubleGradientHashComputer;
use crate::tasks::gpu_resize::{GpuResize, GpuResizeConfig};

/// 感知哈希算法类型。
#[derive(Debug, Clone, Copy)]
pub enum HashAlgorithm {
    Mean,
    Median,
    Gradient,
    Block,
    VertGradient,
    DoubleGradient,
}

impl HashAlgorithm {
    pub fn target_size_for(&self, hash_size: HashSize) -> (u32, u32) {
        let s = hash_size.size();
        match self {
            HashAlgorithm::Mean | HashAlgorithm::Median | HashAlgorithm::Block => (s, s),
            HashAlgorithm::Gradient => (s, s + 1),
            HashAlgorithm::VertGradient => (s + 1, s),
            HashAlgorithm::DoubleGradient => (s + 1, s + 1),
        }
    }
}

/// 感知哈希计算器，支持任意尺寸图像输入。
///
/// 内部自动将图像缩放到算法所需的目标尺寸后计算哈希。
/// 持有 GPU 计算器实例，避免重复创建管线。
///
/// # 示例
///
/// ```no_run
/// use wgpu_compute_engine::GpuContext;
/// use wgpu_compute_engine::tasks::phasher::{PerceptualHasher, HashAlgorithm};
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
    hash_size: HashSize,
    max_batch_size: u64,
}

/// 感知哈希计算器的默认 workgroup 大小。
pub const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
/// 感知哈希批次处理默认最大缓冲区字节数（128 MB）。
pub const DEFAULT_MAX_BATCH_SIZE: u64 = 128 * 1024 * 1024;

impl PerceptualHasher {
    /// 创建感知哈希计算器（默认 64bit，CPU 缩放，默认 workgroup_size）。
    pub fn new(ctx: &mut GpuContext, algorithm: HashAlgorithm) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, false, HashSize::default(), DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建感知哈希计算器，指定哈希位长和是否 GPU 缩放。
    pub fn with_resize_mode(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, use_gpu_resize, HashSize::default(), DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建感知哈希计算器，指定 hash_size（CPU 缩放，默认 workgroup_size）。
    pub fn with_hash_size(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, false, hash_size, DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建感知哈希计算器，完整配置（所有参数均可定制）。
    pub fn with_config(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        Self::with_full_config(ctx, algorithm, use_gpu_resize, hash_size, DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建感知哈希计算器，完整配置（包含 workgroup_size）。
    pub fn with_full_config(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        Self::with_full_config_and_gpu_resize(ctx, algorithm, use_gpu_resize, hash_size, workgroup_size, GpuResizeConfig::default())
    }

    /// 创建感知哈希计算器，完全自定义配置。
    ///
    /// 允许同时指定 workgroup_size、hash_size 和 GpuResizeConfig。
    /// 如果 `use_gpu_resize` 为 true，`gpu_resize_config` 决定 GPU 缩放器的行为。
    pub fn with_full_config_and_gpu_resize(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        use_gpu_resize: bool,
        hash_size: HashSize,
        workgroup_size: [u32; 3],
        gpu_resize_config: GpuResizeConfig,
    ) -> Result<Self, GpuError> {
        let (w, h) = algorithm.target_size_for(hash_size);
        let computer: Box<dyn PerceptualHashComputer> = match algorithm {
            HashAlgorithm::Mean => Box::new(MeanHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Median => Box::new(MedianHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Gradient => Box::new(GradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::Block => Box::new(BlockHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::VertGradient => Box::new(VertGradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
            HashAlgorithm::DoubleGradient => Box::new(DoubleGradientHashComputer::with_config(ctx, workgroup_size, hash_size)?),
        };
        let max_batch_size = gpu_resize_config.max_buffer_size / 2;
        let gpu_resize = if use_gpu_resize {
            Some(GpuResize::with_config(ctx, gpu_resize_config)?)
        } else {
            None
        };
        Ok(Self {
            algorithm,
            target_width: w,
            target_height: h,
            computer,
            gpu_resize,
            hash_size,
            max_batch_size,
        })
    }

    /// 返回哈希位长配置。
    pub fn hash_size(&self) -> HashSize { self.hash_size }
    /// 返回算法类型。
    pub fn algorithm(&self) -> HashAlgorithm { self.algorithm }
    /// 返回目标缩放尺寸。
    pub fn target_size(&self) -> (u32, u32) { (self.target_width, self.target_height) }

    /// 对任意尺寸的灰度像素数据计算感知哈希。
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

        // 检查是否所有图像已经是目标尺寸
        let all_target_size = dimensions.iter().all(|&(w, h)| {
            w == self.target_width && h == self.target_height
        });

        if all_target_size {
            return self.computer.compute(ctx, images);
        }

        // 检查是否所有图像尺寸相同（GPU 缩放要求）
        let all_same_size = dimensions.windows(2).all(|w| w[0] == w[1]);

        if let Some(ref gpu_resize) = self.gpu_resize {
            if all_same_size {
                let (src_w, src_h) = dimensions[0];
                let src_pixels = (src_w * src_h) as usize;
                let dst_pixels = (self.target_width * self.target_height) as usize;
                let src_u32_per_image = src_pixels as u64; // u32-per-pixel
                let dst_u32_per_image = dst_pixels as u64;
                let u32_per_image = src_u32_per_image + dst_u32_per_image;
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

                    let (resized_buffer, resized_u32_count) = gpu_resize.resize_batch_gpu(
                        ctx,
                        chunk_images,
                        chunk_dims,
                        self.target_width,
                        self.target_height,
                    )?;
                    let hashes = compute_phash_from_gpu_buffer(
                        self.computer.pipeline(),
                        ctx,
                        &resized_buffer,
                        resized_u32_count,
                        chunk_images.len(),
                        self.target_width,
                        self.target_height,
                        self.computer.workgroup_size(),
                        gpu_resize.buffer_pool(),
                        self.hash_size,
                    )?;
                    gpu_resize.buffer_pool().release(resized_buffer.into_raw(), BufferUsage::Storage);
                    all_hashes.extend(hashes);
                }
                return Ok(all_hashes);
            }
        }

        // 回退到 CPU 缩放（支持不同尺寸）
        let resized: Vec<Vec<u8>> = images
            .iter()
            .zip(dimensions.iter())
            .map(|(pixels, &(w, h))| resize_grayscale(pixels, w, h, self.target_width, self.target_height))
            .collect();

        self.computer.compute(ctx, &resized)
    }

    /// 对已缩放到目标尺寸的灰度像素数据计算感知哈希（跳过缩放步骤）。
    pub fn compute_resized(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        self.computer.compute(ctx, images)
    }
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
    let mut output = Vec::with_capacity((dst_w * dst_h) as usize);

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
    use image::{DynamicImage, GenericImageView, imageops};

    impl PerceptualHasher {
        /// 从 `image::DynamicImage` 计算感知哈希（需要启用 `image` feature）。
        ///
        /// 内部使用 Lanczos3 高质量缩放。
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

            self.computer.compute(ctx, &resized)
        }
    }
}
