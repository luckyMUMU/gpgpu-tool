use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::hash_common::PerceptualHashComputer;
use crate::tasks::mean_hash::MeanHashComputer;
use crate::tasks::median_hash::MedianHashComputer;
use crate::tasks::gradient_hash::GradientHashComputer;
use crate::tasks::block_hash::BlockHashComputer;
use crate::tasks::vert_gradient_hash::VertGradientHashComputer;
use crate::tasks::double_gradient_hash::DoubleGradientHashComputer;

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
    /// 返回算法的目标缩放尺寸 (width, height)。
    pub fn target_size(&self) -> (u32, u32) {
        match self {
            HashAlgorithm::Mean | HashAlgorithm::Median => (8, 8),
            HashAlgorithm::Gradient => (8, 9),
            HashAlgorithm::Block => (16, 16),
            HashAlgorithm::VertGradient => (9, 8),
            HashAlgorithm::DoubleGradient => (9, 9),
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
}

impl PerceptualHasher {
    /// 创建感知哈希计算器。
    pub fn new(ctx: &mut GpuContext, algorithm: HashAlgorithm) -> Result<Self, GpuError> {
        let (w, h) = algorithm.target_size();
        let computer: Box<dyn PerceptualHashComputer> = match algorithm {
            HashAlgorithm::Mean => Box::new(MeanHashComputer::new(ctx)?),
            HashAlgorithm::Median => Box::new(MedianHashComputer::new(ctx)?),
            HashAlgorithm::Gradient => Box::new(GradientHashComputer::new(ctx)?),
            HashAlgorithm::Block => Box::new(BlockHashComputer::new(ctx)?),
            HashAlgorithm::VertGradient => Box::new(VertGradientHashComputer::new(ctx)?),
            HashAlgorithm::DoubleGradient => Box::new(DoubleGradientHashComputer::new(ctx)?),
        };
        Ok(Self {
            algorithm,
            target_width: w,
            target_height: h,
            computer,
        })
    }

    /// 返回算法类型。
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// 返回目标缩放尺寸。
    pub fn target_size(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    /// 对任意尺寸的灰度像素数据计算感知哈希。
    ///
    /// `images` 中每个元素为一张图像的灰度像素（宽×高字节），
    /// `dimensions` 为对应图像的 (width, height)。
    /// 内部自动缩放到目标尺寸后计算。
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

    let mut output = Vec::with_capacity((dst_w * dst_h) as usize);

    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;

    for dy in 0..dst_h {
        let src_y_start = (dy as f64 * y_ratio) as u32;
        let src_y_end = ((dy + 1) as f64 * y_ratio).min(src_h as f64) as u32;
        let y_count = (src_y_end - src_y_start).max(1);

        for dx in 0..dst_w {
            let src_x_start = (dx as f64 * x_ratio) as u32;
            let src_x_end = ((dx + 1) as f64 * x_ratio).min(src_w as f64) as u32;
            let x_count = (src_x_end - src_x_start).max(1);

            let mut sum: u32 = 0;
            for sy in src_y_start..src_y_end {
                for sx in src_x_start..src_x_end {
                    sum += pixels[(sy * src_w + sx) as usize] as u32;
                }
            }
            output.push((sum / (x_count * y_count)) as u8);
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