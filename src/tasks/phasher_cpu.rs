use crate::error::GpuError;
use crate::tasks::hash_common::HashSize;
use crate::tasks::phasher::HashAlgorithm;
#[cfg(feature = "pdq")]
use crate::tasks::pdq_hash::PdqHashCpu;

/// 感知哈希 CPU 降级实现。
///
/// 当 GPU 不可用时，在 CPU 上计算感知哈希。
/// API 与 [`PerceptualHasher`](super::phasher::PerceptualHasher) 保持一致，
/// 调用方无需感知后端切换。
pub struct PHasherCpu {
    algorithm: HashAlgorithm,
    hash_size: HashSize,
}

impl PHasherCpu {
    /// 创建 CPU 感知哈希计算器（默认 64bit）。
    pub fn new(algorithm: HashAlgorithm) -> Self {
        Self {
            algorithm,
            hash_size: HashSize::default(),
        }
    }

    /// 创建 CPU 感知哈希计算器，指定哈希位长。
    pub fn with_hash_size(algorithm: HashAlgorithm, hash_size: HashSize) -> Self {
        Self { algorithm, hash_size }
    }

    /// 对已缩放到目标尺寸的灰度像素数据计算感知哈希。
    ///
    /// `images` 中每个元素为一张图像的灰度像素，
    /// `width` 和 `height` 为图像尺寸（应与算法目标尺寸一致）。
    pub fn compute(
        &self,
        images: &[Vec<u8>],
        width: u32,
        height: u32,
    ) -> Result<Vec<u64>, GpuError> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        let hash_bits = self.hash_size.bits() as usize;
        let u32s_per_image = self.hash_size.u32s_per_image() as usize;
        let u64s_per_image = self.hash_size.u64s_per_image() as usize;

        #[cfg(feature = "parallel-cpu")]
        {
            use rayon::prelude::*;
            let image_results: Vec<Vec<u64>> = images
                .par_iter()
                .map(|image| {
                    let hash_u32s = match self.algorithm {
                        HashAlgorithm::Mean => mean_hash_cpu(image, width, height, hash_bits),
                        HashAlgorithm::Median => median_hash_cpu(image, width, height, hash_bits),
                        HashAlgorithm::Gradient => {
                            gradient_hash_cpu(image, width, height, hash_bits)
                        }
                        HashAlgorithm::Block => {
                            block_hash_cpu(image, width, height, hash_bits, self.hash_size.size())
                        }
                        HashAlgorithm::VertGradient => {
                            vert_gradient_hash_cpu(image, width, height, hash_bits)
                        }
                        HashAlgorithm::DoubleGradient => {
                            double_gradient_hash_cpu(image, width, height, hash_bits)
                        }
                        #[cfg(feature = "pdq")]
                        HashAlgorithm::Pdq => {
                            let pdq = PdqHashCpu::new();
                            let result = pdq.compute(image).unwrap();
                            let mut hash_u32s = vec![0u32; 8];
                            for (i, &val) in result.hash.iter().enumerate() {
                                hash_u32s[i * 2] = val as u32;
                                hash_u32s[i * 2 + 1] = (val >> 32) as u32;
                            }
                            hash_u32s
                        }
                    };

                    let mut hashes = Vec::with_capacity(u64s_per_image);
                    for chunk in 0..u64s_per_image {
                        let lo_idx = chunk * 2;
                        let low = hash_u32s[lo_idx] as u64;
                        let high = if lo_idx + 1 < u32s_per_image {
                            hash_u32s[lo_idx + 1] as u64
                        } else {
                            0
                        };
                        hashes.push(low | (high << 32));
                    }
                    hashes
                })
                .collect();
            Ok(image_results.into_iter().flatten().collect())
        }
        #[cfg(not(feature = "parallel-cpu"))]
        {
            let mut all_hashes = Vec::with_capacity(images.len() * u64s_per_image);
            for image in images {
                let hash_u32s = match self.algorithm {
                    HashAlgorithm::Mean => mean_hash_cpu(image, width, height, hash_bits),
                    HashAlgorithm::Median => median_hash_cpu(image, width, height, hash_bits),
                    HashAlgorithm::Gradient => gradient_hash_cpu(image, width, height, hash_bits),
                    HashAlgorithm::Block => {
                        block_hash_cpu(image, width, height, hash_bits, self.hash_size.size())
                    }
                    HashAlgorithm::VertGradient => {
                        vert_gradient_hash_cpu(image, width, height, hash_bits)
                    }
                    HashAlgorithm::DoubleGradient => {
                        double_gradient_hash_cpu(image, width, height, hash_bits)
                    }
                    #[cfg(feature = "pdq")]
                    HashAlgorithm::Pdq => {
                        let pdq = PdqHashCpu::new();
                        let result = pdq.compute(image)?;
                        let mut hash_u32s = vec![0u32; 8];
                        for (i, &val) in result.hash.iter().enumerate() {
                            hash_u32s[i * 2] = val as u32;
                            hash_u32s[i * 2 + 1] = (val >> 32) as u32;
                        }
                        hash_u32s
                    }
                };

                for chunk in 0..u64s_per_image {
                    let lo_idx = chunk * 2;
                    let low = hash_u32s[lo_idx] as u64;
                    let high = if lo_idx + 1 < u32s_per_image {
                        hash_u32s[lo_idx + 1] as u64
                    } else {
                        0
                    };
                    all_hashes.push(low | (high << 32));
                }
            }
            Ok(all_hashes)
        }
    }

    /// 返回算法类型。
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// 返回哈希位长配置。
    pub fn hash_size(&self) -> HashSize {
        self.hash_size
    }
}

impl Default for PHasherCpu {
    fn default() -> Self {
        Self::new(HashAlgorithm::Mean)
    }
}

/// 均值哈希 CPU 实现（与 WGSL mean_hash 着色器逻辑一致）。
///
/// 使用 f32 均值，像素 >= 均值生成 1bit。
fn mean_hash_cpu(pixels: &[u8], width: u32, height: u32, hash_bits: usize) -> Vec<u32> {
    let pixels_per_image = (width as u64 * height as u64) as usize;
    let u32s_per_image = hash_bits.div_ceil(32);
    let total_bits = hash_bits.min(pixels_per_image);

    let sum: f32 = pixels[..pixels_per_image].iter().map(|&p| p as f32).sum();
    let mean = sum / pixels_per_image as f32;

    let mut hash_u32s = vec![0u32; u32s_per_image];
    for i in 0..total_bits {
        if pixels[i] as f32 >= mean {
            hash_u32s[i / 32] |= 1u32 << (i % 32);
        }
    }
    hash_u32s
}

/// 中值哈希 CPU 实现（与 WGSL median_hash 着色器逻辑一致）。
///
/// 使用直方图中值算法，像素 > 中值生成 1bit。
fn median_hash_cpu(pixels: &[u8], width: u32, height: u32, hash_bits: usize) -> Vec<u32> {
    let pixels_per_image = (width as u64 * height as u64) as usize;
    let u32s_per_image = hash_bits.div_ceil(32);
    let total_bits = hash_bits.min(pixels_per_image);

    let mut histogram = [0u32; 256];
    for &pixel in &pixels[..pixels_per_image] {
        histogram[pixel as usize] += 1;
    }
    let half = pixels_per_image as u32 / 2;
    let mut cum = 0u32;
    let mut median = 128u32;
    for v in 0..256u32 {
        cum += histogram[v as usize];
        if cum > half {
            median = v;
            break;
        }
    }

    let mut hash_u32s = vec![0u32; u32s_per_image];
    for i in 0..total_bits {
        if pixels[i] as u32 > median {
            hash_u32s[i / 32] |= 1u32 << (i % 32);
        }
    }
    hash_u32s
}

/// 水平梯度哈希 CPU 实现（与 WGSL gradient_hash 着色器逻辑一致）。
///
/// 每行相邻像素水平比较，右 > 左生成 1bit。
fn gradient_hash_cpu(pixels: &[u8], width: u32, height: u32, hash_bits: usize) -> Vec<u32> {
    let u32s_per_image = hash_bits.div_ceil(32);
    let mut hash_u32s = vec![0u32; u32s_per_image];
    let mut bit_pos: usize = 0;
    let width_usize = width as usize;

    for row in 0..height {
        for col in 0..(width - 1) {
            if bit_pos >= hash_bits {
                break;
            }
            let idx = row as usize * width_usize + col as usize;
            let current = pixels[idx];
            let next = pixels[idx + 1];
            if next > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }
    hash_u32s
}

/// 垂直梯度哈希 CPU 实现（与 WGSL vert_gradient_hash 着色器逻辑一致）。
///
/// 每列相邻像素垂直比较，下 > 上生成 1bit。
fn vert_gradient_hash_cpu(pixels: &[u8], width: u32, height: u32, hash_bits: usize) -> Vec<u32> {
    let u32s_per_image = hash_bits.div_ceil(32);
    let mut hash_u32s = vec![0u32; u32s_per_image];
    let mut bit_pos: usize = 0;
    let width_usize = width as usize;

    for col in 0..width {
        for row in 0..(height - 1) {
            if bit_pos >= hash_bits {
                break;
            }
            let idx = row as usize * width_usize + col as usize;
            let current = pixels[idx];
            let below = pixels[idx + width_usize];
            if below > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }
    hash_u32s
}

/// 分块哈希 CPU 实现（与 WGSL block_hash 着色器逻辑一致）。
///
/// 将图像分为 hash_size × hash_size 块，计算每块均值，
/// 先水平比较相邻块，再垂直比较相邻块。
fn block_hash_cpu(
    pixels: &[u8],
    width: u32,
    height: u32,
    hash_bits: usize,
    hash_size: u32,
) -> Vec<u32> {
    let blocks_x = hash_size;
    let blocks_y = hash_size;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;
    let u32s_per_image = hash_bits.div_ceil(32);
    let width_usize = width as usize;

    let mut block_means = Vec::with_capacity((blocks_x as u64 * blocks_y as u64) as usize);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum: u32 = 0;
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let px = bx * block_w + dx;
                    let py = by * block_h + dy;
                    let idx = py as usize * width_usize + px as usize;
                    sum += pixels[idx] as u32;
                }
            }
            let block_area = block_w as u64 * block_h as u64;
            let mean = (sum as u64 / block_area) as u32;
            block_means.push(mean);
        }
    }

    let mut hash_u32s = vec![0u32; u32s_per_image];
    let mut bit_pos: usize = 0;

    for by in 0..blocks_y {
        if bit_pos >= hash_bits {
            break;
        }
        for bx in 0..(blocks_x - 1) {
            if bit_pos >= hash_bits {
                break;
            }
            let idx = (by * blocks_x + bx) as usize;
            let current = block_means[idx];
            let next = block_means[idx + 1];
            if next > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    for by in 0..(blocks_y - 1) {
        if bit_pos >= hash_bits {
            break;
        }
        for bx in 0..blocks_x {
            if bit_pos >= hash_bits {
                break;
            }
            let idx = (by * blocks_x + bx) as usize;
            let current = block_means[idx];
            let below = block_means[idx + blocks_x as usize];
            if below > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    hash_u32s
}

/// 双梯度哈希 CPU 实现（与 WGSL double_gradient_hash 着色器逻辑一致）。
///
/// 前 hash_bits/2 位为水平梯度，后 hash_bits/2 位为垂直梯度。
fn double_gradient_hash_cpu(
    pixels: &[u8],
    width: u32,
    height: u32,
    hash_bits: usize,
) -> Vec<u32> {
    let u32s_per_image = hash_bits.div_ceil(32);
    let mut hash_u32s = vec![0u32; u32s_per_image];
    let mut bit_pos: usize = 0;
    let h_limit = hash_bits / 2;
    let width_usize = width as usize;

    for row in 0..height {
        for col in 0..(width - 1) {
            if bit_pos >= h_limit {
                break;
            }
            let idx = row as usize * width_usize + col as usize;
            let current = pixels[idx];
            let next = pixels[idx + 1];
            if next > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    for col in 0..width {
        for row in 0..(height - 1) {
            if bit_pos >= hash_bits {
                break;
            }
            let idx = row as usize * width_usize + col as usize;
            let current = pixels[idx];
            let below = pixels[idx + width_usize];
            if below > current {
                hash_u32s[bit_pos / 32] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    hash_u32s
}

/// CPU 高斯模糊（5×5 核，Clamp 边界模式）。
///
/// 与 GPU 端 `GpuGaussianBlur` 逻辑一致：生成 2D 高斯核后直接卷积，
/// 边界像素使用 clamp 处理（与 GPU 的 `BorderMode::Clamp` 一致）。
pub(crate) fn cpu_gaussian_blur(
    pixels: &[u8],
    width: u32,
    height: u32,
    sigma: f32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let pixel_count = w * h;

    if pixel_count == 0 || pixels.len() < pixel_count {
        return pixels.to_vec();
    }

    const KERNEL_SIZE: usize = 5;
    let radius = (KERNEL_SIZE / 2) as i32;

    // 生成 2D 高斯核
    let mut kernel = [[0.0f32; KERNEL_SIZE]; KERNEL_SIZE];
    let mut kernel_sum = 0.0f32;
    for ky in -radius..=radius {
        for kx in -radius..=radius {
            let dist_sq = (kx * kx + ky * ky) as f32;
            let val = (-dist_sq / (2.0 * sigma * sigma)).exp();
            kernel[(ky + radius) as usize][(kx + radius) as usize] = val;
            kernel_sum += val;
        }
    }
    // 归一化
    for row in &mut kernel {
        for v in row {
            *v /= kernel_sum;
        }
    }

    let mut output = vec![0u8; pixel_count];
    for py in 0..h {
        for px in 0..w {
            let mut acc = 0.0f32;
            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let sy = (py as i32 + ky).clamp(0, h as i32 - 1) as usize;
                    let sx = (px as i32 + kx).clamp(0, w as i32 - 1) as usize;
                    acc += pixels[sy * w + sx] as f32
                        * kernel[(ky + radius) as usize][(kx + radius) as usize];
                }
            }
            output[py * w + px] = acc.clamp(0.0, 255.0) as u8;
        }
    }
    output
}
