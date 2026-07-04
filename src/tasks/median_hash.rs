use crate::tasks::hash_common::{
    HashSize, PerceptualHashComputer, compute_phash_with_thresholds, phash_pipeline_descriptor,
};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

const MEDIAN_HASH_WGSL: &str = include_str!("median_hash.wgsl");

/// Median Hash（中值哈希）GPU 计算器。
///
/// 对每幅灰度图像计算所有像素的中值，
/// 每个像素与中值比较生成 1bit，最终输出哈希值。
///
/// 与其他哈希算法不同，Median Hash 在 CPU 端预计算中位数，
/// 追加到输入缓冲区末尾传入着色器，避免 GPU 端 O(N²) 循环
/// 导致大 hash_size 下 DX12 着色器编译失败。
pub struct MedianHashComputer {
    pipeline: std::sync::Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
}

impl MedianHashComputer {
    /// 创建 Median Hash 计算器，默认 hash_size=8 (64 bit)。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_config(ctx, [8, 8, 1], HashSize::default())
    }

    /// 创建 Median Hash 计算器，指定 workgroup_size 和 hash_size。
    pub fn with_config(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &phash_pipeline_descriptor(MEDIAN_HASH_WGSL, workgroup_size, ctx.push_constants_supported()),
        )?;
        Ok(Self {
            pipeline: std::sync::Arc::clone(&pipeline),
            workgroup_size,
            hash_size,
        })
    }

    pub fn pipeline(&self) -> &ComputePipeline { &self.pipeline }
    pub fn workgroup_size_val(&self) -> [u32; 3] { self.workgroup_size }
    pub fn hash_size_val(&self) -> HashSize { self.hash_size }
}

/// CPU 端计算每张图像的像素中位数（直方图法，与 WGSL 着色器一致）。
///
/// 使用 256-bin 直方图累加，当累计计数超过 pixel_count/2 时返回该 bin 值。
/// 这与原 WGSL `compute_median` 函数逻辑一致，确保 CPU/GPU 结果相同。
fn compute_medians(images: &[Vec<u8>]) -> Vec<u32> {
    images.iter().map(|img| {
        if img.is_empty() { return 0u32; }
        let mut histogram = [0u32; 256];
        for &pixel in img {
            histogram[pixel as usize] += 1;
        }
        let half = img.len() as u32 / 2;
        let mut count = 0u32;
        for v in 0..256u32 {
            count += histogram[v as usize];
            if count > half {
                return v;
            }
        }
        128u32 // 回退值
    }).collect()
}

impl PerceptualHashComputer for MedianHashComputer {
    fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        self.compute_sized(ctx, images, self.hash_size)
    }

    fn compute_sized(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        hash_size: HashSize,
    ) -> Result<Vec<u64>, GpuError> {
        if images.is_empty() { return Ok(vec![]); }
        let pixels_per_image = images[0].len();
        let width = (pixels_per_image as f64).sqrt() as u32;
        let height = width;

        // CPU 端预计算中位数，追加到输入缓冲区
        let medians = compute_medians(images);
        compute_phash_with_thresholds(
            &self.pipeline, ctx, images, width, height,
            self.workgroup_size, hash_size, &medians,
        )
    }

    fn pipeline(&self) -> &ComputePipeline { &self.pipeline }
    fn workgroup_size(&self) -> [u32; 3] { self.workgroup_size }
    fn hash_size(&self) -> HashSize { self.hash_size }
}
