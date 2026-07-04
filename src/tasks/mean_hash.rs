use crate::tasks::hash_common::{
    HashSize, PerceptualHashComputer, compute_phash_with_thresholds, phash_pipeline_descriptor,
};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

const MEAN_HASH_WGSL: &str = include_str!("mean_hash.wgsl");

/// Mean Hash（均值哈希）GPU 计算器。
///
/// 对每幅灰度图像计算所有像素的平均值，
/// 每个像素与均值比较生成 1bit，最终输出哈希值。
///
/// 与其他哈希算法不同，Mean Hash 在 CPU 端预计算均值，
/// 追加到输入缓冲区末尾传入着色器，避免 GPU 端 O(N²) 循环
/// 导致大 hash_size 下 DX12 着色器编译失败。
pub struct MeanHashComputer {
    pipeline: std::sync::Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    hash_size: HashSize,
}

impl MeanHashComputer {
    /// 创建 Mean Hash 计算器，默认 hash_size=8 (64 bit)。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_config(ctx, [8, 8, 1], HashSize::default())
    }

    /// 创建 Mean Hash 计算器，指定 workgroup_size 和 hash_size。
    pub fn with_config(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &phash_pipeline_descriptor(MEAN_HASH_WGSL, workgroup_size, ctx.push_constants_supported()),
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

/// CPU 端计算每张图像的像素均值，返回 f32 的位模式（u32）。
///
/// 着色器通过 `bitcast<f32>(pixels[base + pixels_per_image])` 读取，
/// 确保与原 WGSL `sum / f32(pixels_per_image)` 计算结果一致。
fn compute_means_as_u32(images: &[Vec<u8>]) -> Vec<u32> {
    images.iter().map(|img| {
        if img.is_empty() { return 0u32; }
        let sum: f32 = img.iter().map(|&p| p as f32).sum();
        let mean = sum / img.len() as f32;
        mean.to_bits()
    }).collect()
}

impl PerceptualHashComputer for MeanHashComputer {
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

        // CPU 端预计算均值（f32 位模式），追加到输入缓冲区
        let means = compute_means_as_u32(images);
        compute_phash_with_thresholds(
            &self.pipeline, ctx, images, width, height,
            self.workgroup_size, hash_size, &means,
        )
    }

    fn pipeline(&self) -> &ComputePipeline { &self.pipeline }
    fn workgroup_size(&self) -> [u32; 3] { self.workgroup_size }
    fn hash_size(&self) -> HashSize { self.hash_size }
}
