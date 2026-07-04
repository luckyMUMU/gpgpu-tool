//! 端到端 GPU 图像匹配器。
//!
//! 组合 [`PerceptualHasher`]（GPU 感知哈希）和 [`GpuHashMatcherBytes`]（GPU 距离矩阵），
//! 提供一站式 API：批量图像 → GPU 哈希 → GPU 并行匹配。
//!
//! 支持任意 [`HashSize`]（8/16/32），内部自动将 `Vec<u64>` 转换为 [`HashBytes`]
//! 以正确处理 64-bit 到 1024-bit 的变长哈希。
//!
//! # 使用示例
//!
//! ```no_run
//! use gpgpu_tool::GpuContext;
//! use gpgpu_tool::tasks::gpu_image_matcher::GpuImageMatcher;
//! use gpgpu_tool::tasks::phasher::HashAlgorithm;
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//! let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
//!
//! // 准备图像数据（灰度格式）和尺寸
//! let query_images = vec![vec![128u8; 64 * 64]];
//! let query_dims = vec![(64u32, 64u32)];
//! let db_images = vec![vec![200u8; 64 * 64]];
//! let db_dims = vec![(64u32, 64u32)];
//!
//! let results = matcher.find_similar(
//!     &ctx, &query_images, &query_dims, &db_images, &db_dims, 10
//! ).unwrap();
//! ```

use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::gpu_matcher::GpuHashMatcherBytes;
use crate::tasks::hash_bytes::HashBytes;
use crate::tasks::matcher_bytes::MatchResultBytes;
use crate::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use crate::HashSize;

/// 端到端 GPU 图像匹配器。
///
/// 组合 [`PerceptualHasher`]（GPU 感知哈希）和 [`GpuHashMatcherBytes`]（GPU 距离矩阵），
/// 提供一站式 API：批量图像 → GPU 哈希 → GPU 并行匹配。
///
/// 支持任意 [`HashSize`]，当 `hash_size ≥ 16` 时产生 256-bit 或更长的哈希，
/// 内部使用 [`GpuHashMatcherBytes`] 正确处理变长哈希的距离计算。
pub struct GpuImageMatcher {
    hasher: PerceptualHasher,
    gpu_matcher: GpuHashMatcherBytes,
    hash_size: HashSize,
}

impl GpuImageMatcher {
    /// 创建端到端 GPU 图像匹配器，使用默认哈希尺寸（8×8 = 64-bit）。
    pub fn new(ctx: &mut GpuContext, algorithm: HashAlgorithm) -> Result<Self, GpuError> {
        Self::with_hash_size(ctx, algorithm, HashSize::default())
    }

    /// 创建端到端 GPU 图像匹配器，指定哈希尺寸。
    pub fn with_hash_size(
        ctx: &mut GpuContext,
        algorithm: HashAlgorithm,
        hash_size: HashSize,
    ) -> Result<Self, GpuError> {
        let hasher = PerceptualHasher::with_hash_size(ctx, algorithm, hash_size)?;
        let gpu_matcher = GpuHashMatcherBytes::new(ctx)?;
        Ok(Self {
            hasher,
            gpu_matcher,
            hash_size,
        })
    }

    /// 获取内部的感知哈希计算器引用。
    pub fn hasher(&self) -> &PerceptualHasher {
        &self.hasher
    }

    /// 获取内部的 GPU 距离匹配器引用。
    pub fn gpu_matcher(&self) -> &GpuHashMatcherBytes {
        &self.gpu_matcher
    }

    /// 获取哈希尺寸。
    pub fn hash_size(&self) -> HashSize {
        self.hash_size
    }

    /// 对 query_images 在 database_images 中查找相似图像。
    ///
    /// 返回每个 query 的匹配结果（distance ≤ threshold）。
    pub fn find_similar(
        &self,
        ctx: &GpuContext,
        query_images: &[Vec<u8>],
        query_dims: &[(u32, u32)],
        database_images: &[Vec<u8>],
        database_dims: &[(u32, u32)],
        threshold: u32,
    ) -> Result<Vec<Vec<MatchResultBytes>>, GpuError> {
        if query_images.is_empty() {
            return Ok(vec![]);
        }
        if database_images.is_empty() {
            return Ok(vec![vec![]; query_images.len()]);
        }

        // Step 1: 计算哈希（返回 Vec<u64>，可能包含多字哈希）
        let query_hashes_u64 = self.hasher.compute(ctx, query_images, query_dims)?;
        let db_hashes_u64 = self.hasher.compute(ctx, database_images, database_dims)?;

        // Step 2: 将 Vec<u64> 按 hash_size 分组转换为 Vec<HashBytes>
        let query_hashes = u64s_to_hash_bytes(&query_hashes_u64, self.hash_size);
        let db_hashes = u64s_to_hash_bytes(&db_hashes_u64, self.hash_size);

        // Step 3: 使用变长哈希 GPU 距离矩阵计算
        let matrix = self
            .gpu_matcher
            .compute_distance_matrix(ctx, &query_hashes, &db_hashes)?;

        // Step 4: CPU 端过滤 threshold 并构造 MatchResultBytes
        let results = matrix
            .iter()
            .map(|distances| {
                distances
                    .iter()
                    .enumerate()
                    .filter(|(_, &dist)| dist <= threshold)
                    .map(|(di, &dist)| MatchResultBytes {
                        hash: db_hashes[di].clone(),
                        distance: dist,
                        quality: None,
                    })
                    .collect()
            })
            .collect();

        Ok(results)
    }

    /// 计算 query 和 database 图像之间的完整距离矩阵。
    ///
    /// 返回 `Vec<Vec<u32>>`，外层长度 = query 数量，内层长度 = database 数量。
    pub fn compute_distance_matrix(
        &self,
        ctx: &GpuContext,
        query_images: &[Vec<u8>],
        query_dims: &[(u32, u32)],
        database_images: &[Vec<u8>],
        database_dims: &[(u32, u32)],
    ) -> Result<Vec<Vec<u32>>, GpuError> {
        if query_images.is_empty() || database_images.is_empty() {
            return Ok(vec![vec![0; database_images.len()]; query_images.len()]);
        }

        // Step 1: 计算哈希
        let query_hashes_u64 = self.hasher.compute(ctx, query_images, query_dims)?;
        let db_hashes_u64 = self.hasher.compute(ctx, database_images, database_dims)?;

        // Step 2: 转换为 HashBytes
        let query_hashes = u64s_to_hash_bytes(&query_hashes_u64, self.hash_size);
        let db_hashes = u64s_to_hash_bytes(&db_hashes_u64, self.hash_size);

        // Step 3: 计算距离矩阵
        self.gpu_matcher
            .compute_distance_matrix(ctx, &query_hashes, &db_hashes)
    }

    /// 仅计算图像哈希，不做匹配。
    ///
    /// 返回 `Vec<HashBytes>`，每个元素为一张图像的感知哈希。
    pub fn compute_hashes(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dims: &[(u32, u32)],
    ) -> Result<Vec<HashBytes>, GpuError> {
        let hashes_u64 = self.hasher.compute(ctx, images, dims)?;
        Ok(u64s_to_hash_bytes(&hashes_u64, self.hash_size))
    }
}

/// 将 `Vec<u64>` 按 `hash_size` 分组转换为 `Vec<HashBytes>`。
///
/// `PerceptualHasher::compute()` 返回的 `Vec<u64>` 中，
/// 每张图像占 `u64s_per_image` 个 u64（hash_size=8 时 1 个，hash_size=16 时 4 个，hash_size=32 时 16 个）。
/// 此函数按组切分，每组转为一个 `HashBytes`。
fn u64s_to_hash_bytes(u64s: &[u64], hash_size: HashSize) -> Vec<HashBytes> {
    let u64s_per_image = hash_size.u64s_per_image() as usize;
    let image_count = u64s.len() / u64s_per_image;
    let mut result = Vec::with_capacity(image_count);
    for i in 0..image_count {
        let start = i * u64s_per_image;
        let end = start + u64s_per_image;
        result.push(HashBytes::from_u64s(&u64s[start..end]));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64s_to_hash_bytes_size8() {
        // hash_size=8: 每 1 个 u64 为一个哈希
        let u64s = vec![0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210];
        let hashes = u64s_to_hash_bytes(&u64s, HashSize::new(8));
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0].byte_len(), 8);
        assert_eq!(hashes[1].byte_len(), 8);
    }

    #[test]
    fn test_u64s_to_hash_bytes_size16() {
        // hash_size=16: 每 4 个 u64 为一个哈希（256-bit）
        let u64s = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let hashes = u64s_to_hash_bytes(&u64s, HashSize::new(16));
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0].byte_len(), 32);
        assert_eq!(hashes[1].byte_len(), 32);
    }

    #[test]
    fn test_u64s_to_hash_bytes_size32() {
        // hash_size=32: 每 16 个 u64 为一个哈希（1024-bit）
        let u64s: Vec<u64> = (0..32).collect();
        let hashes = u64s_to_hash_bytes(&u64s, HashSize::new(32));
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[0].byte_len(), 128);
        assert_eq!(hashes[1].byte_len(), 128);
    }
}
