//! # czkawka GPU 加速兼容层
//!
//! 提供 [`CzkawkaGpuAccelerator`]，封装共享 GPU 上下文 + 感知哈希 + 汉明距离匹配，
//! 允许跨"哈希计算"和"距离比较"两个阶段复用同一 `GpuContext`，避免重复初始化 GPU。
//!
//! ## 设计目标
//!
//! czkawka 的相似图像检测流程分为两个独立阶段：
//!
//! 1. **哈希计算阶段**：`hash_images` — 对所有图像计算感知哈希
//! 2. **距离比较阶段**：`find_similar_hashes` — 在哈希集合中查找相似对
//!
//! `CzkawkaGpuAccelerator` 让两个阶段共享同一 GPU 上下文，管线缓存复用，
//! 无需在阶段间重新初始化 GPU。
//!
//! ## 使用示例
//!
//! ```no_run
//! use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
//! use gpgpu_tool::tasks::phasher::HashAlgorithm;
//!
//! // 创建加速器（hash_size=8 → 64-bit, Gradient 算法）
//! let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient).unwrap();
//!
//! // 阶段 1：批量计算 RGBA 图像的感知哈希
//! let rgba_images = vec![vec![128u8; 64 * 64 * 4]]; // 1 张 64×64 RGBA 图
//! let dims = vec![(64u32, 64u32)];
//! let hashes = accelerator.compute_hashes(&rgba_images, &dims).unwrap();
//!
//! // 阶段 2：查找相似对（tolerance=10）
//! let pairs = accelerator.find_similar_pairs(&hashes, 10).unwrap();
//! for (parent, child, distance) in &pairs {
//!     println!("相似: {} <-> {} (距离={})", parent, child, distance);
//! }
//! ```

use std::sync::{Arc, Mutex};

use crate::tasks::gpu_matcher::{GpuHashMatcherBytes, GPU_FILTERED_PAIRS_THRESHOLD};
use crate::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use crate::{GpuContext, GpuError, HashBytes, HashSize};

/// czkawka GPU 加速器，封装共享 GPU 上下文 + 哈希计算 + 距离匹配。
///
/// 跨阶段共享同一 `GpuContext`，管线缓存复用，避免重复初始化 GPU。
///
/// # 构造
///
/// 使用 [`new()`](Self::new) 创建，需指定 czkawka 格式的 `hash_size`（8/16/32/64）
/// 和哈希算法。内部使用 [`GpuContext::new_for_integration()`] 初始化 GPU，
/// 失败时返回 [`GpuError::GpuUnavailable`]，调用方可据此降级到 CPU 路径。
///
/// # 两阶段使用
///
/// - **哈希阶段**：[`compute_hashes()`](Self::compute_hashes) — 接受 RGBA 数据，返回 `Vec<HashBytes>`
/// - **匹配阶段**：[`find_similar_pairs()`](Self::find_similar_pairs) — 接受哈希列表，返回相似对三元组
///
/// 两个阶段共享同一 `GpuContext`，无需重新初始化。
pub struct CzkawkaGpuAccelerator {
    /// 共享 GPU 上下文（跨阶段复用）
    ctx: Arc<Mutex<GpuContext>>,
    /// 感知哈希计算器
    hasher: PerceptualHasher,
    /// GPU 汉明距离匹配器
    matcher: GpuHashMatcherBytes,
}

impl CzkawkaGpuAccelerator {
    /// 创建 czkawka GPU 加速器。
    ///
    /// # 参数
    ///
    /// - `hash_size`：czkawka 格式的哈希尺寸（8/16/32/64），对应 64/256/1024/4096 bit
    /// - `hash_alg`：感知哈希算法（Mean/Median/Gradient/Block/VertGradient/DoubleGradient）
    ///
    /// # 错误
    ///
    /// - [`GpuError::GpuUnavailable`]：GPU 不可用，调用方可降级到 czkawka CPU 路径
    /// - [`GpuError::InvalidInput`]：`hash_size` 不在 {8, 16, 32, 64} 范围内
    /// - 其他错误：GPU 初始化或管线编译失败
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    ///
    /// let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient).unwrap();
    /// ```
    pub fn new(hash_size: u8, hash_alg: HashAlgorithm) -> Result<Self, GpuError> {
        let hash_size = HashSize::from_czkawka(hash_size)?;

        // 初始化 GPU 上下文（集成模式：失败时返回 GpuUnavailable）
        let ctx = GpuContext::new_for_integration()?;
        let ctx = Arc::new(Mutex::new(ctx));

        // 锁定上下文，创建哈希器和匹配器（需要 &mut GpuContext 来编译管线）
        let mut ctx_guard = ctx.lock().unwrap();
        let hasher = PerceptualHasher::with_hash_size(&mut ctx_guard, hash_alg, hash_size)?;
        let matcher = GpuHashMatcherBytes::new(&mut ctx_guard)?;
        drop(ctx_guard);

        Ok(Self {
            ctx,
            hasher,
            matcher,
        })
    }

    /// 批量计算 RGBA 图像的感知哈希。
    ///
    /// 内部使用 czkawka 兼容的 RGBA→灰度公式 `(R*77 + G*150 + B*29) >> 8`，
    /// 然后通过 GPU 流水线计算感知哈希。
    ///
    /// # 参数
    ///
    /// - `rgba_images`：RGBA8888 格式的图像数据切片，每个元素为一张图的像素数据
    /// - `dims`：图像尺寸切片，与 `rgba_images` 一一对应
    ///
    /// # 返回
    ///
    /// `Vec<HashBytes>` — 每张图对应一个 `HashBytes`（对应 czkawka `ImHash = Vec<u8>`）
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    ///
    /// let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Mean).unwrap();
    /// let images = vec![vec![128u8; 64 * 64 * 4]];
    /// let dims = vec![(64u32, 64u32)];
    /// let hashes = accelerator.compute_hashes(&images, &dims).unwrap();
    /// assert_eq!(hashes.len(), 1);
    /// ```
    pub fn compute_hashes(
        &self,
        rgba_images: &[Vec<u8>],
        dims: &[(u32, u32)],
    ) -> Result<Vec<HashBytes>, GpuError> {
        let ctx = self.ctx.lock().unwrap();
        self.hasher
            .compute_from_rgba_to_hash_bytes(&ctx, rgba_images, dims)
    }

    /// 查找相似哈希对（czkawka 兼容格式）。
    ///
    /// 对称模式：在同一个哈希集合内查找所有距离 ≤ `tolerance` 的对。
    ///
    /// # 参数
    ///
    /// - `hashes`：哈希列表
    /// - `tolerance`：汉明距离容差（绝对值，非归一化）
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(parent_idx, child_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。已过滤 `parent_idx == child_idx` 的自身匹配。
    ///
    /// 对应 czkawka `gpu_compare_hashes_auto()` 的输出格式。
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    /// use gpgpu_tool::HashBytes;
    ///
    /// let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient).unwrap();
    /// let hashes = vec![
    ///     HashBytes::from_u64(0x0000),
    ///     HashBytes::from_u64(0x0001),
    /// ];
    /// let pairs = accelerator.find_similar_pairs(&hashes, 5).unwrap();
    /// for (parent, child, dist) in &pairs {
    ///     println!("相似: {} <-> {} (距离={})", parent, child, dist);
    /// }
    /// ```
    pub fn find_similar_pairs(
        &self,
        hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        let ctx = self.ctx.lock().unwrap();
        // 大规模数据集使用 GPU 端 threshold 过滤，避免下载完整 N×N 距离矩阵
        if hashes.len() >= GPU_FILTERED_PAIRS_THRESHOLD {
            self.matcher
                .compute_similar_pairs_gpu_filtered(&ctx, hashes, tolerance)
        } else {
            self.matcher.compute_similar_pairs(&ctx, hashes, tolerance)
        }
    }

    /// 非对称模式查找相似哈希对。
    ///
    /// 在 `ref_hashes`（参考文件夹）和 `normal_hashes`（普通文件夹）之间查找
    /// 所有距离 ≤ `tolerance` 的对。对应 czkawka `gpu_compare_hashes_asymmetric()`。
    ///
    /// # 返回
    ///
    /// `Vec<(u32, u32, u32)>` — `(ref_idx, normal_idx, distance)` 三元组列表，
    /// 按 distance 升序排序。
    pub fn find_similar_pairs_asymmetric(
        &self,
        ref_hashes: &[HashBytes],
        normal_hashes: &[HashBytes],
        tolerance: u32,
    ) -> Result<Vec<(u32, u32, u32)>, GpuError> {
        let ctx = self.ctx.lock().unwrap();
        // 大规模数据集使用 GPU 端 threshold 过滤
        let total = ref_hashes.len() * normal_hashes.len();
        if total >= GPU_FILTERED_PAIRS_THRESHOLD * GPU_FILTERED_PAIRS_THRESHOLD {
            self.matcher
                .compute_similar_pairs_asymmetric_gpu_filtered(
                    &ctx, ref_hashes, normal_hashes, tolerance,
                )
        } else {
            self.matcher
                .compute_similar_pairs_asymmetric(&ctx, ref_hashes, normal_hashes, tolerance)
        }
    }

    /// 获取共享 GPU 上下文的引用（用于高级用法）。
    ///
    /// 返回 `Arc<Mutex<GpuContext>>`，调用方可以锁定后执行自定义 GPU 操作。
    pub fn shared_context(&self) -> &Arc<Mutex<GpuContext>> {
        &self.ctx
    }

    /// 获取感知哈希计算器的引用（用于高级用法）。
    pub fn hasher(&self) -> &PerceptualHasher {
        &self.hasher
    }
}
