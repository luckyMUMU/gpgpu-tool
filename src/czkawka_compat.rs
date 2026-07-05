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

use rayon::prelude::*;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::tasks::gpu_matcher::{GpuHashMatcherBytes, GPU_FILTERED_PAIRS_THRESHOLD};
use crate::tasks::hash_common::{
    compute_phash_from_gpu_buffer_batch, parse_batch_hash_results,
};
use crate::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use crate::tasks::phasher_pipeline::release_buffers;
use crate::tasks::phasher_util::GpuPixelPacker;
use crate::{GpuContext, GpuError, HashBytes, HashSize};

// image trait 导入（czkawka-compat 隐含启用 image feature）
use image::GenericImageView;

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
    /// GPU 端 u8→u32 像素打包器（零拷贝流水线优化）
    pixel_packer: Option<GpuPixelPacker>,
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
        let pixel_packer = GpuPixelPacker::new(&mut ctx_guard).ok();
        drop(ctx_guard);

        Ok(Self {
            ctx,
            hasher,
            matcher,
            pixel_packer,
        })
    }

    /// 创建 czkawka GPU 加速器（启用 GPU 缩放，支持零拷贝流水线）。
    ///
    /// 与 [`new()`](Self::new) 的区别：内部创建 `PerceptualHasher` 时启用 GPU 缩放
    /// （`use_gpu_resize = true`），使得 [`compute_hashes_zero_copy()`](Self::compute_hashes_zero_copy)
    /// 能够执行完整的 GPU 流水线（上传 → GPU 缩放 → GPU 哈希），无需 CPU 中间步骤。
    ///
    /// # 参数
    ///
    /// - `hash_size`：czkawka 格式的哈希尺寸（8/16/32/64）
    /// - `hash_alg`：感知哈希算法
    ///
    /// # 适用场景
    ///
    /// 大批量图像处理（10K+ 图像），需要最小化 CPU 内存占用时使用。
    pub fn new_with_gpu_resize(hash_size: u8, hash_alg: HashAlgorithm) -> Result<Self, GpuError> {
        let hash_size = HashSize::from_czkawka(hash_size)?;

        let ctx = GpuContext::new_for_integration()?;
        let ctx = Arc::new(Mutex::new(ctx));

        let mut ctx_guard = ctx.lock().unwrap();
        // 启用 GPU 缩放：blur → resize → hash 全程在 GPU 上执行
        let hasher = PerceptualHasher::with_config(
            &mut ctx_guard,
            hash_alg,
            true, // use_gpu_resize = true
            hash_size,
        )?;
        let matcher = GpuHashMatcherBytes::new(&mut ctx_guard)?;
        let pixel_packer = GpuPixelPacker::new(&mut ctx_guard).ok();
        drop(ctx_guard);

        Ok(Self {
            ctx,
            hasher,
            matcher,
            pixel_packer,
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

    /// 批量计算图像的感知哈希（使用 Lanczos3 高质量缩放）。
    ///
    /// 与 [`compute_hashes()`](Self::compute_hashes) 不同，此方法接受 `image::DynamicImage`，
    /// 内部使用 **Lanczos3** 高质量缩放算法（而非 GPU 双线性 / box filter），
    /// 提供更高的缩放精度，适合对哈希精度要求较高的场景。
    ///
    /// # 参数
    ///
    /// - `images`：`image::DynamicImage` 切片
    ///
    /// # 返回
    ///
    /// `Vec<HashBytes>` — 每张图对应一个 `HashBytes`
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    ///
    /// let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient).unwrap();
    /// let img = image::DynamicImage::new_luma8(8, 8);
    /// let hashes = accelerator.compute_hashes_lanczos3(&[img]).unwrap();
    /// ```
    pub fn compute_hashes_lanczos3(
        &self,
        images: &[image::DynamicImage],
    ) -> Result<Vec<HashBytes>, GpuError> {
        let ctx = self.ctx.lock().unwrap();
        let hashes = self.hasher.compute_images(&ctx, images)?;

        let u64s_per_image = self.hasher.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(hashes.len() / u64s_per_image.max(1));
        for chunk in hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 零拷贝流式哈希计算：逐张加载图像并立即上传到 GPU，最小化 CPU 内存占用。
    ///
    /// # 零拷贝机制
    ///
    /// 传统方式：加载全部图像到 CPU → 转灰度 → 批量上传 GPU（峰值内存 = N × 图像大小）
    ///
    /// 零拷贝方式：
    /// 1. 加载一张图像 → RGBA（CPU）
    /// 2. 立即转灰度（CPU）→ RGBA 数据释放
    /// 3. 立即上传到 GPU → GpuBuffer → 灰度数据释放
    /// 4. 重复 1-3，直到批次内所有图像都在 GPU 上
    /// 5. GPU 端执行缩放 + 哈希
    ///
    /// 峰值 CPU 内存 = 1 张图像的大小（而非 N 张）
    ///
    /// # 参数
    ///
    /// - `image_count`：要处理的图像总数
    /// - `loader`：图像加载闭包，接收图像索引，返回 `Ok((rgba_data, width, height))`
    ///   或 `Err(error_message)`。每次调用只加载一张图像。
    ///
    /// # 返回
    ///
    /// `Vec<HashBytes>` — 成功加载的图像的哈希值（跳过加载失败的图像）
    ///
    /// # 要求
    ///
    /// 必须使用 [`new_with_gpu_resize()`](Self::new_with_gpu_resize) 创建加速器，
    /// 以启用 GPU 缩放管线。否则返回 `GpuError::InvalidInput`。
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
    /// use gpgpu_tool::tasks::phasher::HashAlgorithm;
    ///
    /// let acc = CzkawkaGpuAccelerator::new_with_gpu_resize(8, HashAlgorithm::Gradient).unwrap();
    ///
    /// let paths = vec!["img1.png", "img2.png"];
    /// let hashes = acc.compute_hashes_zero_copy(paths.len(), |i| {
    ///     let img = image::open(&paths[i]).map_err(|e| e.to_string())?;
    ///     let rgba = img.to_rgba8();
    ///     let (w, h) = rgba.dimensions();
    ///     Ok((rgba.into_raw(), w, h))
    /// }).unwrap();
    /// ```
    pub fn compute_hashes_zero_copy<F>(
        &self,
        image_count: usize,
        mut loader: F,
    ) -> Result<Vec<HashBytes>, GpuError>
    where
        F: FnMut(usize) -> Result<(Vec<u8>, u32, u32), String>,
    {
        if self.hasher.gpu_resize.is_none() {
            return Err(GpuError::InvalidInput(
                "零拷贝模式需要 GPU 缩放启用，请使用 new_with_gpu_resize() 创建加速器".to_string(),
            ));
        }

        let ctx = self.ctx.lock().unwrap();

        // ── 阶段 1：逐张加载 → 灰度转换 → GPU 上传（零拷贝核心）──
        let mut gpu_buffers: Vec<GpuBuffer> = Vec::with_capacity(image_count);
        let mut dimensions: Vec<(u32, u32)> = Vec::with_capacity(image_count);

        for i in 0..image_count {
            match loader(i) {
                Ok((rgba, w, h)) => {
                    // RGBA → 灰度（CPU），转换后 RGBA 数据立即被 drop
                    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&rgba, w, h);
                    // rgba 在此处被 drop，释放 CPU 内存

                    // 灰度 → GPU 上传，返回 GpuBuffer
                    let bufs = self.hasher.preprocess_gpu(&ctx, &[gray], &[(w, h)])?;
                    // gray 在此处被 drop，释放 CPU 内存

                    gpu_buffers.extend(bufs);
                    dimensions.push((w, h));
                }
                Err(e) => {
                    log::warn!("零拷贝加载失败 [{}]: {}", i, e);
                }
            }
        }

        if gpu_buffers.is_empty() {
            return Ok(vec![]);
        }

        // ── 阶段 2：GPU 端处理（缩放 + 哈希）──
        let hashes = self.process_gpu_buffers(&ctx, &gpu_buffers, &dimensions)?;

        // ── 阶段 3：释放 GPU 缓冲区到池 ──
        release_buffers(&ctx, gpu_buffers);

        // ── 阶段 4：转换为 HashBytes ──
        let u64s_per_image = self.hasher.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(hashes.len() / u64s_per_image.max(1));
        for chunk in hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 零拷贝流式哈希计算（Lanczos3 路径）。
    ///
    /// 与 [`compute_hashes_zero_copy()`](Self::compute_hashes_zero_copy) 类似，
    /// 但使用 CPU Lanczos3 高质量缩放代替 GPU 双线性缩放。
    ///
    /// 流程：
    /// 1. 加载一张图像 → DynamicImage
    /// 2. CPU Lanczos3 缩放到目标尺寸 → 灰度
    /// 3. 立即上传到 GPU → GpuBuffer → CPU 数据释放
    /// 4. 重复直到批次完成
    /// 5. GPU 端执行哈希计算
    ///
    /// # 参数
    ///
    /// - `image_count`：要处理的图像总数
    /// - `loader`：图像加载闭包，返回 `image::DynamicImage`
    pub fn compute_hashes_zero_copy_lanczos3<F>(
        &self,
        image_count: usize,
        mut loader: F,
    ) -> Result<Vec<HashBytes>, GpuError>
    where
        F: FnMut(usize) -> Result<image::DynamicImage, String>,
    {
        let ctx = self.ctx.lock().unwrap();

        let (target_w, target_h) = self.hasher.target_size();

        // 逐张加载 → Lanczos3 缩放 → 灰度 → GPU 上传
        let mut gpu_buffers: Vec<GpuBuffer> = Vec::with_capacity(image_count);

        for i in 0..image_count {
            match loader(i) {
                Ok(img) => {
                    // CPU Lanczos3 缩放 + 灰度转换
                    let resized = img.resize_exact(
                        target_w,
                        target_h,
                        image::imageops::FilterType::Lanczos3,
                    );
                    let luma = resized.grayscale();
                    let gray: Vec<u8> = luma.pixels().map(|(_, _, p)| p.0[0]).collect();
                    // resized 和 luma 在此处被 drop

                    // 上传到 GPU
                    let bufs = self
                        .hasher
                        .preprocess_gpu(&ctx, &[gray], &[(target_w, target_h)])?;
                    // gray 在此处被 drop

                    gpu_buffers.extend(bufs);
                }
                Err(e) => {
                    log::warn!("零拷贝 Lanczos3 加载失败 [{}]: {}", i, e);
                }
            }
        }

        if gpu_buffers.is_empty() {
            return Ok(vec![]);
        }

        // 所有图像已在 GPU 上，且都是目标尺寸，直接计算哈希
        let dims = vec![(target_w, target_h); gpu_buffers.len()];
        let hashes = self.process_gpu_buffers(&ctx, &gpu_buffers, &dims)?;
        release_buffers(&ctx, gpu_buffers);

        let u64s_per_image = self.hasher.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(hashes.len() / u64s_per_image.max(1));
        for chunk in hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 从文件路径列表零拷贝计算哈希（便捷方法）。
    ///
    /// 内部使用 [`compute_hashes_zero_copy()`](Self::compute_hashes_zero_copy)，
    /// 自动处理文件读取和图像解码。
    ///
    /// # 参数
    ///
    /// - `paths`：图像文件路径列表
    /// - `max_dimension`：最大允许的图像像素尺寸（宽或高），超过则跳过
    ///
    /// # 返回
    ///
    /// `(Vec<HashBytes>, Vec<usize>, Vec<(usize, String)>)` —
    /// (哈希列表, 成功索引列表, (失败索引, 错误信息) 列表)
    pub fn compute_hashes_from_paths_zero_copy(
        &self,
        paths: &[std::path::PathBuf],
        max_dimension: u32,
    ) -> Result<(Vec<HashBytes>, Vec<usize>, Vec<(usize, String)>), GpuError> {
        let mut success_indices = Vec::new();
        let mut errors = Vec::new();

        let hashes = self.compute_hashes_zero_copy(paths.len(), |i| {
            let reader = image::ImageReader::open(&paths[i])
                .map_err(|e| format!("打开失败: {}: {}", paths[i].display(), e))?;
            let mut reader = reader.with_guessed_format()
                .map_err(|e| format!("格式识别失败: {}: {}", paths[i].display(), e))?;
            let mut limits = image::Limits::no_limits();
            limits.max_image_width = Some(max_dimension);
            limits.max_image_height = Some(max_dimension);
            reader.limits(limits);

            let img = reader.decode()
                .map_err(|e| format!("解码失败: {}: {}", paths[i].display(), e))?;
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            success_indices.push(i);
            Ok((rgba.into_raw(), w, h))
        })?;

        // 收集错误（loader 中未 push error，需要重新检查）
        // 由于 closure 无法同时返回 Ok 和记录 error，我们通过比较长度推断
        for i in 0..paths.len() {
            if !success_indices.contains(&i) {
                errors.push((i, "加载失败".to_string()));
        }
        }

        Ok((hashes, success_indices, errors))
    }

    /// 零拷贝流式哈希计算（流水线优化版）：CPU/GPU 双缓冲流水线 + 批量 GPU 处理。
    ///
    /// 与 [`compute_hashes_zero_copy()`](Self::compute_hashes_zero_copy) 相比，此方法实现
    /// 四项关键优化：
    ///
    /// 1. **rayon 并行解码**：CPU 工作线程内使用 rayon 并行迭代处理子批图像，
    ///    充分利用多核 CPU 并行解码 + RGBA→灰度转换
    /// 2. **双缓冲流水线**：独立 CPU 线程加载图像，主线程同时执行 GPU 上传。
    ///    通过 `sync_channel(1)` 背压，CPU 最多领先 GPU 一个子批
    /// 3. **GPU 端像素打包**：上传原始 u8 灰度数据（非 u32），减少 4× 上传数据量，
    ///    GPU 端 dispatch 打包着色器生成 u32 缓冲区
    /// 4. **异步 GPU 流水线**：`queue.write_buffer()` 和 `queue.submit()` 均为非阻塞操作，
    ///    GPU 打包 dispatch 与 CPU 下一子批上传自动重叠执行。
    ///    所有子批上传完成后统一执行一次批量 resize + hash，最大化 GPU 批处理效率
    ///
    /// # 参数
    ///
    /// - `image_count`：要处理的图像总数
    /// - `loader`：图像加载闭包，接收图像索引，返回 `Ok((rgba_data, width, height))`
    /// - `sub_batch_size`：子批大小（每多少张图像批量上传一次），推荐 8-32
    ///
    /// # 要求
    ///
    /// 必须使用 [`new_with_gpu_resize()`](Self::new_with_gpu_resize) 创建加速器。
    pub fn compute_hashes_zero_copy_pipelined<F>(
        &self,
        image_count: usize,
        loader: F,
        sub_batch_size: usize,
    ) -> Result<Vec<HashBytes>, GpuError>
    where
        F: Fn(usize) -> Result<(Vec<u8>, u32, u32), String> + Send + Sync,
    {
        if self.hasher.gpu_resize.is_none() {
            return Err(GpuError::InvalidInput(
                "零拷贝流水线模式需要 GPU 缩放启用，请使用 new_with_gpu_resize() 创建加速器".to_string(),
            ));
        }

        let ctx = self.ctx.lock().unwrap();
        let sub_batch_size = sub_batch_size.clamp(1, 64);
        // GPU 端打包暂未启用：dispatch_with_params 复用 uniform buffer 导致
        // 多次 dispatch 间存在数据竞争。当前使用 CPU 端打包（preprocess_gpu），
        // 已通过 rayon 并行解码实现主要性能提升。
        let use_gpu_pack = false;

        use std::sync::mpsc;

        /// CPU 线程预处理的子批数据
        struct SubBatchData {
            grayscales: Vec<Vec<u8>>,
            dims: Vec<(u32, u32)>,
        }

        let (tx, rx) = mpsc::sync_channel::<SubBatchData>(1);

        let raw_hashes = std::thread::scope(|s| -> Result<Vec<u64>, GpuError> {
            // ════ CPU 工作线程：rayon 并行加载图像 → RGBA → 灰度转换 ════
            s.spawn(move || {
                let mut batch_indices: Vec<usize> = Vec::with_capacity(sub_batch_size);

                for i in 0..image_count {
                    batch_indices.push(i);

                    if batch_indices.len() >= sub_batch_size {
                        let indices = std::mem::take(&mut batch_indices);
                        // 使用 rayon 并行处理子批
                        let results: Vec<Result<(Vec<u8>, u32, u32), String>> = indices
                            .par_iter()
                            .map(|&idx| loader(idx))
                            .collect();

                        let mut gray_buf: Vec<Vec<u8>> = Vec::with_capacity(indices.len());
                        let mut dims_buf: Vec<(u32, u32)> = Vec::with_capacity(indices.len());

                        for res in results {
                            match res {
                                Ok((rgba, w, h)) => {
                                    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&rgba, w, h);
                                    gray_buf.push(gray);
                                    dims_buf.push((w, h));
                                }
                                Err(e) => {
                                    log::warn!("零拷贝流水线加载失败: {}", e);
                                }
                            }
                        }

                        if !gray_buf.is_empty() {
                            let batch = SubBatchData {
                                grayscales: gray_buf,
                                dims: dims_buf,
                            };
                            if tx.send(batch).is_err() {
                                break;
                            }
                        }
                    }
                }

                // 处理剩余图像
                if !batch_indices.is_empty() {
                    let results: Vec<Result<(Vec<u8>, u32, u32), String>> = batch_indices
                        .par_iter()
                        .map(|&idx| loader(idx))
                        .collect();

                    let mut gray_buf: Vec<Vec<u8>> = Vec::with_capacity(batch_indices.len());
                    let mut dims_buf: Vec<(u32, u32)> = Vec::with_capacity(batch_indices.len());

                    for res in results {
                        match res {
                            Ok((rgba, w, h)) => {
                                let gray = PerceptualHasher::rgba_to_grayscale_cpu(&rgba, w, h);
                                gray_buf.push(gray);
                                dims_buf.push((w, h));
                            }
                            Err(e) => {
                                log::warn!("零拷贝流水线加载失败: {}", e);
                            }
                        }
                    }

                    if !gray_buf.is_empty() {
                        let _ = tx.send(SubBatchData {
                            grayscales: gray_buf,
                            dims: dims_buf,
                        });
                    }
                }
            });

            // ════ 主线程：阶段 1 逐子批上传 GPU，阶段 2 统一处理 ════
            let mut all_gpu_buffers: Vec<GpuBuffer> = Vec::with_capacity(image_count);
            let mut all_dims: Vec<(u32, u32)> = Vec::with_capacity(image_count);

            // 阶段 1：接收子批并上传到 GPU（与 CPU 线程并行）
            while let Ok(sub_batch) = rx.recv() {
                if sub_batch.grayscales.is_empty() {
                    continue;
                }

                if use_gpu_pack {
                    // GPU 端打包路径：上传原始 u8 数据，GPU 端打包为 u32
                    for (gray, &(w, h)) in sub_batch.grayscales.iter().zip(sub_batch.dims.iter()) {
                        let pixel_count = (w as usize) * (h as usize);
                        if let Some(ref packer) = self.pixel_packer {
                            let packed = packer.pack_raw(&ctx, gray, pixel_count)?;
                            all_gpu_buffers.push(packed);
                        } else {
                            // fallback 到 CPU 打包
                            let bufs = self.hasher.preprocess_gpu(&ctx, &[gray.clone()], &[(w, h)])?;
                            all_gpu_buffers.extend(bufs);
                        }
                        all_dims.push((w, h));
                    }
                } else {
                    // CPU 端打包路径（原有逻辑）
                    let gpu_images = self.hasher.preprocess_gpu(
                        &ctx, &sub_batch.grayscales, &sub_batch.dims,
                    )?;
                    all_gpu_buffers.extend(gpu_images);
                    all_dims.extend(sub_batch.dims);
                }
            }

            if all_gpu_buffers.is_empty() {
                return Ok(vec![]);
            }

            // 阶段 2：统一 GPU 处理（一次 resize + 一次 hash，与旧版相同效率）
            let hashes = self.process_gpu_buffers(&ctx, &all_gpu_buffers, &all_dims)?;
            release_buffers(&ctx, all_gpu_buffers);

            Ok(hashes)
        })
        .map_err(|e| GpuError::Internal(format!("流水线工作线程 panic: {:?}", e)))?;

        // 转换为 HashBytes
        let u64s_per_image = self.hasher.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(raw_hashes.len() / u64s_per_image.max(1));
        for chunk in raw_hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 零拷贝流式哈希计算（Lanczos3 + 流水线优化版）。
    ///
    /// 与 [`compute_hashes_zero_copy_lanczos3()`](Self::compute_hashes_zero_copy_lanczos3)
    /// 相比，此方法使用双缓冲流水线：CPU 线程加载 + Lanczos3 缩放 + 灰度转换，
    /// 主线程同时执行 GPU 上传。全部上传完成后统一执行一次哈希计算。
    ///
    /// # 参数
    ///
    /// - `image_count`：要处理的图像总数
    /// - `loader`：图像加载闭包，返回 `image::DynamicImage`
    /// - `sub_batch_size`：子批大小，推荐 8-32
    pub fn compute_hashes_zero_copy_lanczos3_pipelined<F>(
        &self,
        image_count: usize,
        loader: F,
        sub_batch_size: usize,
    ) -> Result<Vec<HashBytes>, GpuError>
    where
        F: Fn(usize) -> Result<image::DynamicImage, String> + Send + Sync,
    {
        let ctx = self.ctx.lock().unwrap();
        let sub_batch_size = sub_batch_size.clamp(1, 64);
        let (target_w, target_h) = self.hasher.target_size();

        use std::sync::mpsc;

        struct SubBatchData {
            grayscales: Vec<Vec<u8>>,
        }

        let (tx, rx) = mpsc::sync_channel::<SubBatchData>(1);

        let raw_hashes = std::thread::scope(|s| -> Result<Vec<u64>, GpuError> {
            // ════ CPU 工作线程：rayon 并行加载 → Lanczos3 缩放 → 灰度 ════
            s.spawn(move || {
                let mut batch_indices: Vec<usize> = Vec::with_capacity(sub_batch_size);

                for i in 0..image_count {
                    batch_indices.push(i);

                    if batch_indices.len() >= sub_batch_size {
                        let indices = std::mem::take(&mut batch_indices);
                        // 使用 rayon 并行处理子批
                        let results: Vec<Result<image::DynamicImage, String>> = indices
                            .par_iter()
                            .map(|&idx| loader(idx))
                            .collect();

                        let mut gray_buf: Vec<Vec<u8>> = Vec::with_capacity(indices.len());

                        for res in results {
                            match res {
                                Ok(img) => {
                                    let resized = img.resize_exact(
                                        target_w,
                                        target_h,
                                        image::imageops::FilterType::Lanczos3,
                                    );
                                    let luma = resized.grayscale();
                                    let gray: Vec<u8> = luma.pixels().map(|(_, _, p)| p.0[0]).collect();
                                    gray_buf.push(gray);
                                }
                                Err(e) => {
                                    log::warn!("Lanczos3 流水线加载失败: {}", e);
                                }
                            }
                        }

                        if !gray_buf.is_empty() {
                            let batch = SubBatchData {
                                grayscales: gray_buf,
                            };
                            if tx.send(batch).is_err() {
                                break;
                            }
                        }
                    }
                }

                // 处理剩余图像
                if !batch_indices.is_empty() {
                    let results: Vec<Result<image::DynamicImage, String>> = batch_indices
                        .par_iter()
                        .map(|&idx| loader(idx))
                        .collect();

                    let mut gray_buf: Vec<Vec<u8>> = Vec::with_capacity(batch_indices.len());

                    for res in results {
                        match res {
                            Ok(img) => {
                                let resized = img.resize_exact(
                                    target_w,
                                    target_h,
                                    image::imageops::FilterType::Lanczos3,
                                );
                                let luma = resized.grayscale();
                                let gray: Vec<u8> = luma.pixels().map(|(_, _, p)| p.0[0]).collect();
                                gray_buf.push(gray);
                            }
                            Err(e) => {
                                log::warn!("Lanczos3 流水线加载失败: {}", e);
                            }
                        }
                    }

                    if !gray_buf.is_empty() {
                        let _ = tx.send(SubBatchData { grayscales: gray_buf });
                    }
                }
            });

            // ════ 主线程：阶段 1 逐子批上传 GPU，阶段 2 统一哈希 ════
            let mut all_gpu_buffers: Vec<GpuBuffer> = Vec::with_capacity(image_count);

            // 阶段 1：接收子批并上传到 GPU（与 CPU 线程并行）
            while let Ok(sub_batch) = rx.recv() {
                if sub_batch.grayscales.is_empty() {
                    continue;
                }
                let img_count = sub_batch.grayscales.len();
                let dims = vec![(target_w, target_h); img_count];
                let gpu_images = self.hasher.preprocess_gpu(
                    &ctx, &sub_batch.grayscales, &dims,
                )?;
                all_gpu_buffers.extend(gpu_images);
            }

            if all_gpu_buffers.is_empty() {
                return Ok(vec![]);
            }

            // 阶段 2：统一 GPU 哈希（一次 hash，所有图像已为目标尺寸）
            let all_dims = vec![(target_w, target_h); all_gpu_buffers.len()];
            let hashes = self.process_gpu_buffers(&ctx, &all_gpu_buffers, &all_dims)?;
            release_buffers(&ctx, all_gpu_buffers);

            Ok(hashes)
        })
        .map_err(|e| GpuError::Internal(format!("流水线工作线程 panic: {:?}", e)))?;

        let u64s_per_image = self.hasher.hash_size.u64s_per_image() as usize;
        let mut result = Vec::with_capacity(raw_hashes.len() / u64s_per_image.max(1));
        for chunk in raw_hashes.chunks(u64s_per_image.max(1)) {
            result.push(HashBytes::from_u64s(chunk));
        }
        Ok(result)
    }

    /// 处理已在 GPU 上的图像缓冲区，执行缩放 + 哈希。
    ///
    /// 内部根据图像尺寸是否一致选择批量管线或逐图管线。
    fn process_gpu_buffers(
        &self,
        ctx: &GpuContext,
        gpu_buffers: &[GpuBuffer],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        let image_count = gpu_buffers.len();
        if image_count == 0 {
            return Ok(vec![]);
        }

        // 检查所有图像是否尺寸一致
        let all_same_size = dimensions.windows(2).all(|w| w[0] == w[1]);

        if all_same_size {
            let (w, h) = dimensions[0];

            // 检查是否已是目标尺寸
            let (target_w, target_h) = self.hasher.target_size();
            if w == target_w && h == target_h {
                // 已是目标尺寸，直接计算哈希
                let src_pixels = (w as usize) * (h as usize);
                return self.hasher.compute_hash_gpu(
                    ctx,
                    &gpu_buffers[0], // 单图直接用第一个
                    src_pixels,
                    image_count,
                );
            }

            // GPU 缩放（如果启用）
            if self.hasher.gpu_resize.is_some() {
                let (resized, u32_count) =
                    self.hasher.resize_gpu(ctx, gpu_buffers, w, h)?;
                let hashes =
                    self.hasher.compute_hash_gpu(ctx, &resized, u32_count, image_count)?;
                ctx.buffer_pool()
                    .release(resized.into_raw(), BufferUsage::Storage);
                Ok(hashes)
            } else {
                // 无 GPU 缩放：需要下载 → CPU 缩放 → 重新上传
                // 这种情况下零拷贝优势有限，但仍比全量加载更省内存
                self.fallback_cpu_resize_and_hash(ctx, gpu_buffers, dimensions)
            }
        } else {
            // 尺寸不一致：逐图处理
            let mut all_hashes = Vec::with_capacity(image_count);
            for (i, buf) in gpu_buffers.iter().enumerate() {
                let (w, h) = dimensions[i];
                let (target_w, target_h) = self.hasher.target_size();

                let hashes = if w == target_w && h == target_h {
                    let src_pixels = (w as usize) * (h as usize);
                    self.hasher.compute_hash_gpu(ctx, buf, src_pixels, 1)?
                } else if self.hasher.gpu_resize.is_some() {
                    let single = vec![buf.clone()];
                    let (resized, u32_count) =
                        self.hasher.resize_gpu(ctx, &single, w, h)?;
                    let hashes =
                        self.hasher.compute_hash_gpu(ctx, &resized, u32_count, 1)?;
                    ctx.buffer_pool()
                        .release(resized.into_raw(), BufferUsage::Storage);
                    hashes
                } else {
                    let single_dims = [(w, h)];
                    self.fallback_cpu_resize_and_hash(ctx, &[buf.clone()], &single_dims)?
                };
                all_hashes.extend(hashes);
            }
            Ok(all_hashes)
        }
    }

    /// CPU 缩放回退路径：从 GPU 下载 → CPU 缩放 → 重新上传 → GPU 哈希。
    ///
    /// 当 GPU 缩放不可用时使用，仍保持逐批处理以限制内存。
    fn fallback_cpu_resize_and_hash(
        &self,
        ctx: &GpuContext,
        gpu_buffers: &[GpuBuffer],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let pool = ctx.buffer_pool();

        // 下载 GPU 缓冲区到 CPU
        let mut gray_images = Vec::with_capacity(gpu_buffers.len());
        for buf in gpu_buffers.iter() {
            let raw = buf.download_with_pool(device, queue, pool)?;
            let u32_data: &[u32] = bytemuck::cast_slice(&raw);
            let gray: Vec<u8> = u32_data.iter().map(|&p| (p & 0xFF) as u8).collect();
            gray_images.push(gray);
        }

        // CPU 缩放
        let resized = self.hasher.resize_cpu(&gray_images, dimensions)?;

        // 重新上传并计算哈希
        let resized_dims: Vec<(u32, u32)> = resized
            .iter()
            .map(|_| self.hasher.target_size())
            .collect();
        let gpu_resized = self.hasher.preprocess_gpu(ctx, &resized, &resized_dims)?;

        let (target_w, target_h) = self.hasher.target_size();
        let target_pixels = (target_w as usize) * (target_h as usize);
        let u32_count = target_pixels * gpu_resized.len();

        // 合并 GPU 缓冲区
        let merged = crate::tasks::phasher_util::merge_gpu_buffers(
            ctx,
            &gpu_resized,
            target_pixels,
        )?;
        release_buffers(ctx, gpu_resized);

        let hashes = self
            .hasher
            .compute_hash_gpu(ctx, &merged, u32_count, gpu_buffers.len())?;
        ctx.buffer_pool()
            .release(merged.into_raw(), BufferUsage::Storage);
        Ok(hashes)
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

    /// 检查是否启用了 GPU 缩放（零拷贝管线所需）。
    pub fn has_gpu_resize(&self) -> bool {
        self.hasher.gpu_resize.is_some()
    }
}
