//! # GPGPU-tool
//!
//! 基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPGPU 加速算法库。
//!
//! 优先使用 GPU 并行加速，当 GPU 不可用时自动降级到 CPU 实现。开箱即用，无需手写 WGSL 着色器。
//!
//! ## ├── 能力层（Core）
//!
//! 封装 wgpu 底层细节，提供 GPU 资源管理基础设施：
//!
//! | 类型 | 用途 |
//! |------|------|
//! | [`GpuContext`] | GPU 上下文（设备/队列/管线缓存），所有 GPU 操作入口 |
//! | [`GpuBuffer`] | CPU-GPU 数据传输缓冲区 |
//! | [`BufferPool`] | 按尺寸分档的缓冲区复用池 |
//! | [`BufferPoolConfig`] | 缓冲区池配置（分档基数、容量、释放阈值） |
//! | [`ComputePipeline`] | 计算管线创建与 dispatch |
//! | [`GpuBatchSubmitter`] | 异步批量提交，合并多次 dispatch 为一次 GPU 提交 |
//! | [`GpuError`] | 统一错误类型 |
//!
//! ## ├── 业务层（Tasks）
//!
//! 基于能力层实现的 GPU 加速算法：
//!
//! | 模块 | 功能 | 入口 |
//! |------|------|------|
//! | [`tasks::sha256`] | SHA-256 并行哈希 | [`Sha256Computer`](tasks::sha256::Sha256Computer) |
//! | [`tasks::phasher`] | 感知哈希（Mean/Gradient/Block 等 6 种算法） | [`PerceptualHasher`](tasks::phasher::PerceptualHasher) |
//! | [`tasks::convolution`] | GPU 2D 卷积（Full2D / Separable） | [`GpuConvolution`] |
//! | [`tasks::gaussian_blur`] | GPU 高斯模糊（基于可分离卷积） | [`GpuGaussianBlur`] |
//! | [`tasks::dihedral`] | 二面体变换（D4 群 8 种旋转/翻转） | [`DihedralHashes64`] |
//! | [`tasks::matcher`] | 哈希匹配策略（线性/BK-tree/链式/门面） | [`HashMatcherFacade`] |
//! | [`tasks::bktree`] | BK-tree 近似最近邻搜索（基于 Hamming 距离） | [`BkTree`](tasks::bktree::BkTree) |
//! | `tasks::pdq_hash`* | PDQ 感知哈希（需 `pdq` feature） | `PdqHashGpu` |
//!
//! *\* `pdq_hash` 模块需启用 `pdq` feature 方可访问。*
//!
//! ## └── 快速开始
//!
//! ### 同步 SHA-256 哈希
//!
//! ```no_run
//! use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//! let sha256 = Sha256Computer::new(&mut ctx).unwrap();
//!
//! let messages = vec![b"hello".to_vec(), b"world".to_vec()];
//! let hashes = sha256.compute(&ctx, &messages).unwrap();
//!
//! for (i, hash) in hashes.iter().enumerate() {
//!     println!("消息 {} 的 SHA-256: {:02x?}", i, hash);
//! }
//! ```
//!
//! ### 感知哈希（图像相似度检测）
//!
//! ```no_run
//! use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//! let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
//!
//! // 任意尺寸的灰度图像，内部自动缩放
//! let images = vec![vec![128u8; 256 * 256]];
//! let dimensions = vec![(256u32, 256u32)];
//! let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
//! ```
//!
//! ### 自定义哈希位数
//!
//! ```no_run
//! use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}, HashSize};
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//!
//! // 16x16 网格 → 256 bit 哈希
//! let hasher = PerceptualHasher::with_hash_size(
//!     &mut ctx, HashAlgorithm::Mean, HashSize::new(16),
//! ).unwrap();
//!
//! // 32x32 网格 → 1024 bit 哈希
//! let hasher = PerceptualHasher::with_hash_size(
//!     &mut ctx, HashAlgorithm::Block, HashSize::new(32),
//! ).unwrap();
//! ```
//!
//! ### BK-tree 近似图像检索
//!
//! ```no_run
//! use gpgpu_tool::tasks::bktree::{BkTree, hamming_distance};
//!
//! let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321];
//! let tree = BkTree::from_hashes(hashes.iter().copied());
//!
//! // 查找 Hamming 距离 <= 5 的近似图像
//! let similar = tree.find(0xA1B2C3D4, 5);
//! // 查找最近邻
//! let nearest = tree.find_nearest(0xA1B2C3D4);
//! ```
//!
//! ### GPU 加速缩放 + 哈希（零拷贝流水线）
//!
//! ```no_run
//! use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};
//!
//! let mut ctx = GpuContext::new_sync().unwrap();
//!
//! // use_gpu_resize = true 启用 GPU 缩放，适合大批量大图
//! let hasher = PerceptualHasher::with_resize_mode(
//!     &mut ctx, HashAlgorithm::Mean, true,
//! ).unwrap();
//!
//! let images = vec![vec![128u8; 1024 * 1024]];
//! let dimensions = vec![(1024u32, 1024u32)];
//! let hashes = hasher.compute(&ctx, &images, &dimensions).unwrap();
//! ```
//!
//! ## ── 特性开关
//!
//! | Feature | 默认 | 说明 |
//! |---------|------|------|
//! | `image` | 否 | 启用 `image` crate 集成，支持直接从 `DynamicImage` 计算哈希 |
//! | `cpu-fallback` | 是 | 启用 CPU 降级实现（`Sha256Cpu`、`PHasherCpu`） |
//! | `pdq` | 否 | 启用 PDQ 感知哈希算法 |
//!
//! ## ── 相关链接
//!
//! - [GitHub 仓库](https://github.com/solo-king/gpgpu-tool)
//! - [crates.io](https://crates.io/crates/gpgpu-tool)
//! - [文档](https://docs.rs/gpgpu-tool)

// ── 能力层模块 ──────────────────────────────────────────────────
mod batch;
mod buffer;
mod buffer_pool;
mod context;
mod error;
mod pipeline;
pub mod pixel_pack;

// ── 业务层模块 ──────────────────────────────────────────────────
pub mod tasks;

// ── 能力层公共 API ──────────────────────────────────────────────

/// 计算后端类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Gpu,
    Cpu,
}

/// GPU 上下文，封装 wgpu 的 Instance、Adapter、Device、Queue 及内部管线缓存。
///
/// 所有 GPU 操作都通过此上下文进行。详见 [`context::GpuContext`]。
pub use context::GpuContext;

/// GPU 缓冲区，封装 CPU-GPU 数据传输。
///
/// 支持从任意 `bytemuck::Pod` 类型创建、写入和读取数据。
pub use buffer::GpuBuffer;

/// 缓冲区使用类型标识（Storage / Uniform）。
pub use buffer::BufferUsage;

/// 按尺寸分档的 GPU 缓冲区复用池，减少重复分配开销。
pub use buffer_pool::BufferPool;

/// 缓冲区复用池配置参数。
pub use buffer_pool::BufferPoolConfig;

/// 计算管线，封装 WGSL 着色器的编译、绑定组创建和 dispatch 调度。
pub use pipeline::ComputePipeline;

/// 绑定类型，定义管线中单个 binding 的缓冲区访问模式。
pub use pipeline::BindingType;

/// 管线描述符，定义管线的绑定布局、WGSL 源码和 workgroup 尺寸。
pub use pipeline::PipelineDescriptor;

/// 异步批量提交器，将多次 GPU dispatch 合并为一次 submit。
pub use batch::GpuBatchSubmitter;

/// 批量计算任务描述，描述一次 GPU dispatch 所需的全部资源。
pub use batch::BatchJob;

/// 统一的 GPU 计算错误类型。
pub use error::GpuError;

/// 卷积边界模式（Zero / Clamp / Reflect）。
pub use tasks::convolution::BorderMode;

/// 卷积计算模式（Full2D / Separable）。
pub use tasks::convolution::ConvMode;

/// GPU 2D 卷积，支持 Full2D 和 Separable 两种模式。
pub use tasks::convolution::GpuConvolution;

/// GPU 高斯模糊，基于可分离卷积实现。
pub use tasks::gaussian_blur::GpuGaussianBlur;

/// 感知哈希尺寸，支持任意网格大小（8/16/32/64）。
pub use tasks::hash_common::HashSize;

/// 256-bit 二面体变换结果，包含 D4 群 8 种旋转/翻转哈希。
pub use tasks::dihedral::DihedralHashes256;

/// 64-bit 二面体变换结果，包含 D4 群 8 种旋转/翻转哈希。
pub use tasks::dihedral::DihedralHashes64;

/// 基于 BK-tree 的哈希匹配器。
pub use tasks::matcher::BkTreeMatcher;

/// 链式匹配策略，先粗筛再精筛。
pub use tasks::matcher::ChainStrategy;

/// 链式匹配器，组合多种策略依次执行。
pub use tasks::matcher::ChainedMatcher;

/// 哈希匹配 trait，定义匹配接口。
pub use tasks::matcher::HashMatcher;

/// 哈希匹配门面，自动选择最优匹配策略。
pub use tasks::matcher::HashMatcherFacade;

/// 线性扫描匹配器，逐个比较所有哈希。
pub use tasks::matcher::LinearScanMatcher;

/// 匹配结果，包含匹配的哈希值和距离。
pub use tasks::matcher::MatchResult;

/// CPU 降级 SHA-256 实现（需 `cpu-fallback` feature）。
#[cfg(feature = "cpu-fallback")]
pub use tasks::sha256_cpu::Sha256Cpu;

/// CPU 降级感知哈希实现（需 `cpu-fallback` feature）。
#[cfg(feature = "cpu-fallback")]
pub use tasks::phasher_cpu::PHasherCpu;

/// GPU PDQ 感知哈希（需 `pdq` feature）。
#[cfg(feature = "pdq")]
pub use tasks::pdq_hash::PdqHashGpu;

// ── 内部辅助重导出（doc(hidden) 但不破坏外部测试兼容性）───────
// 以下内容对库使用者不可见，但外部集成测试和基准测试仍可访问。
pub use tasks::bktree::hamming_distance as __hamming_distance;