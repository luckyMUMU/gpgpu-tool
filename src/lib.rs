//! # wgpu-compute-engine
//!
//! 基于 [wgpu](https://github.com/gfx-rs/wgpu) 的跨平台 GPU 并行加速引擎。
//!
//! 为 CPU 密集型计算任务提供 GPU 并行加速能力，开箱即用，无需手写 WGSL 着色器。
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
//! | [`tasks::bktree`] | BK-tree 近似最近邻搜索（基于 Hamming 距离） | [`BkTree`](tasks::bktree::BkTree) |
//!
//! ## └── 快速开始
//!
//! ### 同步 SHA-256 哈希
//!
//! ```no_run
//! use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};
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
//! use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};
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
//! use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}, HashSize};
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
//! use wgpu_compute_engine::tasks::bktree::{BkTree, hamming_distance};
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
//! use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};
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
//! | Feature | 说明 |
//! |---------|------|
//! | `image` | 启用 `image` crate 集成，支持直接从 `DynamicImage` 计算哈希 |
//!
//! ## ── 相关链接
//!
//! - [GitHub 仓库](https://github.com/solo-king/wgpu-tool)
//! - [crates.io](https://crates.io/crates/wgpu-compute-engine)
//! - [文档](https://docs.rs/wgpu-compute-engine)

// ── 能力层模块 ──────────────────────────────────────────────────
mod batch;
mod buffer;
mod buffer_pool;
mod context;
mod error;
mod pipeline;

// ── 业务层模块 ──────────────────────────────────────────────────
pub mod tasks;

// ── 能力层公共 API ──────────────────────────────────────────────

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

/// 异步批量提交器，将多次 GPU dispatch 合并为一次 submit。
pub use batch::GpuBatchSubmitter;

/// 批量计算任务描述，描述一次 GPU dispatch 所需的全部资源。
pub use batch::BatchJob;

/// 统一的 GPU 计算错误类型。
pub use error::GpuError;

pub use tasks::hash_common::HashSize;

// ── 内部辅助重导出（doc(hidden) 但不破坏外部测试兼容性）───────
// 以下内容对库使用者不可见，但外部集成测试和基准测试仍可访问。
pub use tasks::bktree::hamming_distance as __hamming_distance;