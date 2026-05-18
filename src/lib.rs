//! # wgpu-compute-engine
//!
//! 基于 wgpu 的 GPU 通用计算引擎，为 CPU 密集型任务提供 GPU 并行加速能力。
//!
//! ## 架构
//!
//! - **能力层**：封装 wgpu 底层，提供 `GpuContext`（设备/队列/管线缓存）、
//!   `GpuBuffer`（缓冲区管理）、`ComputePipeline`（计算管线）、`Chunker`（通用分批）
//! - **业务层**：基于能力层实现具体算法，如 `tasks::sha256::Sha256Computer`
//!
//! ## 示例
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
//!     println!("消息 {} 的哈希: {:02x?}", i, hash);
//! }
//! ```

mod batch;
mod buffer;
mod buffer_pool;
mod chunker;
mod context;
mod error;
mod pipeline;

pub mod tasks;

pub use batch::{BatchJob, GpuBatchSubmitter};
pub use buffer::{BufferUsage, GpuBuffer};
pub use buffer_pool::BufferPool;
pub use chunker::{BatchInfo, Chunker};
pub use context::GpuContext;
pub use error::GpuError;
pub use pipeline::ComputePipeline;
