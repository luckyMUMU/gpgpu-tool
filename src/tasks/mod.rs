//! # GPU 加速计算任务
//!
//! 提供基于 wgpu 的 GPU 并行加速业务算法。

pub mod bktree;
pub mod dihedral;
pub mod matcher;
pub mod phasher;
pub mod sha256;

#[cfg(feature = "cpu-fallback")]
pub mod sha256_cpu;

#[cfg(feature = "cpu-fallback")]
pub mod phasher_cpu;

// ── 以下为内部实现模块，对外隐藏 ──────────────────────────────
// 使用者应通过 phasher::PerceptualHasher 使用感知哈希能力，
// 不直接依赖具体哈希算法实现或内部辅助模块。

#[doc(hidden)]
pub mod block_hash;

#[doc(hidden)]
pub mod double_gradient_hash;

#[doc(hidden)]
pub mod convolution;

#[doc(hidden)]
pub mod gaussian_blur;

#[doc(hidden)]
pub mod gpu_resize;

#[doc(hidden)]
pub mod gradient_hash;

#[doc(hidden)]
pub mod hash_common;

#[doc(hidden)]
pub mod mean_hash;

#[doc(hidden)]
pub mod median_hash;

#[doc(hidden)]
pub mod vert_gradient_hash;

#[cfg(feature = "pdq")]
pub mod pdq_hash;