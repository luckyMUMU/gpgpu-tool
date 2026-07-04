use sha2::{Digest, Sha256};

use crate::error::GpuError;

/// SHA-256 CPU 降级实现。
///
/// 当 GPU 不可用时，使用 `sha2` crate 在 CPU 上计算 SHA-256 哈希。
/// API 与 [`Sha256Computer`](super::sha256::Sha256Computer) 保持一致，
/// 调用方无需感知后端切换。
pub struct Sha256Cpu;

impl Sha256Cpu {
    pub fn new() -> Self {
        Self
    }

    /// 计算多条消息的 SHA-256 哈希。
    pub fn compute(&self, messages: &[Vec<u8>]) -> Result<Vec<[u8; 32]>, GpuError> {
        #[cfg(feature = "parallel-cpu")]
        {
            use rayon::prelude::*;
            messages
                .par_iter()
                .map(|msg| {
                    let mut hasher = Sha256::new();
                    hasher.update(msg);
                    let result = hasher.finalize();
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&result);
                    Ok(hash)
                })
                .collect()
        }
        #[cfg(not(feature = "parallel-cpu"))]
        {
            messages
                .iter()
                .map(|msg| {
                    let mut hasher = Sha256::new();
                    hasher.update(msg);
                    let result = hasher.finalize();
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&result);
                    Ok(hash)
                })
                .collect()
        }
    }
}

impl Default for Sha256Cpu {
    fn default() -> Self {
        Self::new()
    }
}
