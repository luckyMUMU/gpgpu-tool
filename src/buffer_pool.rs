use std::collections::HashMap;
use std::sync::Mutex;

use wgpu::{Buffer, BufferUsages, Device};

use crate::buffer::BufferUsage;
use crate::error::GpuError;

/// 缓冲区复用池的配置参数。
///
/// 控制每档最大缓存数、释放阈值等行为。
/// 默认配置适用于大多数场景，可根据显存大小和并发量调整。
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// 每个尺寸档位最多缓存的缓冲区数。默认 8。
    pub max_per_class: usize,
    /// 归还时直接销毁的字节阈值，超过此值的缓冲区不回收。
    /// 默认 4MB (4 * 1024 * 1024)。
    pub release_threshold: u64,
    /// 是否缓存大缓冲区（超过 release_threshold 的缓冲区）。
    /// 启用后，大缓冲区也会进入缓存池，减少重复分配开销。
    /// 默认 `false`。适用于需要频繁处理大图像的场景。
    pub large_buffer_cache: bool,
    /// 大缓冲区每档最大缓存数。默认 2。
    pub large_buffer_max_per_class: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_per_class: 8,
            release_threshold: 4 * 1024 * 1024,
            large_buffer_cache: false,
            large_buffer_max_per_class: 2,
        }
    }
}

impl BufferPoolConfig {
    /// 创建默认配置。
    pub fn new() -> Self { Self::default() }

    /// 设置每档最大缓存数。
    pub fn max_per_class(mut self, v: usize) -> Self { self.max_per_class = v; self }
    /// 设置归还时的释放阈值（超过此值不回收）。
    pub fn release_threshold(mut self, v: u64) -> Self { self.release_threshold = v; self }
    /// 设置是否缓存大缓冲区（超过 release_threshold 的缓冲区）。
    /// 启用后，大缓冲区也会进入缓存池，减少重复分配开销。
    /// 默认 `false`。适用于需要频繁处理大图像的场景。
    pub fn large_buffer_cache(mut self, v: bool) -> Self { self.large_buffer_cache = v; self }
    /// 设置大缓冲区每档最大缓存数。默认 2。
    pub fn large_buffer_max_per_class(mut self, v: usize) -> Self { self.large_buffer_max_per_class = v; self }
}

/// 256 字节对齐的固定尺寸档位表。
///
/// 所有档位均为 256 的倍数，符合 GPU Storage Buffer 最佳对齐实践。
/// 超过最大档位（4MB）的请求按 256 字节对齐后精确分档。
const SIZE_CLASSES: [u64; 15] = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
];

/// 将大小向上取整到 256 字节对齐。
const fn align256(size: u64) -> u64 {
    (size + 255) & !255
}

#[derive(Hash, Eq, PartialEq)]
struct PoolKey {
    size_class: u64,
    usage: BufferUsage,
}

/// 将大小映射到最近的 size class 档位。
///
/// 先 256 字节对齐，再匹配到 SIZE_CLASSES 中最近的档位。
/// 超过最大档位时返回对齐后的精确值。
fn size_class(size: u64) -> u64 {
    let aligned = align256(size);
    for &class in &SIZE_CLASSES {
        if class >= aligned {
            return class;
        }
    }
    aligned
}

/// 按尺寸分档的 GPU 缓冲区复用池，减少重复分配与销毁开销。
///
/// # 设计
///
/// 缓冲区按 256 字节对齐的固定尺寸档位表分档（256B → 512B → 1KB → … → 4MB），
/// 所有档位均为 256 的倍数，符合 GPU Storage Buffer 最佳对齐实践。
/// 超过 4MB 的请求按 256 字节对齐后精确分档。
/// 每档最多缓存若干个缓冲区。归还时超过释放阈值的缓冲区直接销毁，
/// 小于阈值的回收复用。详见 [`BufferPoolConfig`]。
///
/// # 线程安全
///
/// `BufferPool` 内部使用 `Mutex` 保护内部状态。**不可重入**：
/// 持有锁时（如在 `acquire`/`release` 回调中）不要再次调用 `acquire`/`release`，
/// 否则会导致死锁。
///
/// # 使用方式
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, BufferPool, BufferUsage, BufferPoolConfig};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ctx = GpuContext::new_sync().unwrap();
///
///     // 默认配置
///     let pool = BufferPool::new();
///
///     // 自定义配置：每档最多 16 个，释放阈值 2MB
///     let pool = BufferPool::with_config(BufferPoolConfig::default()
///         .max_per_class(16)
///         .release_threshold(2 * 1024 * 1024));
///
///     // 从池中获取或创建
///     let buf = pool.acquire(ctx.device()?, 65536, BufferUsage::Storage);
///     // ... 使用后归还
///     pool.release(buf, BufferUsage::Storage);
///     Ok(())
/// }
/// ```
pub struct BufferPool {
    pools: Mutex<HashMap<PoolKey, Vec<Buffer>>>,
    staging_pools: Mutex<HashMap<u64, Vec<Buffer>>>,
    config: BufferPoolConfig,
}

impl BufferPool {
    /// 创建新的缓冲区池（默认配置）。
    pub fn new() -> Self {
        Self::with_config(BufferPoolConfig::default())
    }

    /// 创建指定配置的缓冲区池。
    pub fn with_config(config: BufferPoolConfig) -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            staging_pools: Mutex::new(HashMap::new()),
            config,
        }
    }

    /// 返回当前配置的只读引用。
    pub fn config(&self) -> &BufferPoolConfig { &self.config }

    /// 从池中获取一个合适大小的缓冲区，若池空则创建新缓冲区。
    ///
    /// 请求大小会先向上取整到 256 字节对齐，再匹配到最近的档位。
    ///
    /// # 错误
    /// 当 `size` 超过 `device.limits().max_storage_buffer_binding_size` 时，
    /// 返回 [`GpuError::Oom`] 以避免后续 wgpu 创建缓冲区时触发驱动级 OOM。
    /// 调用方应通过 `?` 传播错误，由上层决定是否降级到 CPU。
    pub fn acquire(
        &self,
        device: &Device,
        size: u64,
        usage: BufferUsage,
    ) -> Result<Buffer, GpuError> {
        // P0-层2：预检查 max_storage_buffer_binding_size，避免驱动级 OOM 崩溃
        let max_binding = device.limits().max_storage_buffer_binding_size as u64;
        if size > max_binding {
            return Err(GpuError::Oom {
                requested: size,
                limit: max_binding,
            });
        }

        let aligned = align256(size);
        let class = size_class(aligned);
        let key = PoolKey { size_class: class, usage };

        let mut pools = self.pools.lock().unwrap();
        if let Some(buffers) = pools.get_mut(&key) {
            if let Some(buf) = buffers.pop() {
                log::debug!("缓冲区池命中: class={}, usage={:?}", class, usage);
                return Ok(buf);
            }
        }

        log::debug!("缓冲区池未命中，创建新缓冲区: class={}, usage={:?}", class, usage);
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: class,
            usage: usage.to_wgpu_usage(true),
            mapped_at_creation: false,
        }))
    }

    /// 将缓冲区归还到池中，超过释放阈值的直接销毁。
    ///
    /// `usage` 必须与缓冲区创建时一致，否则后续 `acquire` 可能返回错误类型的缓冲区。
    ///
    /// 当 [`BufferPoolConfig::large_buffer_cache`] 启用时，大缓冲区也会被缓存，
    /// 每档最多缓存 `large_buffer_max_per_class` 个。
    pub fn release(&self, buffer: Buffer, usage: BufferUsage) {
        let size = buffer.size();
        if size > self.config.release_threshold {
            if self.config.large_buffer_cache {
                let key = PoolKey {
                    size_class: size_class(size),
                    usage,
                };
                let mut pools = self.pools.lock().unwrap();
                let entry = pools.entry(key).or_default();
                if entry.len() < self.config.large_buffer_max_per_class {
                    log::debug!("大缓冲区缓存: size={}", size);
                    entry.push(buffer);
                    return;
                }
            }
            log::debug!("缓冲区过大 (>{})，直接销毁: size={}", self.config.release_threshold, size);
            return;
        }

        let key = PoolKey {
            size_class: size_class(size),
            usage,
        };

        let mut pools = self.pools.lock().unwrap();
        let entry = pools.entry(key).or_default();
        if entry.len() < self.config.max_per_class {
            entry.push(buffer);
        }
    }

    #[doc(hidden)]
    pub fn acquire_staging(&self, device: &Device, size: u64) -> Buffer {
        let class = size_class(size);
        let mut pools = self.staging_pools.lock().unwrap();
        if let Some(buffers) = pools.get_mut(&class) {
            if let Some(buf) = buffers.pop() {
                log::debug!("staging 缓冲区池命中: class={}", class);
                return buf;
            }
        }

        log::debug!("staging 缓冲区池未命中，创建新缓冲区: class={}", class);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_pooled"),
            size: class,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[doc(hidden)]
    pub fn release_staging(&self, buffer: Buffer) {
        let size = buffer.size();
        if size > self.config.release_threshold {
            return;
        }
        let class = size_class(size);
        let mut pools = self.staging_pools.lock().unwrap();
        let entry = pools.entry(class).or_default();
        if entry.len() < self.config.max_per_class {
            entry.push(buffer);
        }
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class() {
        assert_eq!(size_class(1), 256);
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 512);
        assert_eq!(size_class(1024), 1024);
        assert_eq!(size_class(1025), 2048);
        assert_eq!(size_class(1024 * 1024), 1024 * 1024);
        // 超过 1MB 进入 2MB 档位（不再精确分档）
        assert_eq!(size_class(1024 * 1024 + 1), 2 * 1024 * 1024);
        assert_eq!(size_class(2 * 1024 * 1024), 2 * 1024 * 1024);
        assert_eq!(size_class(2 * 1024 * 1024 + 1), 4 * 1024 * 1024);
        assert_eq!(size_class(4 * 1024 * 1024), 4 * 1024 * 1024);
        // 超过最大档位 4MB 时按 256 字节对齐后精确分档
        assert_eq!(size_class(4 * 1024 * 1024 + 1), 4 * 1024 * 1024 + 256);
    }

    #[test]
    fn test_align256() {
        assert_eq!(align256(0), 0);
        assert_eq!(align256(1), 256);
        assert_eq!(align256(255), 256);
        assert_eq!(align256(256), 256);
        assert_eq!(align256(257), 512);
        assert_eq!(align256(512), 512);
        assert_eq!(align256(513), 768);
    }

    #[test]
    fn test_config_builder() {
        let cfg = BufferPoolConfig::default()
            .max_per_class(16)
            .release_threshold(2 * 1024 * 1024);
        assert_eq!(cfg.max_per_class, 16);
        assert_eq!(cfg.release_threshold, 2 * 1024 * 1024);
    }

    #[test]
    fn test_default_release_threshold() {
        let cfg = BufferPoolConfig::default();
        assert_eq!(cfg.release_threshold, 4 * 1024 * 1024);
    }
}