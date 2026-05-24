use std::cell::RefCell;
use std::collections::HashMap;

use wgpu::{Buffer, BufferUsages, Device};

use crate::buffer::BufferUsage;

/// 缓冲区复用池的配置参数。
///
/// 控制分档基数、每档最大缓存数、释放阈值等行为。
/// 默认配置适用于大多数场景，可根据显存大小和并发量调整。
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// 分档起始大小（字节），所有 ≤ 此值的缓冲区归入第一档。
    /// 后续按 2 的幂次逐档递增。默认 256。
    pub bin_base: u64,
    /// 每个尺寸档位最多缓存的缓冲区数。默认 8。
    pub max_per_class: usize,
    /// 归还时直接销毁的字节阈值，超过此值的缓冲区不回收。
    /// 默认 1MB (1024 * 1024)。
    pub release_threshold: u64,
    /// 分档上限（字节），超过此值的缓冲区不做幂次对齐，按实际大小精确分档。
    /// 默认 1GB (1024 * 1024 * 1024)。
    pub max_bin: u64,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            bin_base: 256,
            max_per_class: 8,
            release_threshold: 1024 * 1024,
            max_bin: 1024 * 1024 * 1024,
        }
    }
}

impl BufferPoolConfig {
    /// 创建默认配置。
    pub fn new() -> Self { Self::default() }

    /// 设置分档起始大小。
    pub fn bin_base(mut self, v: u64) -> Self { self.bin_base = v; self }
    /// 设置每档最大缓存数。
    pub fn max_per_class(mut self, v: usize) -> Self { self.max_per_class = v; self }
    /// 设置归还时的释放阈值（超过此值不回收）。
    pub fn release_threshold(mut self, v: u64) -> Self { self.release_threshold = v; self }
    /// 设置分档上限。
    pub fn max_bin(mut self, v: u64) -> Self { self.max_bin = v; self }
}

#[derive(Hash, Eq, PartialEq)]
struct PoolKey {
    size_class: u64,
    usage: BufferUsage,
}

impl BufferPool {
    fn size_class(&self, size: u64) -> u64 {
        let base = self.config.bin_base;
        if size <= base {
            return base;
        }
        let mut class = base;
        while class < size {
            class = class.saturating_mul(2);
            if class > self.config.max_bin {
                return size;
            }
        }
        class
    }
}

fn to_wgpu_usage(usage: BufferUsage) -> BufferUsages {
    match usage {
        BufferUsage::Storage => {
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
        }
        BufferUsage::Uniform => {
            BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
        }
    }
}

/// 按尺寸分档的 GPU 缓冲区复用池，减少重复分配与销毁开销。
///
/// # 设计
///
/// 缓冲区按 2 的幂次尺寸分档（默认 256B → 512B → 1KB → 2KB → …），
/// 每档最多缓存若干个缓冲区。归还时超过释放阈值的缓冲区直接销毁，
/// 小于阈值的回收复用。详见 [`BufferPoolConfig`]。
///
/// # 使用方式
///
/// ```no_run
/// use wgpu_compute_engine::{GpuContext, BufferPool, BufferUsage, BufferPoolConfig};
///
/// let ctx = GpuContext::new_sync().unwrap();
///
/// // 默认配置
/// let pool = BufferPool::new();
///
/// // 自定义配置：每档最多 16 个，释放阈值 2MB
/// let pool = BufferPool::with_config(BufferPoolConfig::default()
///     .max_per_class(16)
///     .release_threshold(2 * 1024 * 1024));
///
/// // 从池中获取或创建
/// let buf = pool.acquire(ctx.device(), 65536, BufferUsage::Storage);
/// // ... 使用后归还
/// pool.release(buf, BufferUsage::Storage);
/// ```
pub struct BufferPool {
    pools: RefCell<HashMap<PoolKey, Vec<Buffer>>>,
    staging_pools: RefCell<HashMap<u64, Vec<Buffer>>>,
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
            pools: RefCell::new(HashMap::new()),
            staging_pools: RefCell::new(HashMap::new()),
            config,
        }
    }

    /// 返回当前配置的只读引用。
    pub fn config(&self) -> &BufferPoolConfig { &self.config }

    /// 从池中获取一个合适大小的缓冲区，若池空则创建新缓冲区。
    pub fn acquire(&self, device: &Device, size: u64, usage: BufferUsage) -> Buffer {
        let class = self.size_class(size);
        let key = PoolKey { size_class: class, usage };

        let mut pools = self.pools.borrow_mut();
        if let Some(buffers) = pools.get_mut(&key) {
            if let Some(buf) = buffers.pop() {
                log::debug!("缓冲区池命中: class={}, usage={:?}", class, usage);
                return buf;
            }
        }

        log::debug!("缓冲区池未命中，创建新缓冲区: class={}, usage={:?}", class, usage);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size: class,
            usage: to_wgpu_usage(usage),
            mapped_at_creation: false,
        })
    }

    /// 将缓冲区归还到池中，超过释放阈值的直接销毁。
    ///
    /// `usage` 必须与缓冲区创建时一致，否则后续 `acquire` 可能返回错误类型的缓冲区。
    pub fn release(&self, buffer: Buffer, usage: BufferUsage) {
        let size = buffer.size();
        if size > self.config.release_threshold {
            log::debug!("缓冲区过大 (>{})，直接销毁: size={}", self.config.release_threshold, size);
            return;
        }

        let key = PoolKey {
            size_class: self.size_class(size),
            usage,
        };

        let mut pools = self.pools.borrow_mut();
        let entry = pools.entry(key).or_default();
        if entry.len() < self.config.max_per_class {
            entry.push(buffer);
        }
    }

    #[doc(hidden)]
    pub fn acquire_staging(&self, device: &Device, size: u64) -> Buffer {
        let mut pools = self.staging_pools.borrow_mut();
        if let Some(buffers) = pools.get_mut(&size) {
            if let Some(buf) = buffers.pop() {
                return buf;
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_pooled"),
            size,
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
        let mut pools = self.staging_pools.borrow_mut();
        let entry = pools.entry(size).or_default();
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
        let pool = BufferPool::new();
        assert_eq!(pool.size_class(1), 256);
        assert_eq!(pool.size_class(256), 256);
        assert_eq!(pool.size_class(257), 512);
        assert_eq!(pool.size_class(1024), 1024);
        assert_eq!(pool.size_class(1025), 2048);
        assert_eq!(pool.size_class(1024 * 1024), 1024 * 1024);
    }

    #[test]
    fn test_size_class_custom_base() {
        let pool = BufferPool::with_config(BufferPoolConfig::default().bin_base(64));
        assert_eq!(pool.size_class(1), 64);
        assert_eq!(pool.size_class(64), 64);
        assert_eq!(pool.size_class(65), 128);
    }

    #[test]
    fn test_config_builder() {
        let cfg = BufferPoolConfig::default()
            .bin_base(512)
            .max_per_class(16)
            .release_threshold(2 * 1024 * 1024);
        assert_eq!(cfg.bin_base, 512);
        assert_eq!(cfg.max_per_class, 16);
        assert_eq!(cfg.release_threshold, 2 * 1024 * 1024);
    }
}