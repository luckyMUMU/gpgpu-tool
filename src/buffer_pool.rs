use std::cell::RefCell;
use std::collections::HashMap;

use wgpu::{Buffer, BufferUsages, Device};

use crate::buffer::BufferUsage;

#[derive(Hash, Eq, PartialEq)]
struct PoolKey {
    size_class: u64,
    usage: BufferUsage,
}

fn size_class(size: u64) -> u64 {
    if size <= 256 {
        return 256;
    }
    let mut class = 256u64;
    while class < size {
        class = class.saturating_mul(2);
        if class > 1024 * 1024 * 1024 {
            return size;
        }
    }
    class
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
/// 缓冲区按 2 的幂次尺寸分档（256B, 512B, 1KB, 2KB, …），
/// 每个档位最多缓存 8 个缓冲区。释放时小于 1MB 的缓冲区回收复用，
/// 超过此阈值的直接销毁以避免占用显存。
///
/// # 使用方式
///
/// ```no_run
/// use wgpu_compute_engine::{GpuContext, BufferPool, BufferUsage};
///
/// let ctx = GpuContext::new_sync().unwrap();
/// let pool = BufferPool::new();
///
/// // 从池中获取或创建
/// let buf = pool.acquire(ctx.device(), 65536, BufferUsage::Storage);
/// // ... 使用后归还
/// pool.release(buf);
/// ```
pub struct BufferPool {
    pools: RefCell<HashMap<PoolKey, Vec<Buffer>>>,
    staging_pools: RefCell<HashMap<u64, Vec<Buffer>>>,
    max_per_class: usize,
}

impl BufferPool {
    /// 创建新的缓冲区池。
    pub fn new() -> Self {
        Self {
            pools: RefCell::new(HashMap::new()),
            staging_pools: RefCell::new(HashMap::new()),
            max_per_class: 8,
        }
    }

    /// 从池中获取一个合适大小的缓冲区，若池空则创建新缓冲区。
    pub fn acquire(&self, device: &Device, size: u64, usage: BufferUsage) -> Buffer {
        let class = size_class(size);
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

    /// 将缓冲区归还到池中，大于 1MB 的直接销毁。
    pub fn release(&self, buffer: Buffer) {
        let size = buffer.size();
        if size > 1024 * 1024 {
            log::debug!("缓冲区过大 (>1MB)，直接销毁: size={}", size);
            return;
        }

        let key = PoolKey {
            size_class: size_class(size),
            usage: BufferUsage::Storage,
        };

        let mut pools = self.pools.borrow_mut();
        let entry = pools.entry(key).or_default();
        if entry.len() < self.max_per_class {
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
        if size > 1024 * 1024 {
            return;
        }
        let mut pools = self.staging_pools.borrow_mut();
        let entry = pools.entry(size).or_default();
        if entry.len() < self.max_per_class {
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
    }
}