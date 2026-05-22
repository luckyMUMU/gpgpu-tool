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
        BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    }
}

pub struct BufferPool {
    pools: RefCell<HashMap<PoolKey, Vec<Buffer>>>,
    staging_pools: RefCell<HashMap<u64, Vec<Buffer>>>,
    max_per_class: usize,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            pools: RefCell::new(HashMap::new()),
            staging_pools: RefCell::new(HashMap::new()),
            max_per_class: 8,
        }
    }

    pub fn acquire(&self, device: &Device, size: u64, usage: BufferUsage) -> Buffer {
        let class = size_class(size);
        let key = PoolKey { size_class: class, usage };
        let mut pools = self.pools.borrow_mut();

        if let Some(vec) = pools.get_mut(&key) {
            while let Some(buffer) = vec.pop() {
                if buffer.size() >= size {
                    return buffer;
                }
            }
        }

        let next_class = class.saturating_mul(2);
        let next_key = PoolKey { size_class: next_class, usage };
        if let Some(vec) = pools.get_mut(&next_key) {
            if let Some(buffer) = vec.pop() {
                if buffer.size() >= size {
                    return buffer;
                }
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_pool"),
            size: class,
            usage: to_wgpu_usage(usage),
            mapped_at_creation: false,
        })
    }

    pub fn release(&self, buffer: Buffer) {
        let size = buffer.size();
        let class = size_class(size);
        let usage = infer_usage(&buffer);
        let key = PoolKey { size_class: class, usage };

        let mut pools = self.pools.borrow_mut();
        let vec = pools.entry(key).or_default();
        if vec.len() < self.max_per_class {
            vec.push(buffer);
        }
    }

    pub fn acquire_staging(&self, device: &Device, size: u64) -> Buffer {
        let class = size_class(size);
        let mut pools = self.staging_pools.borrow_mut();

        if let Some(vec) = pools.get_mut(&class) {
            while let Some(buffer) = vec.pop() {
                if buffer.size() >= size {
                    return buffer;
                }
            }
        }

        let next_class = class.saturating_mul(2);
        if let Some(vec) = pools.get_mut(&next_class) {
            if let Some(buffer) = vec.pop() {
                if buffer.size() >= size {
                    return buffer;
                }
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_pool"),
            size: class,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn release_staging(&self, buffer: Buffer) {
        let size = buffer.size();
        let class = size_class(size);

        let mut pools = self.staging_pools.borrow_mut();
        let vec = pools.entry(class).or_default();
        if vec.len() < self.max_per_class {
            vec.push(buffer);
        }
    }

    pub fn clear(&self) {
        self.pools.borrow_mut().clear();
        self.staging_pools.borrow_mut().clear();
    }

    pub fn cached_count(&self) -> usize {
        self.pools.borrow().values().map(|v| v.len()).sum()
    }
}

fn infer_usage(buffer: &Buffer) -> BufferUsage {
    let usage = buffer.usage();
    if usage.contains(BufferUsages::STORAGE) {
        BufferUsage::Storage
    } else {
        BufferUsage::Uniform
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
        assert_eq!(size_class(512), 512);
        assert_eq!(size_class(1000), 1024);
        assert_eq!(size_class(1025), 2048);
    }
}
