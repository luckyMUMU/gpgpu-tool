use std::cell::RefCell;
use std::collections::HashMap;

use wgpu::{Buffer, BufferUsages, Device};

use crate::buffer::BufferUsage;

/// 缓冲区池，按尺寸分级复用 GPU 缓冲区，减少重复分配开销。
///
/// 采用 Size-Class 策略：将尺寸向上取整到最近的 2 的幂次（或固定档位），
/// 相同档位的缓冲区放入同一池子复用。
///
/// # 设计原则
///
/// - 只复用 Storage/Uniform 用途的缓冲区，不缓存 staging buffer（因 staging 需 MAP_READ）
/// - 缓冲区从池中取出后归调用者所有，用完后可选择归还
/// - 池本身不持有 Buffer 引用，归还时才重新存入
pub struct BufferPool {
    /// 按尺寸分档的可用缓冲区队列
    /// Key: 档位大小（字节），Value: 该档位的空闲 Buffer 列表
    pools: RefCell<HashMap<u64, Vec<Buffer>>>,
    /// 每个档位最大缓存数量，防止无限制增长
    max_per_class: usize,
}

/// 计算尺寸对应的档位（向上取整到最近的 2 的幂次，最小 256 字节）
fn size_class(size: u64) -> u64 {
    if size <= 256 {
        return 256;
    }
    let mut class = 256u64;
    while class < size {
        class = class.saturating_mul(2);
        if class > 1024 * 1024 * 1024 {
            // 超过 1GB 直接按原始尺寸，避免溢出
            return size;
        }
    }
    class
}

impl BufferPool {
    /// 创建新的缓冲区池。
    pub fn new() -> Self {
        Self {
            pools: RefCell::new(HashMap::new()),
            max_per_class: 8,
        }
    }

    /// 从池中获取或创建指定大小的缓冲区。
    ///
    /// 优先复用池中已有的空闲缓冲区（尺寸 >= 请求大小），
    /// 无可用时创建新的。
    pub fn acquire(&self, device: &Device, size: u64, usage: BufferUsage) -> Buffer {
        let class = size_class(size);
        let mut pools = self.pools.borrow_mut();

        // 尝试从对应档位获取
        if let Some(vec) = pools.get_mut(&class) {
            while let Some(buffer) = vec.pop() {
                // 验证尺寸是否足够（理论上同档位一定足够）
                if buffer.size() >= size {
                    return buffer;
                }
                // 尺寸不足的丢弃（不应发生）
            }
        }

        // 尝试从更大档位获取（避免浪费，只查一级）
        let next_class = class.saturating_mul(2);
        if let Some(vec) = pools.get_mut(&next_class) {
            if let Some(buffer) = vec.pop() {
                if buffer.size() >= size {
                    return buffer;
                }
            }
        }

        // 无可用，创建新缓冲区
        let wgpu_usage = match usage {
            BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        };

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buffer_pool"),
            size: class, // 按档位大小分配，便于复用
            usage: wgpu_usage,
            mapped_at_creation: false,
        })
    }

    /// 将使用完毕的缓冲区归还到池中。
    ///
    /// 如果该档位已满（超过 max_per_class），直接丢弃（由 wgpu 回收）。
    pub fn release(&self, buffer: Buffer) {
        let size = buffer.size();
        let class = size_class(size);

        let mut pools = self.pools.borrow_mut();
        let vec = pools.entry(class).or_default();
        if vec.len() < self.max_per_class {
            vec.push(buffer);
        }
        // 否则丢弃，由 wgpu 内部释放
    }

    /// 清空所有缓存的缓冲区。
    pub fn clear(&self) {
        self.pools.borrow_mut().clear();
    }

    /// 返回当前缓存的缓冲区总数（用于调试/监控）。
    pub fn cached_count(&self) -> usize {
        self.pools.borrow().values().map(|v| v.len()).sum()
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
