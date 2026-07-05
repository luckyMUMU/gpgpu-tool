//! 全局 poll(Wait) 调用计数器。
//!
//! 用于量化 GPU 同步开销：每次 `device.poll(wgpu::Maintain::Wait)` 调用
//! 都会阻塞 CPU 等待 GPU 完成，是性能分析的关键指标。
//!
//! # 用法
//!
//! ```ignore
//! use gpgpu_tool::poll_counter;
//!
//! poll_counter::reset();
//! // ... 执行 GPU 操作 ...
//! let count = poll_counter::get();
//! println!("poll(Wait) 调用次数: {}", count);
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

/// 全局 poll(Wait) 调用计数器。
static POLL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// 递增 poll(Wait) 计数器（每次 `device.poll(Maintain::Wait)` 调用时触发）。
///
/// 使用 `Relaxed` 排序以最小化开销——我们只关心最终计数的准确性，
/// 不需要与其他内存操作建立 happens-before 关系。
#[inline]
pub fn increment() {
    POLL_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// 读取当前 poll(Wait) 调用次数。
#[inline]
pub fn get() -> usize {
    POLL_COUNT.load(Ordering::Relaxed)
}

/// 重置计数器为 0。
#[inline]
pub fn reset() {
    POLL_COUNT.store(0, Ordering::Relaxed);
}
