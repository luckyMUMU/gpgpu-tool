//! # 动态内存管理器
//!
//! 根据 GPU 显存限制、CPU 内存预算和图像尺寸动态调整批处理大小，
//! 支持运行时 OOM 回退和成功后自动扩展。
//!
//! ## 核心机制
//!
//! 1. **初始化估算**：根据 GPU `max_storage_buffer_binding_size` 和 CPU 内存预算，
//!    结合图像平均尺寸计算初始批次大小
//! 2. **运行时调整**：连续成功 3 次后尝试扩大批次（×1.5），OOM 时立即减半
//! 3. **内存跟踪**：跟踪每批次的预估内存使用量，为后续批次提供参考
//!
//! ## 使用示例
//!
//! ```no_run
//! use gpgpu_tool::{GpuContext, MemoryManager};
//!
//! let ctx = GpuContext::new_sync().unwrap();
//! let mut mm = MemoryManager::from_context(&ctx)
//!     .with_cpu_budget(1024 * 1024 * 1024) // 1GB CPU 预算
//!     .with_batch_range(2, 128);            // 批次范围 2~128
//!
//! // 根据图像尺寸计算批次大小
//! let batch_size = mm.calculate_batch_size(1920, 1080);
//! println!("批次大小: {}", batch_size);
//! ```

use crate::context::GpuContext;

/// 动态内存管理器，根据 GPU/CPU 资源动态调整批处理大小。
///
/// # 设计原则
///
/// - **GPU 约束**：单批次 GPU 缓冲区不超过 `max_storage_buffer_binding_size / 2`
///   （预留一半给中间缓冲区：缩放输出、哈希输出等）
/// - **CPU 约束**：单批次 CPU 内存不超过可配置的预算（默认 512MB）
/// - **渐进式调整**：成功后逐步扩大，OOM 后立即减半
pub struct MemoryManager {
    /// GPU 单缓冲区最大绑定大小（字节），来自 `wgpu::Limits`
    gpu_max_buffer_size: u64,
    /// GPU 批次缓冲区上限（通常为 `gpu_max_buffer_size / 2`）
    gpu_batch_buffer_limit: u64,
    /// CPU 内存预算（字节），每批次 CPU 侧数据不超过此值
    cpu_memory_budget: u64,
    /// 当前批次大小
    current_batch_size: usize,
    /// 最小批次大小（下限）
    min_batch_size: usize,
    /// 最大批次大小（上限）
    max_batch_size: usize,
    /// 连续成功计数（用于自动扩展）
    success_streak: u32,
    /// 上次操作是否发生 OOM
    last_oom: bool,
    /// 预估每张图像的 CPU 内存占用（字节）
    estimated_bytes_per_image: u64,
    /// 预估每张图像的 GPU 内存占用（字节）
    estimated_gpu_bytes_per_image: u64,
    /// 历史峰值内存使用（字节）
    peak_memory_usage: u64,
    /// 总处理图像数
    total_processed: u64,
    /// 总批次数
    total_batches: u64,
}

/// 扩展因子：连续成功后批次大小乘以 3/2
const SCALE_UP_FACTOR: usize = 3;
const SCALE_UP_DIVISOR: usize = 2;
/// 连续成功多少次后触发扩展
const SUCCESS_THRESHOLD: u32 = 3;
/// 默认 CPU 内存预算：512 MB
const DEFAULT_CPU_BUDGET: u64 = 512 * 1024 * 1024;
/// 默认最小批次大小
const DEFAULT_MIN_BATCH: usize = 2;
/// 默认最大批次大小
const DEFAULT_MAX_BATCH: usize = 256;
/// 默认初始批次大小
const DEFAULT_INITIAL_BATCH: usize = 32;

impl MemoryManager {
    /// 从 GPU 上下文创建内存管理器。
    ///
    /// 自动读取 `max_storage_buffer_binding_size` 作为 GPU 约束。
    pub fn from_context(ctx: &GpuContext) -> Self {
        let gpu_max = ctx.limits().max_storage_buffer_binding_size as u64;
        Self {
            gpu_max_buffer_size: gpu_max,
            gpu_batch_buffer_limit: gpu_max / 2,
            cpu_memory_budget: DEFAULT_CPU_BUDGET,
            current_batch_size: DEFAULT_INITIAL_BATCH,
            min_batch_size: DEFAULT_MIN_BATCH,
            max_batch_size: DEFAULT_MAX_BATCH,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 0,
            estimated_gpu_bytes_per_image: 0,
            peak_memory_usage: 0,
            total_processed: 0,
            total_batches: 0,
        }
    }

    /// 设置 CPU 内存预算（字节）。
    pub fn with_cpu_budget(mut self, budget: u64) -> Self {
        self.cpu_memory_budget = budget;
        self
    }

    /// 设置批次大小范围 `[min, max]`。
    pub fn with_batch_range(mut self, min: usize, max: usize) -> Self {
        self.min_batch_size = min;
        self.max_batch_size = max;
        self.current_batch_size = self.current_batch_size.clamp(min, max);
        self
    }

    /// 设置初始批次大小。
    pub fn with_initial_batch_size(mut self, size: usize) -> Self {
        self.current_batch_size = size.clamp(self.min_batch_size, self.max_batch_size);
        self
    }

    /// 根据图像平均尺寸计算最优批次大小。
    ///
    /// 综合考虑：
    /// - CPU 内存约束：RGBA(w×h×4) + 灰度(w×h) ≈ w×h×5 字节/图
    /// - GPU 缓冲区约束：u32 打包输入(w×h×4) + 缩放输出(target×target×4)
    /// - 上次 OOM 状态（如 OOM 则强制减半）
    pub fn calculate_batch_size(&mut self, avg_width: u32, avg_height: u32) -> usize {
        let pixels = avg_width as u64 * avg_height as u64;

        // CPU 侧每图内存：RGBA(4字节/像素) + 灰度(1字节/像素) = 5字节/像素
        let cpu_bytes_per_image = pixels * 5;
        // GPU 侧每图内存：u32 打包输入(4字节/像素) = 4字节/像素
        let gpu_bytes_per_image = pixels * 4;

        self.estimated_bytes_per_image = cpu_bytes_per_image;
        self.estimated_gpu_bytes_per_image = gpu_bytes_per_image;

        // CPU 内存约束
        let cpu_constrained = if cpu_bytes_per_image > 0 {
            (self.cpu_memory_budget / cpu_bytes_per_image) as usize
        } else {
            self.max_batch_size
        };

        // GPU 缓冲区约束
        let gpu_constrained = if gpu_bytes_per_image > 0 {
            (self.gpu_batch_buffer_limit / gpu_bytes_per_image) as usize
        } else {
            self.max_batch_size
        };

        // 取所有约束的最小值
        let calculated = cpu_constrained
            .min(gpu_constrained)
            .min(self.max_batch_size)
            .max(self.min_batch_size);

        // OOM 回退：强制减半
        if self.last_oom {
            self.current_batch_size =
                (self.current_batch_size / 2).max(self.min_batch_size);
            self.last_oom = false;
        } else {
            self.current_batch_size = calculated;
        }

        log::info!(
            "MemoryManager: batch_size={}, cpu_budget={}MB, gpu_limit={}MB, img={}×{} (cpu/img={}KB, gpu/img={}KB)",
            self.current_batch_size,
            self.cpu_memory_budget / 1024 / 1024,
            self.gpu_batch_buffer_limit / 1024 / 1024,
            avg_width,
            avg_height,
            cpu_bytes_per_image / 1024,
            gpu_bytes_per_image / 1024,
        );

        self.current_batch_size
    }

    /// 获取当前批次大小（不重新计算）。
    pub fn batch_size(&self) -> usize {
        self.current_batch_size
    }

    /// 手动设置批次大小（受 min/max 约束）。
    pub fn set_batch_size(&mut self, size: usize) {
        self.current_batch_size = size.clamp(self.min_batch_size, self.max_batch_size);
    }

    /// 报告批次处理成功，可能触发自动扩展。
    ///
    /// 连续成功 `SUCCESS_THRESHOLD` 次后，批次大小 ×1.5（不超过 max）。
    pub fn report_success(&mut self, images_processed: usize) {
        self.success_streak += 1;
        self.last_oom = false;
        self.total_processed += images_processed as u64;
        self.total_batches += 1;

        // 更新峰值内存使用估算
        let batch_mem = images_processed as u64 * self.estimated_bytes_per_image;
        if batch_mem > self.peak_memory_usage {
            self.peak_memory_usage = batch_mem;
        }

        if self.success_streak >= SUCCESS_THRESHOLD && self.current_batch_size < self.max_batch_size {
            let new_size = (self.current_batch_size * SCALE_UP_FACTOR / SCALE_UP_DIVISOR)
                .min(self.max_batch_size);
            if new_size > self.current_batch_size {
                log::debug!(
                    "MemoryManager: 自动扩展批次 {} -> {} (连续成功 {})",
                    self.current_batch_size, new_size, self.success_streak
                );
                self.current_batch_size = new_size;
            }
            self.success_streak = 0;
        }
    }

    /// 报告 OOM 或内存压力，立即减半批次大小。
    pub fn report_oom(&mut self) {
        self.last_oom = true;
        self.success_streak = 0;
        let new_size = (self.current_batch_size / 2).max(self.min_batch_size);
        log::warn!(
            "MemoryManager: OOM 检测，批次缩减 {} -> {}",
            self.current_batch_size, new_size
        );
        self.current_batch_size = new_size;
    }

    /// 获取 GPU 单缓冲区最大绑定大小（字节）。
    pub fn gpu_max_buffer_size(&self) -> u64 {
        self.gpu_max_buffer_size
    }

    /// 获取 GPU 批次缓冲区上限（字节）。
    pub fn gpu_batch_buffer_limit(&self) -> u64 {
        self.gpu_batch_buffer_limit
    }

    /// 获取 CPU 内存预算（字节）。
    pub fn cpu_memory_budget(&self) -> u64 {
        self.cpu_memory_budget
    }

    /// 获取预估每图 CPU 内存占用（字节）。
    pub fn estimated_bytes_per_image(&self) -> u64 {
        self.estimated_bytes_per_image
    }

    /// 获取预估每图 GPU 内存占用（字节）。
    pub fn estimated_gpu_bytes_per_image(&self) -> u64 {
        self.estimated_gpu_bytes_per_image
    }

    /// 获取历史峰值内存使用（字节）。
    pub fn peak_memory_usage(&self) -> u64 {
        self.peak_memory_usage
    }

    /// 获取已处理图像总数。
    pub fn total_processed(&self) -> u64 {
        self.total_processed
    }

    /// 获取已处理批次总数。
    pub fn total_batches(&self) -> u64 {
        self.total_batches
    }

    /// 生成内存统计摘要字符串。
    pub fn summary(&self) -> String {
        format!(
            "批次: {} (范围 {}-{}), CPU预算: {}MB, GPU限制: {}MB, \
             每图: {}KB(CPU)/{}KB(GPU), 峰值: {}MB, \
             已处理: {}图/{}批",
            self.current_batch_size,
            self.min_batch_size,
            self.max_batch_size,
            self.cpu_memory_budget / 1024 / 1024,
            self.gpu_batch_buffer_limit / 1024 / 1024,
            self.estimated_bytes_per_image / 1024,
            self.estimated_gpu_bytes_per_image / 1024,
            self.peak_memory_usage / 1024 / 1024,
            self.total_processed,
            self.total_batches,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_size_calculation_small_images() {
        // 模拟小图像 64×64
        // 不需要真实 GPU，直接构造 MemoryManager
        let mut mm = MemoryManager {
            gpu_max_buffer_size: 256 * 1024 * 1024, // 256MB
            gpu_batch_buffer_limit: 128 * 1024 * 1024, // 128MB
            cpu_memory_budget: 512 * 1024 * 1024,   // 512MB
            current_batch_size: 32,
            min_batch_size: 2,
            max_batch_size: 256,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 0,
            estimated_gpu_bytes_per_image: 0,
            peak_memory_usage: 0,
            total_processed: 0,
            total_batches: 0,
        };

        // 64×64 图像：cpu = 64*64*5 = 20480 字节, gpu = 64*64*4 = 16384 字节
        let batch = mm.calculate_batch_size(64, 64);
        // CPU: 512MB / 20KB = 26214, GPU: 128MB / 16KB = 8192
        // min(26214, 8192, 256) = 256
        assert_eq!(batch, 256);
    }

    #[test]
    fn test_batch_size_calculation_large_images() {
        let mut mm = MemoryManager {
            gpu_max_buffer_size: 256 * 1024 * 1024,
            gpu_batch_buffer_limit: 128 * 1024 * 1024,
            cpu_memory_budget: 512 * 1024 * 1024,
            current_batch_size: 32,
            min_batch_size: 2,
            max_batch_size: 256,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 0,
            estimated_gpu_bytes_per_image: 0,
            peak_memory_usage: 0,
            total_processed: 0,
            total_batches: 0,
        };

        // 4K 图像 3840×2160：cpu = 3840*2160*5 = 41472000 字节(~39.5MB)
        // CPU: 512MB / 39.5MB ≈ 12, GPU: 128MB / 33.2MB ≈ 3
        let batch = mm.calculate_batch_size(3840, 2160);
        assert!(batch >= 2 && batch <= 4, "4K 图像批次应在 2-4 之间, got {}", batch);
    }

    #[test]
    fn test_oom_backoff() {
        let mut mm = MemoryManager {
            gpu_max_buffer_size: 256 * 1024 * 1024,
            gpu_batch_buffer_limit: 128 * 1024 * 1024,
            cpu_memory_budget: 512 * 1024 * 1024,
            current_batch_size: 64,
            min_batch_size: 2,
            max_batch_size: 256,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 0,
            estimated_gpu_bytes_per_image: 0,
            peak_memory_usage: 0,
            total_processed: 0,
            total_batches: 0,
        };

        mm.report_oom();
        assert_eq!(mm.batch_size(), 32);

        mm.report_oom();
        assert_eq!(mm.batch_size(), 16);
    }

    #[test]
    fn test_success_scale_up() {
        let mut mm = MemoryManager {
            gpu_max_buffer_size: 256 * 1024 * 1024,
            gpu_batch_buffer_limit: 128 * 1024 * 1024,
            cpu_memory_budget: 512 * 1024 * 1024,
            current_batch_size: 32,
            min_batch_size: 2,
            max_batch_size: 256,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 0,
            estimated_gpu_bytes_per_image: 0,
            peak_memory_usage: 0,
            total_processed: 0,
            total_batches: 0,
        };

        // 3 次成功后应该扩展
        mm.report_success(32);
        mm.report_success(32);
        assert_eq!(mm.batch_size(), 32); // 还没到 3 次
        mm.report_success(32);
        assert_eq!(mm.batch_size(), 48); // 32 * 3/2 = 48
    }

    #[test]
    fn test_summary() {
        let mm = MemoryManager {
            gpu_max_buffer_size: 256 * 1024 * 1024,
            gpu_batch_buffer_limit: 128 * 1024 * 1024,
            cpu_memory_budget: 512 * 1024 * 1024,
            current_batch_size: 32,
            min_batch_size: 2,
            max_batch_size: 256,
            success_streak: 0,
            last_oom: false,
            estimated_bytes_per_image: 20480,
            estimated_gpu_bytes_per_image: 16384,
            peak_memory_usage: 655360,
            total_processed: 1000,
            total_batches: 32,
        };

        let s = mm.summary();
        assert!(s.contains("批次: 32"));
        assert!(s.contains("已处理: 1000图/32批"));
    }
}
