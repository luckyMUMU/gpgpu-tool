use crate::context::GpuContext;
use crate::error::GpuError;
use crate::ComputeBackend;

/// GPU/CPU 后端降级调度 trait。
///
/// 根据当前计算后端自动选择 GPU 或 CPU 执行路径，
/// 替代各业务模块中散布的 `if ctx.backend() == Cpu` 手动分支。
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuError, DefaultBackendDispatcher, BackendDispatcher};
///
/// fn do_work(ctx: &GpuContext) -> Result<Vec<u8>, GpuError> {
///     let dispatcher = DefaultBackendDispatcher;
///     dispatcher.dispatch_gpu(
///         ctx,
///         |ctx| gpu_compute(ctx),
///         || cpu_compute(),
///     )
/// }
/// # fn gpu_compute(_: &GpuContext) -> Result<Vec<u8>, GpuError> { Ok(vec![]) }
/// # fn cpu_compute() -> Result<Vec<u8>, GpuError> { Ok(vec![]) }
/// ```
pub trait BackendDispatcher {
    /// 根据 `ctx.backend()` 自动选择执行路径。
    ///
    /// - GPU 模式：调用 `gpu_fn(ctx)`
    /// - CPU 模式：调用 `cpu_fn()`
    ///
    /// 当处于 CPU 模式且未启用 `cpu-fallback` feature 时，返回 `GpuError::CpuFallback`。
    fn dispatch_gpu<F, C, R>(
        &self,
        ctx: &GpuContext,
        gpu_fn: F,
        cpu_fn: C,
    ) -> Result<R, GpuError>
    where
        F: FnOnce(&GpuContext) -> Result<R, GpuError>,
        C: FnOnce() -> Result<R, GpuError>;
}

/// 默认后端调度器，根据 `ctx.backend()` 自动选择 GPU 或 CPU 执行路径。
pub struct DefaultBackendDispatcher;

impl BackendDispatcher for DefaultBackendDispatcher {
    fn dispatch_gpu<F, C, R>(
        &self,
        ctx: &GpuContext,
        gpu_fn: F,
        cpu_fn: C,
    ) -> Result<R, GpuError>
    where
        F: FnOnce(&GpuContext) -> Result<R, GpuError>,
        C: FnOnce() -> Result<R, GpuError>,
    {
        if ctx.backend() == ComputeBackend::Cpu {
            #[cfg(feature = "cpu-fallback")]
            return cpu_fn();
            #[cfg(not(feature = "cpu-fallback"))]
            return Err(GpuError::CpuFallback("CPU 降级未启用".to_string()));
        }
        gpu_fn(ctx)
    }
}
