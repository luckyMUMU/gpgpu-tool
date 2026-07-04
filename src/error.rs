use thiserror::Error;

/// GPU 计算错误类型，覆盖设备初始化、着色器编译、缓冲区操作等全生命周期场景。
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("未找到支持计算管线的 GPU 适配器")]
    NoAdapter,

    #[error("GPU 设备请求失败: {0}")]
    DeviceRequest(String),

    #[error("着色器编译失败: {0}")]
    ShaderCompile(String),

    #[error("缓冲区映射失败: {0}")]
    MapFailed(String),

    #[error("GPU 验证错误: {0}")]
    Validation(String),

    #[error("GPU 设备丢失")]
    DeviceLost,

    #[error("GPU 显存不足: 请求 {requested} 字节，限制 {limit} 字节")]
    Oom { requested: u64, limit: u64 },

    #[error("GPU 内部错误: {0}")]
    Internal(String),

    #[error("输入参数无效: {0}")]
    InvalidInput(String),

    #[error("CPU 降级执行失败: {0}")]
    CpuFallback(String),

    #[error("GPU 计算超时 ({ms}ms)")]
    Timeout { ms: u64 },

    /// 无可用 GPU 适配器，或 GPU 初始化失败但非致命。
    ///
    /// 与 [`NoAdapter`](GpuError::NoAdapter) 的区别：
    /// - `NoAdapter` 表示 wgpu 初始化阶段未找到适配器（内部错误）
    /// - `GpuUnavailable` 表示集成层显式判定 GPU 不可用，调用方可据此降级
    ///
    /// 集成场景（如 czkawka）应捕获此错误并降级到 CPU 路径，
    /// 而非将错误传播给最终用户。
    #[error("GPU 不可用，调用方可降级到 CPU 路径")]
    GpuUnavailable,
}
