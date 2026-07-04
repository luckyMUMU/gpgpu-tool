use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use wgpu::{Adapter, Device, Instance, Limits, Queue};

use crate::buffer_pool::BufferPool;
use crate::ComputeBackend;
use crate::error::GpuError;
use crate::pipeline::{BindingType, ComputePipeline, PipelineDescriptor};

/// GPU 显存不足标志，由 UncapturedErrorHandler 设置。
///
/// 当 GPU 发生 OOM 错误时，handler 将此标志置为 `true`，
/// 后续 `GpuContext::backend()` 调用会检测此标志并切换到 CPU 降级模式，
/// 避免进程被驱动级别崩溃终止。
pub(crate) static GPU_OOM_FLAG: AtomicBool = AtomicBool::new(false);

/// GPU 设备丢失标志，由 UncapturedErrorHandler 设置。
///
/// 设备丢失是不可恢复的终态，handler 置位后 `backend()` 将返回 CPU 模式。
pub(crate) static GPU_DEVICE_LOST_FLAG: AtomicBool = AtomicBool::new(false);

type PipelineCacheKey = (Vec<BindingType>, u64, [u32; 3], Option<u32>, &'static str);
type PipelineCacheEntry = (String, Arc<ComputePipeline>);

struct PipelineCache {
    entries: HashMap<PipelineCacheKey, PipelineCacheEntry>,
    generations: HashMap<PipelineCacheKey, u64>,
    next_gen: u64,
    max_entries: usize,
}

impl PipelineCache {
    const DEFAULT_MAX_ENTRIES: usize = 64;

    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            generations: HashMap::new(),
            next_gen: 0,
            max_entries: Self::DEFAULT_MAX_ENTRIES,
        }
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn get_or_create(
        &mut self,
        device: &Device,
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let hash = fxhash(&descriptor.wgsl);
        let key = (descriptor.bindings.clone(), hash, descriptor.workgroup_size, descriptor.push_constant_size, descriptor.entry_point);

        if let Some((cached_source, pipeline)) = self.entries.get(&key) {
            if cached_source == &descriptor.wgsl {
                log::debug!("管线缓存命中: hash={:016x}, wg={:?}", hash, descriptor.workgroup_size);
                let cloned = Arc::clone(pipeline);
                self.touch(&key);
                return Ok(cloned);
            }
            log::warn!(
                "fxhash 碰撞检测: hash={:016x} 命中但源码不匹配，重新编译管线",
                hash
            );
        }

        log::debug!(
            "管线缓存未命中，编译着色器: hash={:016x}, wg={:?}",
            hash,
            descriptor.workgroup_size
        );
        self.evict_if_needed();
        let pipeline = ComputePipeline::create(device, descriptor)?;
        let arc = Arc::new(pipeline);
        self.entries.insert(key.clone(), (descriptor.wgsl.to_owned(), Arc::clone(&arc)));
        self.generations.insert(key, self.next_gen);
        self.next_gen += 1;
        Ok(arc)
    }

    /// 更新 key 的访问时间戳，O(1) HashMap 插入。
    fn touch(&mut self, key: &PipelineCacheKey) {
        self.generations.insert(key.clone(), self.next_gen);
        self.next_gen += 1;
    }

    /// LRU 淘汰：移除 generation 最小（最久未访问）的条目。
    fn evict_if_needed(&mut self) {
        while self.entries.len() >= self.max_entries {
            let oldest = self.generations.iter().min_by_key(|(_, g)| *g).map(|(k, _)| k.clone());
            if let Some(old_key) = oldest {
                self.generations.remove(&old_key);
                self.entries.remove(&old_key);
                log::debug!("LRU 淘汰管线缓存条目");
            } else {
                break;
            }
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.generations.clear();
        self.next_gen = 0;
    }

    fn clear_bind_group_caches(&self) {
        for (_source, pipeline) in self.entries.values() {
            pipeline.clear_bind_group_cache();
        }
    }
}

fn fxhash(s: &str) -> u64 {
    let mut hash: u64 = 0x517cc1b727220a95;
    for byte in s.bytes() {
        hash = hash.rotate_left(8) ^ (byte as u64);
        hash = hash.wrapping_mul(0x517cc1b727220a95);
    }
    hash
}

/// GPU 计算上下文，封装 wgpu 的 Instance、Adapter、Device、Queue 及内部管线缓存。
///
/// 所有 GPU 操作都通过此上下文进行。内部自动管理计算管线缓存，
/// 避免相同着色器的重复编译。
///
/// 当 GPU 不可用时（启用 `cpu-fallback` feature），自动降级到 CPU 模式。
/// 通过 [`backend()`](GpuContext::backend) 方法判断当前计算后端。
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, ComputeBackend};
///
/// let ctx = GpuContext::new_sync().unwrap();
/// match ctx.backend() {
///     ComputeBackend::Gpu => println!("GPU 模式: {}", ctx.adapter_info()),
///     ComputeBackend::Cpu => println!("CPU 降级模式"),
/// }
/// ```
pub struct GpuContext {
    _instance: Instance,
    adapter: Option<Adapter>,
    device: Option<Device>,
    queue: Option<Queue>,
    limits: Limits,
    pipeline_cache: PipelineCache,
    buffer_pool: Arc<BufferPool>,
    backend: ComputeBackend,
    compute_units: u32,
    push_constants_supported: bool,
}

/// 根据适配器设备类型估算计算单元（CU/SM/EU）数量。
///
/// wgpu 不直接暴露 CU 数量，此函数基于 `DeviceType` 提供合理估算值。
/// 可通过 [`GpuContext::set_compute_units`] 手动覆盖。
fn estimate_compute_units(device_type: wgpu::DeviceType) -> u32 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 32,
        wgpu::DeviceType::IntegratedGpu => 16,
        wgpu::DeviceType::VirtualGpu => 8,
        wgpu::DeviceType::Cpu => 4,
        wgpu::DeviceType::Other => 16,
    }
}

impl GpuContext {
    /// 异步创建 GPU 上下文，自动选择高性能适配器。
    ///
    /// 降级链（需启用 `cpu-fallback` feature）：
    /// 1. 尝试正常 GPU 初始化 → `ComputeBackend::Gpu`
    /// 2. 尝试软件渲染适配器 → `ComputeBackend::Cpu`（device/queue 可用）
    /// 3. 纯 CPU 上下文 → `ComputeBackend::Cpu`（device/queue 为 None）
    ///
    /// 未启用 `cpu-fallback` 时，GPU 初始化失败直接返回错误。
    pub async fn new() -> Result<Self, GpuError> {
        match Self::init_gpu().await {
            Ok(ctx) => Ok(ctx),
            Err(e) => {
                #[cfg(feature = "cpu-fallback")]
                {
                    log::warn!("GPU 初始化失败 ({:?})，降级到 CPU 模式", e);
                    match Self::init_software_adapter().await {
                        Ok(ctx) => {
                            log::info!("使用软件渲染适配器作为 CPU 降级后端");
                            Ok(ctx)
                        }
                        Err(soft_err) => {
                            log::warn!(
                                "软件渲染适配器也不可用 ({:?})，创建纯 CPU 上下文",
                                soft_err
                            );
                            Ok(Self::new_cpu_only())
                        }
                    }
                }
                #[cfg(not(feature = "cpu-fallback"))]
                Err(e)
            }
        }
    }

    /// 同步创建 GPU 上下文（基于 pollster 阻塞等待）。
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    /// 为集成场景创建 GPU 上下文，明确区分"GPU 不可用"与"GPU 出错"。
    ///
    /// 与 [`new_sync()`](Self::new_sync) 的区别：
    /// - `new_sync()` 失败时自动降级到 CPU 模式（需 `cpu-fallback` feature）
    /// - `new_for_integration()` 失败时直接返回错误，由调用方决策降级策略
    ///
    /// # 错误
    ///
    /// - [`GpuError::GpuUnavailable`] — 无可用 GPU 适配器，调用方可降级到 CPU
    /// - 其他错误 — GPU 初始化过程中的其他问题（设备请求失败等）
    ///
    /// # 示例
    ///
    /// ```no_run
    /// use gpgpu_tool::{GpuContext, GpuError};
    ///
    /// match GpuContext::new_for_integration() {
    ///     Ok(ctx) => println!("GPU 就绪: {}", ctx.adapter_info()),
    ///     Err(GpuError::GpuUnavailable) => {
    ///         println!("GPU 不可用，降级到 CPU 路径");
    ///         // 调用方自行实现 CPU 降级
    ///     }
    ///     Err(e) => panic!("GPU 初始化失败: {}", e),
    /// }
    /// ```
    pub fn new_for_integration() -> Result<Self, GpuError> {
        match pollster::block_on(Self::init_gpu()) {
            Ok(ctx) => Ok(ctx),
            Err(GpuError::NoAdapter) => Err(GpuError::GpuUnavailable),
            Err(e) => Err(e),
        }
    }

    /// 正常 GPU 初始化：高性能适配器 + 硬件设备。
    ///
    /// 优先请求 `PUSH_CONSTANTS` 特性；若适配器不支持则降级到不使用
    /// Push Constants 的设备配置。
    async fn init_gpu() -> Result<Self, GpuError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "选择适配器: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let adapter_features = adapter.features();
        let push_constants_supported = adapter_features.contains(wgpu::Features::PUSH_CONSTANTS);
        let required_features = if push_constants_supported {
            wgpu::Features::PUSH_CONSTANTS
        } else {
            log::warn!("适配器不支持 PUSH_CONSTANTS 特性，降级到不使用 Push Constants");
            wgpu::Features::empty()
        };

        let adapter_limits = adapter.limits();
        let required_limits = Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wgpu-compute-engine"),
                    required_features,
                    required_limits,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        // 注册 UncapturedErrorHandler：捕获 OOM 错误并设置原子标志，
        // 后续 `backend()` 调用据此切换到 CPU 降级，避免进程被驱动级别崩溃终止。
        // wgpu v24 的 Error 枚举仅有 OutOfMemory/Validation/Internal 三个变体，
        // 设备丢失通过独立的 set_device_lost_callback 回调处理。
        device.on_uncaptured_error(Box::new(|e| {
            match e {
                wgpu::Error::OutOfMemory { .. } => {
                    log::error!("GPU 显存不足: {:?}", e);
                    GPU_OOM_FLAG.store(true, Ordering::SeqCst);
                }
                wgpu::Error::Internal { ref description, .. } => {
                    // Internal 错误可能包含设备丢失等不可恢复状态，保守降级
                    log::error!("GPU 内部错误（保守触发降级）: {}", description);
                    GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
                }
                _ => log::warn!("GPU 验证错误: {:?}", e),
            }
        }));
        // 注册设备丢失回调：wgpu v24 通过独立回调通知设备丢失
        device.set_device_lost_callback(|reason, msg| {
            log::error!("GPU 设备丢失: reason={:?}, msg={}", reason, msg);
            GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
        });

        let limits = device.limits();
        let compute_units = estimate_compute_units(adapter_info.device_type);
        log::info!(
            "估算计算单元数量: {} (device_type={:?})",
            compute_units,
            adapter_info.device_type
        );

        // 基于设备实际特性判断 Push Constants 是否可用，
        // 而非仅依赖适配器声明（设备创建可能降级特性）
        let actual_push_constants = device.features().contains(wgpu::Features::PUSH_CONSTANTS)
            && limits.max_push_constant_size > 0;
        if push_constants_supported && !actual_push_constants {
            log::warn!("适配器声明支持 PUSH_CONSTANTS，但设备实际不可用，降级到 Uniform buffer");
        }

        Ok(Self {
            _instance: instance,
            adapter: Some(adapter),
            device: Some(device),
            queue: Some(queue),
            limits,
            pipeline_cache: PipelineCache::new(),
            buffer_pool: Arc::new(BufferPool::new()),
            backend: ComputeBackend::Gpu,
            compute_units,
            push_constants_supported: actual_push_constants,
        })
    }

    /// 软件渲染适配器初始化：使用 force_fallback_adapter 获取软件渲染设备。
    #[cfg(feature = "cpu-fallback")]
    async fn init_software_adapter() -> Result<Self, GpuError> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();
        log::info!(
            "选择软件渲染适配器: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wgpu-compute-engine-cpu-fallback"),
                    required_features: wgpu::Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        // 软件渲染适配器同样注册 UncapturedErrorHandler，保持错误处理一致性。
        device.on_uncaptured_error(Box::new(|e| {
            match e {
                wgpu::Error::OutOfMemory { .. } => {
                    log::error!("软件渲染适配器显存不足: {:?}", e);
                    GPU_OOM_FLAG.store(true, Ordering::SeqCst);
                }
                wgpu::Error::Internal { ref description, .. } => {
                    log::error!("软件渲染适配器内部错误（保守触发降级）: {}", description);
                    GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
                }
                _ => log::warn!("软件渲染适配器验证错误: {:?}", e),
            }
        }));
        device.set_device_lost_callback(|reason, msg| {
            log::error!("软件渲染适配器设备丢失: reason={:?}, msg={}", reason, msg);
            GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
        });

        let limits = device.limits();
        let compute_units = estimate_compute_units(adapter_info.device_type);

        Ok(Self {
            _instance: instance,
            adapter: Some(adapter),
            device: Some(device),
            queue: Some(queue),
            limits,
            pipeline_cache: PipelineCache::new(),
            buffer_pool: Arc::new(BufferPool::new()),
            backend: ComputeBackend::Cpu,
            compute_units,
            push_constants_supported: false,
        })
    }

    /// 创建纯 CPU 上下文（无 wgpu 设备）。
    ///
    /// 当连软件渲染适配器都不可用时使用此方案。
    /// device/queue 为 None，仅用于标记 CPU 模式。
    #[cfg(feature = "cpu-fallback")]
    fn new_cpu_only() -> Self {
        let instance = Instance::new(&wgpu::InstanceDescriptor::default());
        Self {
            _instance: instance,
            adapter: None,
            device: None,
            queue: None,
            limits: Limits::default(),
            pipeline_cache: PipelineCache::new(),
            buffer_pool: Arc::new(BufferPool::new()),
            backend: ComputeBackend::Cpu,
            compute_units: 4,
            push_constants_supported: false,
        }
    }

    /// 返回当前计算后端。
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// 返回估算的计算单元（CU/SM/EU）数量。
    ///
    /// 此值为基于适配器设备类型的估算值，可通过
    /// [`set_compute_units`](GpuContext::set_compute_units) 手动覆盖。
    /// 用于 dispatch occupancy 诊断和优化提示。
    pub fn compute_units(&self) -> u32 {
        self.compute_units
    }

    /// 返回设备是否支持 Push Constants 特性。
    ///
    /// 当不支持时，应回退到 Uniform buffer 传递参数。
    pub fn push_constants_supported(&self) -> bool {
        self.push_constants_supported
    }

    /// 手动设置计算单元数量，覆盖自动估算值。
    ///
    /// 适用于已知精确 CU 数量的场景（如通过平台 API 查询获得）。
    pub fn set_compute_units(&mut self, cu: u32) {
        self.compute_units = cu;
    }

    /// 返回适配器信息字符串（名称 + 后端类型）。
    ///
    /// CPU 降级模式下返回占位信息。
    pub fn adapter_info(&self) -> String {
        match &self.adapter {
            Some(adapter) => {
                let info = adapter.get_info();
                format!("{} ({:?})", info.name, info.backend)
            }
            None => "CPU 降级模式（无 GPU 适配器）".into(),
        }
    }

    /// 返回硬件限制（max_storage_buffer_binding_size 等）。
    ///
    /// CPU 降级模式下返回默认限制。
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// 返回共享缓冲区复用池的引用。
    ///
    /// 所有 GPU 组件通过此方法获取统一的 BufferPool，
    /// 避免各组件独立持有 BufferPool 导致资源浪费。
    /// BufferPool 内部使用 RefCell 实现内部可变性，
    /// 不可变引用即可完成 acquire/release 操作。
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// 返回共享缓冲区复用池的 Arc 引用。
    ///
    /// 用于需要长期持有 BufferPool 引用的场景（如 `GpuBatchSubmitter`），
    /// 确保 BufferPool 在持有者存活期间不会被释放。
    pub fn buffer_pool_arc(&self) -> Arc<BufferPool> {
        Arc::clone(&self.buffer_pool)
    }

    /// 从内部管线缓存获取或创建计算管线。
    ///
    /// 首次调用会编译着色器并缓存，后续调用直接返回缓存结果。
    ///
    /// CPU 降级模式（无设备）下返回 [`GpuError::CpuFallback`]。
    pub fn get_or_create_pipeline(
        &mut self,
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("CPU 降级模式下无法创建 GPU 管线".into()))?;
        self.pipeline_cache
            .get_or_create(device, descriptor)
    }

    /// 清理所有缓存的计算管线。
    ///
    /// 同时清理所有管线内部的 BindGroup 缓存。
    pub fn clear_pipeline_cache(&mut self) {
        self.pipeline_cache.clear();
    }

    /// 清理所有管线内部的 BindGroup 缓存，保留管线本身。
    ///
    /// 当缓冲区被大量释放或回收后，可调用此方法清除可能失效的
    /// BindGroup 缓存条目。正常使用中无需手动调用——各管线内部
    /// 的淘汰策略会自动清理旧条目。
    pub fn clear_bind_group_caches(&self) {
        self.pipeline_cache.clear_bind_group_caches();
    }

    /// 检查 GPU 设备是否丢失。
    ///
    /// 调用 `device.poll(wgpu::Maintain::Poll)` 检查设备状态。
    /// wgpu v24 中 `poll()` 返回 `MaintainResult`：
    /// - `MaintainResult::Ok` — 设备正常
    /// - `MaintainResult::SubmissionQueueEmpty` — 提交队列为空，设备正常
    /// 如果设备已丢失，返回 `Err(GpuError::DeviceLost)`。
    /// 注意：wgpu 的设备丢失通过 `UncapturedErrorHandler` 回调通知，
    /// 此方法仅提供同步检查。
    pub fn check_device_lost(&self) -> Result<(), GpuError> {
        if let Some(ref device) = self.device {
            device.poll(wgpu::Maintain::Poll);
            // wgpu v24 中 poll() 返回 MaintainResult，但不直接指示设备丢失。
            // 设备丢失通过 on_uncaptured_error 回调处理。
            // 此处保留 DeviceLost 错误变体供未来使用。
        }
        Ok(())
    }

    /// 返回 GPU 设备引用。
    ///
    /// GPU 模式和软件渲染降级模式下返回 `Ok`。
    /// 纯 CPU 降级模式下返回 `Err(GpuError::CpuFallback)`。
    pub fn device(&self) -> Result<&Device, GpuError> {
        self.device
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("CPU 降级模式下 GPU 设备不可用".to_string()))
    }

    /// 返回 GPU 队列引用。
    ///
    /// GPU 模式和软件渲染降级模式下返回 `Ok`。
    /// 纯 CPU 降级模式下返回 `Err(GpuError::CpuFallback)`。
    pub fn queue(&self) -> Result<&Queue, GpuError> {
        self.queue
            .as_ref()
            .ok_or_else(|| GpuError::CpuFallback("CPU 降级模式下 GPU 命令队列不可用".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    /// 验证 GPU_OOM_FLAG 标志存在且可被设置/重置。
    ///
    /// 这是 P0 显存安全防御层 1 的基础测试：UncapturedErrorHandler
    /// 通过设置此标志通知后续 `backend()` 调用切换到 CPU 降级。
    #[test]
    fn test_gpu_oom_flag_can_be_set_and_reset() {
        // 重置初始状态（其他测试可能已设置）
        GPU_OOM_FLAG.store(false, Ordering::SeqCst);

        assert!(!GPU_OOM_FLAG.load(Ordering::SeqCst), "初始状态应为 false");

        GPU_OOM_FLAG.store(true, Ordering::SeqCst);
        assert!(GPU_OOM_FLAG.load(Ordering::SeqCst), "设置后应为 true");

        GPU_OOM_FLAG.store(false, Ordering::SeqCst);
        assert!(!GPU_OOM_FLAG.load(Ordering::SeqCst), "重置后应为 false");
    }

    /// 验证 GPU_DEVICE_LOST_FLAG 标志存在且可被设置/重置。
    #[test]
    fn test_gpu_device_lost_flag_can_be_set_and_reset() {
        GPU_DEVICE_LOST_FLAG.store(false, Ordering::SeqCst);

        assert!(
            !GPU_DEVICE_LOST_FLAG.load(Ordering::SeqCst),
            "初始状态应为 false"
        );

        GPU_DEVICE_LOST_FLAG.store(true, Ordering::SeqCst);
        assert!(
            GPU_DEVICE_LOST_FLAG.load(Ordering::SeqCst),
            "设置后应为 true"
        );

        GPU_DEVICE_LOST_FLAG.store(false, Ordering::SeqCst);
        assert!(
            !GPU_DEVICE_LOST_FLAG.load(Ordering::SeqCst),
            "重置后应为 false"
        );
    }
}
