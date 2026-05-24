use std::collections::HashMap;
use std::sync::Arc;

use wgpu::{Adapter, Device, Instance, Limits, Queue};

use crate::ComputeBackend;
use crate::error::GpuError;
use crate::pipeline::{BindingType, ComputePipeline, PipelineDescriptor};

struct PipelineCache {
    entries: HashMap<(Vec<BindingType>, u64, [u32; 3]), Arc<ComputePipeline>>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        device: &Device,
        descriptor: &PipelineDescriptor,
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let hash = fxhash(descriptor.wgsl);
        let key = (descriptor.bindings.clone(), hash, descriptor.workgroup_size);

        if let Some(pipeline) = self.entries.get(&key) {
            log::debug!("管线缓存命中: hash={:016x}, wg={:?}", hash, descriptor.workgroup_size);
            return Ok(Arc::clone(pipeline));
        }

        log::debug!(
            "管线缓存未命中，编译着色器: hash={:016x}, wg={:?}",
            hash,
            descriptor.workgroup_size
        );
        let pipeline = ComputePipeline::create(device, descriptor)?;
        let arc = Arc::new(pipeline);
        self.entries.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    fn clear(&mut self) {
        self.entries.clear();
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
    backend: ComputeBackend,
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

    /// 正常 GPU 初始化：高性能适配器 + 硬件设备。
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wgpu-compute-engine"),
                    required_features: wgpu::Features::empty(),
                    required_limits: Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequest(e.to_string()))?;

        let limits = device.limits();

        Ok(Self {
            _instance: instance,
            adapter: Some(adapter),
            device: Some(device),
            queue: Some(queue),
            limits,
            pipeline_cache: PipelineCache::new(),
            backend: ComputeBackend::Gpu,
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

        let limits = device.limits();

        Ok(Self {
            _instance: instance,
            adapter: Some(adapter),
            device: Some(device),
            queue: Some(queue),
            limits,
            pipeline_cache: PipelineCache::new(),
            backend: ComputeBackend::Cpu,
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
            backend: ComputeBackend::Cpu,
        }
    }

    /// 返回当前计算后端。
    pub fn backend(&self) -> ComputeBackend {
        self.backend
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
    pub fn clear_pipeline_cache(&mut self) {
        self.pipeline_cache.clear();
    }

    /// 返回 GPU 设备引用。
    ///
    /// GPU 模式和软件渲染降级模式下始终可用。
    /// 纯 CPU 降级模式下调用会 panic（此模式下不应执行 GPU 操作）。
    pub fn device(&self) -> &Device {
        self.device
            .as_ref()
            .expect("CPU 降级模式下无 GPU 设备，不应调用 GPU 操作")
    }

    /// 返回 GPU 队列引用。
    ///
    /// GPU 模式和软件渲染降级模式下始终可用。
    /// 纯 CPU 降级模式下调用会 panic（此模式下不应执行 GPU 操作）。
    pub fn queue(&self) -> &Queue {
        self.queue
            .as_ref()
            .expect("CPU 降级模式下无 GPU 队列，不应调用 GPU 操作")
    }

    /// 安全获取 GPU 设备引用，纯 CPU 降级模式下返回 `None`。
    pub fn try_device(&self) -> Option<&Device> {
        self.device.as_ref()
    }

    /// 安全获取 GPU 队列引用，纯 CPU 降级模式下返回 `None`。
    pub fn try_queue(&self) -> Option<&Queue> {
        self.queue.as_ref()
    }
}
