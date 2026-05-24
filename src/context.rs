use std::collections::HashMap;
use std::sync::Arc;

use wgpu::{Adapter, Device, Instance, Limits, Queue};

use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

struct PipelineCache {
    entries: HashMap<(u64, [u32; 3]), Arc<ComputePipeline>>,
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
        wgsl_source: &str,
        workgroup_size: [u32; 3],
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        let hash = fxhash(wgsl_source);
        let key = (hash, workgroup_size);

        if let Some(pipeline) = self.entries.get(&key) {
            log::debug!("管线缓存命中: hash={:016x}, wg={:?}", hash, workgroup_size);
            return Ok(Arc::clone(pipeline));
        }

        log::debug!(
            "管线缓存未命中，编译着色器: hash={:016x}, wg={:?}",
            hash,
            workgroup_size
        );
        let pipeline = ComputePipeline::create(device, wgsl_source, workgroup_size)?;
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
/// # 示例
///
/// ```no_run
/// use wgpu_compute_engine::GpuContext;
///
/// let ctx = GpuContext::new_sync().unwrap();
/// println!("适配器: {}", ctx.adapter_info());
/// ```
pub struct GpuContext {
    /// wgpu 要求 Instance 生命周期覆盖所有 GPU 资源，仅用于保活，不直接读取。
    _instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    limits: Limits,
    pipeline_cache: PipelineCache,
}

impl GpuContext {
    /// 异步创建 GPU 上下文，自动选择高性能适配器。
    pub async fn new() -> Result<Self, GpuError> {
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
            adapter,
            device,
            queue,
            limits,
            pipeline_cache: PipelineCache::new(),
        })
    }

    /// 同步创建 GPU 上下文（基于 pollster 阻塞等待）。
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    /// 返回适配器信息字符串（名称 + 后端类型）。
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }

    /// 返回硬件限制（max_storage_buffer_binding_size 等）。
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// 从内部管线缓存获取或创建计算管线。
    ///
    /// 首次调用会编译着色器并缓存，后续调用直接返回缓存结果。
    pub fn get_or_create_pipeline(
        &mut self,
        wgsl_source: &str,
        workgroup_size: [u32; 3],
    ) -> Result<Arc<ComputePipeline>, GpuError> {
        self.pipeline_cache
            .get_or_create(&self.device, wgsl_source, workgroup_size)
    }

    /// 清理所有缓存的计算管线。
    pub fn clear_pipeline_cache(&mut self) {
        self.pipeline_cache.clear();
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}
