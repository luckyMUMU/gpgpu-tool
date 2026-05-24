use wgpu::{
    BindGroupLayout, BufferUsages, CommandEncoder,
    ComputePipeline as WgpuComputePipeline, Device, Queue,
};

use crate::buffer::GpuBuffer;
use crate::error::GpuError;

/// 绑定类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    StorageReadOnly,
    StorageReadWrite,
    Uniform,
}

impl BindingType {
    /// 返回此绑定类型对应的 wgpu BufferBindingType。
    pub fn buffer_binding_type(&self) -> wgpu::BufferBindingType {
        match self {
            BindingType::StorageReadOnly => wgpu::BufferBindingType::Storage { read_only: true },
            BindingType::StorageReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
            BindingType::Uniform => wgpu::BufferBindingType::Uniform,
        }
    }

    /// 返回此绑定类型对应的 BufferUsages。
    pub fn buffer_usages(&self) -> BufferUsages {
        match self {
            BindingType::StorageReadOnly => BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BindingType::StorageReadWrite => {
                BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
            }
            BindingType::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }
    }
}

/// 管线描述符，定义绑定布局。
#[derive(Debug, Clone)]
pub struct PipelineDescriptor {
    pub bindings: Vec<BindingType>,
    pub wgsl: &'static str,
    pub workgroup_size: [u32; 3],
}

impl PipelineDescriptor {
    /// 默认 3-binding 布局（StorageReadOnly, StorageReadWrite, Uniform）。
    pub fn default_3_binding(wgsl: &'static str, workgroup_size: [u32; 3]) -> Self {
        Self {
            bindings: vec![
                BindingType::StorageReadOnly,
                BindingType::StorageReadWrite,
                BindingType::Uniform,
            ],
            wgsl,
            workgroup_size,
        }
    }
}

/// 计算管线，封装 WGSL 着色器的编译、绑定组创建和 dispatch 调度。
///
/// wgpu-compute-engine 内部使用 `PipelineCache` 实现着色器级缓存，
/// 同一 WGSL 源码 + workgroup size 组合只编译一次。
///
/// # 使用方式
///
/// 通常不直接构造，而是通过 `GpuContext::get_or_create_pipeline()` 获取缓存实例：
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, PipelineDescriptor};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let pipeline = ctx.get_or_create_pipeline(
///     &PipelineDescriptor::default_3_binding(
///         include_str!("path/to/shader.wgsl"),
///         [256, 1, 1],
///     ),
/// ).unwrap();
/// ```
pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
    bind_group_layout: BindGroupLayout,
    bindings: Vec<BindingType>,
    workgroup_size: [u32; 3],
}

impl ComputePipeline {
    /// 从管线描述符编译计算管线。
    ///
    /// `descriptor` 指定绑定布局、WGSL 源码和 workgroup 尺寸。
    pub fn create(device: &Device, descriptor: &PipelineDescriptor) -> Result<Self, GpuError> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(descriptor.wgsl.into()),
        });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptor
            .bindings
            .iter()
            .enumerate()
            .map(|(i, bt)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: bt.buffer_binding_type(),
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            bindings: descriptor.bindings.clone(),
            workgroup_size: descriptor.workgroup_size,
        })
    }

    fn create_bind_group(&self, device: &Device, buffers: &[&GpuBuffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.raw().as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        })
    }

    /// 执行一次 GPU dispatch，内部创建 encoder、绑定资源和提交。
    ///
    /// 适用于独立调用的场景。对于批量提交场景使用
    /// [`encode_dispatch_into`](ComputePipeline::encode_dispatch_into)。
    pub fn dispatch(
        &self,
        device: &Device,
        queue: &Queue,
        buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
    ) {
        let bind_group = self.create_bind_group(device, buffers);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count[0], dispatch_count[1], dispatch_count[2]);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    #[doc(hidden)]
    pub fn encode_dispatch_into(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
    ) {
        let bind_group = self.create_bind_group(device, buffers);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_count[0], dispatch_count[1], dispatch_count[2]);
        }
    }

    /// 返回此管线的 workgroup 尺寸。
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// 返回此管线的绑定类型列表。
    pub fn bindings(&self) -> &[BindingType] {
        &self.bindings
    }
}