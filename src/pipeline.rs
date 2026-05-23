use wgpu::{
    BindGroupLayout, CommandEncoder, ComputePipeline as WgpuComputePipeline, Device, Queue,
};

use crate::buffer::GpuBuffer;
use crate::error::GpuError;

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
/// use wgpu_compute_engine::{GpuContext, GpuBuffer, BufferUsage};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let pipeline = ctx.get_or_create_pipeline(
///     include_str!("path/to/shader.wgsl"),
///     [256, 1, 1],
/// ).unwrap();
/// ```
pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: [u32; 3],
}

impl ComputePipeline {
    /// 从 WGSL 源码编译计算管线。
    ///
    /// `workgroup_size` 指定着色器 `@workgroup_size(x, y, z)` 参数，
    /// 用于计算 dispatch 时的 workgroup 数量。
    pub fn create(
        device: &Device,
        wgsl_source: &str,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let bind_group_layout_entry_0 = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout_entry_1 = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout_entry_2 = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &[
                bind_group_layout_entry_0,
                bind_group_layout_entry_1,
                bind_group_layout_entry_2,
            ],
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
            workgroup_size,
        })
    }

    fn create_bind_group(
        &self,
        device: &Device,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.raw().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.raw().as_entire_binding(),
                },
            ],
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
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
        dispatch_count: [u32; 3],
    ) {
        let bind_group = self.create_bind_group(device, input_buffer, output_buffer, params_buffer);

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
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
        dispatch_count: [u32; 3],
    ) {
        let bind_group = self.create_bind_group(device, input_buffer, output_buffer, params_buffer);

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
}