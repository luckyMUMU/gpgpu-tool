use wgpu::{BindGroupLayout, ComputePipeline as WgpuComputePipeline, Device, Queue};
use wgpu::CommandEncoder;

use crate::buffer::GpuBuffer;
use crate::error::GpuError;

/// GPU 计算管线，封装 wgpu ComputePipeline 和 BindGroupLayout。
///
/// 负责执行 dispatch 操作（构建 bind group、编码命令、提交队列）。
/// 通过 `GpuContext::get_or_create_pipeline()` 获取，业务层持有 `Arc<ComputePipeline>`。
pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: [u32; 3],
}

impl ComputePipeline {
    /// 从 WGSL 源码创建计算管线（编译着色器 + 创建 bind group layout）。
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

    /// 执行计算调度：构建 bind group、编码 compute pass、提交队列。
    pub fn dispatch(
        &self,
        device: &Device,
        queue: &Queue,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
        dispatch_count: [u32; 3],
    ) {
        let encoder = self.encode_dispatch(
            device,
            input_buffer,
            output_buffer,
            params_buffer,
            dispatch_count,
        );
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// 编码计算调度到大 encoder 中（不提交），用于批量提交场景。
    pub fn encode_dispatch_into(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
        dispatch_count: [u32; 3],
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
    }

    /// 编码计算调度到独立 encoder 中，返回 encoder（调用者负责 submit）。
    fn encode_dispatch(
        &self,
        device: &Device,
        input_buffer: &GpuBuffer,
        output_buffer: &GpuBuffer,
        params_buffer: &GpuBuffer,
        dispatch_count: [u32; 3],
    ) -> CommandEncoder {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        });

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

        encoder
    }

    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}
