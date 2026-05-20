use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder};

use crate::buffer::GpuBuffer;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

/// 批量计算任务描述，算法无关。
///
/// 业务层构建此结构体描述一次 GPU dispatch 所需的所有资源，
/// 然后通过 `GpuBatchSubmitter::submit()` 提交到批量队列。
pub struct BatchJob {
    /// 输入数据缓冲区
    pub input: GpuBuffer,
    /// 输出数据缓冲区
    pub output: GpuBuffer,
    /// Uniform 参数缓冲区
    pub params: GpuBuffer,
    /// 计算管线
    pub pipeline: Arc<ComputePipeline>,
    /// Dispatch 尺寸
    pub dispatch: [u32; 3],
}

/// 已提交但未完成的批量任务槽位。
struct PendingJob {
    staging_buffer: Buffer,
}

/// GPU 异步批量提交器，能力层核心组件。
///
/// 将所有 dispatch + copy 命令编码到同一个 CommandEncoder，
/// 最终通过一次 `queue.submit()` 统一提交，大幅减少提交开销。
///
/// # 设计原则
///
/// - **算法无关**：不绑定任何具体算法，通过 `BatchJob` 描述任意计算任务
/// - **真批量提交**：所有 dispatch 编码到一个 encoder，一次 submit
/// - **延迟下载**：所有结果下载延迟到 `wait_all()` 统一执行
/// - **资源安全**：`wait_all()` 消费所有 pending jobs，无泄漏
pub struct GpuBatchSubmitter {
    encoder: Option<CommandEncoder>,
    pending: Vec<PendingJob>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
}

impl GpuBatchSubmitter {
    /// 创建新的批量提交器。
    pub fn new() -> Self {
        Self {
            encoder: None,
            pending: Vec::new(),
            device: None,
            queue: None,
        }
    }

    /// 提交一个计算任务到批量队列。
    ///
    /// 此方法**非阻塞**：仅编码 dispatch + copy 命令，
    /// 不提交到 GPU，也不等待 GPU 完成。
    pub fn submit(&mut self, ctx: &crate::context::GpuContext, job: BatchJob) {
        let device = ctx.device().clone();
        let queue = ctx.queue().clone();

        if self.encoder.is_none() {
            self.encoder = Some(device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("batch_encoder"),
                },
            ));
            self.device = Some(device.clone());
            self.queue = Some(queue.clone());
        }

        let encoder = self.encoder.as_mut().unwrap();

        // 编码 dispatch
        job.pipeline.encode_dispatch_into(
            &device,
            encoder,
            &job.input,
            &job.output,
            &job.params,
            job.dispatch,
        );

        // 编码 copy 到 staging
        let staging_size = job.output.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(job.output.raw(), 0, &staging, 0, staging_size);

        self.pending.push(PendingJob {
            staging_buffer: staging,
        });
    }

    /// 统一提交并等待所有已提交任务完成，返回结果数据。
    ///
    /// 此方法调用一次 `queue.submit()` + `device.poll()`，
    /// 然后逐个映射 staging buffer 读取结果。
    ///
    /// 返回的 `Vec<Vec<u8>>` 与提交顺序一一对应。
    pub fn wait_all(&mut self, _ctx: &crate::context::GpuContext) -> Result<Vec<Vec<u8>>, GpuError> {
        if self.pending.is_empty() {
            return Ok(vec![]);
        }

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // 提交所有累积的命令
        if let Some(encoder) = self.encoder.take() {
            queue.submit(std::iter::once(encoder.finish()));
        }

        // 等待 GPU 完成
        device.poll(wgpu::Maintain::Wait);

        // 映射 staging buffer 读取结果
        let mut results = Vec::with_capacity(self.pending.len());

        for pending in self.pending.drain(..) {
            pending
                .staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| {
                    if let Err(e) = result {
                        log::error!("批量 staging buffer 映射失败: {}", e);
                    }
                });

            device.poll(wgpu::Maintain::Wait);

            let view = pending.staging_buffer.slice(..).get_mapped_range();
            let data = view.to_vec();
            drop(view);
            pending.staging_buffer.unmap();

            results.push(data);
        }

        self.device = None;
        self.queue = None;

        Ok(results)
    }

    /// 返回当前待处理的任务数量。
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// 清空所有待处理任务（不获取结果，直接丢弃）。
    pub fn clear(&mut self) {
        self.pending.clear();
        self.encoder = None;
        self.device = None;
        self.queue = None;
    }
}

impl Default for GpuBatchSubmitter {
    fn default() -> Self {
        Self::new()
    }
}