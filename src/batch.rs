use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder};

use crate::buffer::GpuBuffer;
use crate::error::GpuError;
use crate::pipeline::{BindingType, ComputePipeline};

/// 批量计算任务描述，算法无关。
///
/// 业务层构建此结构体描述一次 GPU dispatch 所需的所有资源，
/// 然后通过 `GpuBatchSubmitter::submit()` 提交到批量队列。
///
/// `buffers` 的顺序必须与管线 `PipelineDescriptor::bindings` 一一对应。
/// `submit()` 会自动从管线绑定布局中找到第一个 `StorageReadWrite` 缓冲区
/// 作为输出，用于 staging buffer 回读。
pub struct BatchJob {
    /// 计算管线
    pub pipeline: Arc<ComputePipeline>,
    /// 绑定缓冲区列表，顺序与管线 bindings 一一对应
    pub buffers: Vec<GpuBuffer>,
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
    ///
    /// # Errors
    ///
    /// 当 GPU 设备或队列不可用（纯 CPU 降级模式）时返回 `GpuError::CpuFallback`。
    ///
    /// # Panics
    ///
    /// 当 `job.buffers` 数量与管线 bindings 数量不匹配时 panic。
    pub fn submit(&mut self, ctx: &crate::context::GpuContext, job: BatchJob) -> Result<(), GpuError> {
        let device = ctx.device()?.clone();
        let queue = ctx.queue()?.clone();

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

        // 动态构建绑定列表
        let buffer_refs: Vec<&GpuBuffer> = job.buffers.iter().collect();
        assert!(
            buffer_refs.len() == job.pipeline.bindings().len(),
            "BatchJob buffers 数量 ({}) 与管线 bindings 数量 ({}) 不匹配",
            buffer_refs.len(),
            job.pipeline.bindings().len()
        );

        // 编码 dispatch
        job.pipeline.encode_dispatch_into(
            &device,
            encoder,
            &buffer_refs,
            job.dispatch,
            None,
            ctx.compute_units(),
        );

        // 从管线绑定布局中找到第一个 StorageReadWrite 缓冲区作为输出
        let output_index = job.pipeline.bindings().iter().position(|b| *b == BindingType::StorageReadWrite);
        if let Some(idx) = output_index {
            let output_buffer = &job.buffers[idx];
            let staging_size = output_buffer.size();
            let staging = ctx.buffer_pool().acquire_staging(&device, staging_size);
            encoder.copy_buffer_to_buffer(output_buffer.raw(), 0, &staging, 0, staging_size);

            self.pending.push(PendingJob {
                staging_buffer: staging,
            });
        }

        Ok(())
    }

    /// 统一提交并等待所有已提交任务完成，返回结果数据。
    ///
    /// 此方法调用一次 `queue.submit()` + `device.poll()`，
    /// 然后逐个映射 staging buffer 读取结果。
    ///
    /// 返回的 `Vec<Vec<u8>>` 与提交顺序一一对应。
    pub fn wait_all(&mut self, ctx: &crate::context::GpuContext) -> Result<Vec<Vec<u8>>, GpuError> {
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

        // 映射 staging buffer 读取结果（使用 channel 验证映射成功）
        let mut results = Vec::with_capacity(self.pending.len());
        let mut map_rxs: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(self.pending.len());

        for pending in &self.pending {
            let (tx, rx) = std::sync::mpsc::channel();
            pending.staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            map_rxs.push(rx);
        }

        device.poll(wgpu::Maintain::Wait);

        // 验证所有映射成功
        for rx in &map_rxs {
            match rx
                .recv()
                .map_err(|_| GpuError::MapFailed("批量映射通道关闭".into()))?
            {
                Ok(()) => {}
                Err(e) => return Err(GpuError::MapFailed(format!("批量映射失败: {}", e))),
            }
        }

        for pending in self.pending.drain(..) {
            let view = pending.staging_buffer.slice(..).get_mapped_range();
            let data = view.to_vec();
            drop(view);
            pending.staging_buffer.unmap();
            ctx.buffer_pool().release_staging(pending.staging_buffer);
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

impl Drop for GpuBatchSubmitter {
    fn drop(&mut self) {
        // 清理 pending jobs（staging buffers 由 wgpu 自动回收）
        // 清理未提交的 encoder（丢弃未执行的 GPU 命令）
        // 不执行任何 GPU 操作（如 queue.submit），仅释放 Rust 端资源
        self.clear();
    }
}