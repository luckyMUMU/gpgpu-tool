use std::sync::Arc;

use wgpu::{Buffer, CommandEncoder};

use crate::buffer::GpuBuffer;
use crate::buffer_pool::BufferPool;
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
    pool: Option<Arc<BufferPool>>,
}

impl GpuBatchSubmitter {
    /// 创建新的批量提交器。
    pub fn new() -> Self {
        Self {
            encoder: None,
            pending: Vec::new(),
            device: None,
            queue: None,
            pool: None,
        }
    }

    /// 确保内部 encoder 和设备引用已初始化。
    ///
    /// 惰性创建 CommandEncoder，首次调用时从 GpuContext 获取 device/queue/pool。
    /// 后续调用复用已有实例。
    fn ensure_initialized(&mut self, ctx: &crate::context::GpuContext) -> Result<(), GpuError> {
        if self.encoder.is_none() {
            let device = ctx.device()?.clone();
            let queue = ctx.queue()?.clone();
            self.encoder = Some(device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("batch_encoder"),
                },
            ));
            self.device = Some(device);
            self.queue = Some(queue);
            self.pool = Some(ctx.buffer_pool_arc());
        }
        Ok(())
    }

    /// 仅编码 dispatch 到批量队列，不创建 staging buffer。
    ///
    /// 用于中间计算步骤（结果保留在 GPU 上，不需要回读到 CPU）。
    /// 多个 `encode_dispatch()` 调用共享同一个 CommandEncoder，
    /// 最终通过 `wait_all()` 或 `flush()` 统一提交。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `pipeline` — 计算管线
    /// - `buffers` — 绑定缓冲区列表，顺序与管线 bindings 一一对应
    /// - `dispatch` — Dispatch 尺寸 [x, y, z]
    /// - `push_constants` — Push Constant 数据（`None` 表示不使用）
    /// - `compute_units` — GPU 计算单元数量，用于 occupancy 诊断
    ///
    /// # Errors
    ///
    /// 当 GPU 设备或队列不可用（纯 CPU 降级模式）时返回 `GpuError::CpuFallback`。
    pub fn encode_dispatch(
        &mut self,
        ctx: &crate::context::GpuContext,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        dispatch: [u32; 3],
        push_constants: Option<&[u8]>,
        compute_units: u32,
    ) -> Result<(), GpuError> {
        self.ensure_initialized(ctx)?;
        let device = self.device.as_ref().unwrap();
        let encoder = self.encoder.as_mut().unwrap();

        pipeline.encode_dispatch_into(
            device,
            encoder,
            buffers,
            dispatch,
            push_constants,
            compute_units,
        );

        Ok(())
    }

    /// 编码 buffer-to-buffer 拷贝到批量队列。
    ///
    /// 用于中间数据搬运（如合并多个 GPU 缓冲区），
    /// 拷贝命令与其他 dispatch 命令共享同一个 CommandEncoder。
    ///
    /// # 参数
    ///
    /// - `ctx` — GPU 上下文
    /// - `src` — 源缓冲区
    /// - `src_offset` — 源缓冲区起始偏移（字节）
    /// - `dst` — 目标缓冲区
    /// - `dst_offset` — 目标缓冲区起始偏移（字节）
    /// - `size` — 拷贝字节数
    pub fn encode_copy(
        &mut self,
        ctx: &crate::context::GpuContext,
        src: &wgpu::Buffer,
        src_offset: u64,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<(), GpuError> {
        self.ensure_initialized(ctx)?;
        let encoder = self.encoder.as_mut().unwrap();
        encoder.copy_buffer_to_buffer(src, src_offset, dst, dst_offset, size);
        Ok(())
    }

    /// 提交一个计算任务到批量队列。
    ///
    /// 此方法**非阻塞**：仅编码 dispatch + staging copy 命令，
    /// 不提交到 GPU，也不等待 GPU 完成。
    ///
    /// 与 `encode_dispatch()` 的区别：此方法自动为输出缓冲区创建 staging buffer
    /// 并编码 copy 命令，`wait_all()` 会自动读取结果。
    ///
    /// # Errors
    ///
    /// 当 GPU 设备或队列不可用（纯 CPU 降级模式）时返回 `GpuError::CpuFallback`。
    ///
    /// # Panics
    ///
    /// 当 `job.buffers` 数量与管线 bindings 数量不匹配时 panic。
    pub fn submit(&mut self, ctx: &crate::context::GpuContext, job: BatchJob) -> Result<(), GpuError> {
        self.ensure_initialized(ctx)?;
        let device = self.device.as_ref().unwrap().clone();
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

    /// 提交所有已编码的命令到 GPU，但不等待完成。
    ///
    /// 用于仅通过 `encode_dispatch()` 和 `encode_copy()` 编码命令、
    /// 不需要通过 staging buffer 回读结果的场景。
    /// 调用后 encoder 被消耗，后续可继续添加新命令。
    ///
    /// 如果同时有通过 `submit()` 添加的带 staging buffer 的任务，
    /// 它们也会被提交，但不会自动读取结果——需要手动管理。
    ///
    /// # Errors
    ///
    /// 当 encoder 未初始化（无已编码命令）时返回 `Ok(())` 而非错误。
    pub fn flush(&mut self) -> Result<(), GpuError> {
        if let Some(encoder) = self.encoder.take() {
            let queue = self.queue.as_ref()
                .ok_or(GpuError::Internal("GpuBatchSubmitter: queue 未初始化".into()))?;
            queue.submit(std::iter::once(encoder.finish()));
        }
        // encoder 已提交，重置引用（但保留 device/queue/pool 以便后续复用）
        Ok(())
    }

    /// 手动注册一个 staging buffer 用于结果收集。
    ///
    /// 当需要通过 `encode_dispatch()` 和 `encode_copy()` 手动编码命令时，
    /// 可以使用此方法将 staging buffer 注册到 pending 列表。
    /// `wait_all()` 会自动映射并读取所有注册的 staging buffer。
    ///
    /// # 参数
    ///
    /// - `staging_buffer` — 已编码 copy 命令的 staging buffer
    pub fn add_staging_buffer(&mut self, staging_buffer: wgpu::Buffer) {
        self.pending.push(PendingJob {
            staging_buffer,
        });
    }

    /// 统一提交并等待所有已提交任务完成，返回结果数据。
    ///
    /// 此方法调用一次 `queue.submit()` + `device.poll()`，
    /// 然后逐个映射 staging buffer 读取结果。
    ///
    /// 返回的 `Vec<Vec<u8>>` 与提交顺序一一对应。
    /// 如果没有 staging buffer（仅使用 `encode_dispatch()` 编码的命令），
    /// 仍会提交所有命令并等待完成，返回空 Vec。
    pub fn wait_all(&mut self) -> Result<Vec<Vec<u8>>, GpuError> {
        let device = self.device.as_ref()
            .ok_or(GpuError::Internal("GpuBatchSubmitter: device 未初始化".into()))?;
        let queue = self.queue.as_ref()
            .ok_or(GpuError::Internal("GpuBatchSubmitter: queue 未初始化".into()))?;

        // 提交所有累积的命令（包括仅 encode_dispatch 编码的命令）
        if let Some(encoder) = self.encoder.take() {
            queue.submit(std::iter::once(encoder.finish()));
        }

        if self.pending.is_empty() {
            // 无 staging buffer 但仍需等待 GPU 完成（如纯 encode_dispatch 场景）
            device.poll(wgpu::Maintain::Wait);
            self.device = None;
            self.queue = None;
            self.pool = None;
            return Ok(vec![]);
        }

        // 提交后立即注册所有 map_async 回调，然后单次 poll 同时等待 GPU 执行 + 映射完成
        let mut map_rxs: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(self.pending.len());

        for pending in &self.pending {
            let (tx, rx) = std::sync::mpsc::channel();
            pending.staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            map_rxs.push(rx);
        }

        // 单次 poll：同时等待 GPU 执行完成和 staging buffer 映射就绪
        device.poll(wgpu::Maintain::Wait);
        crate::poll_counter::increment();

        // 验证所有映射成功，同时记录映射状态以便错误路径清理
        let mut results = Vec::with_capacity(self.pending.len());
        let mut map_succeeded = Vec::with_capacity(self.pending.len());
        let mut first_error: Option<GpuError> = None;

        for rx in &map_rxs {
            let succeeded = match rx.recv() {
                Ok(Ok(())) => true,
                Ok(Err(ref e)) if first_error.is_none() => {
                    first_error = Some(GpuError::MapFailed(format!("批量映射失败: {}", e)));
                    false
                }
                Ok(Err(_)) => false,
                Err(_) if first_error.is_none() => {
                    first_error = Some(GpuError::MapFailed("批量映射通道关闭".into()));
                    false
                }
                Err(_) => false,
            };
            map_succeeded.push(succeeded);
        }

        if let Some(e) = first_error {
            // 错误路径：unmap 已映射的 staging buffer，归还所有到 pool
            for (i, pending) in self.pending.iter().enumerate() {
                if map_succeeded[i] {
                    pending.staging_buffer.unmap();
                }
            }
            if let Some(pool) = self.pool.as_ref() {
                for pending in self.pending.drain(..) {
                    pool.release_staging(pending.staging_buffer);
                }
            }
            self.device = None;
            self.queue = None;
            self.pool = None;
            return Err(e);
        }

        if let Some(pool) = self.pool.as_ref() {
            for pending in self.pending.drain(..) {
                let view = pending.staging_buffer.slice(..).get_mapped_range();
                let data = view.to_vec();
                drop(view);
                pending.staging_buffer.unmap();
                pool.release_staging(pending.staging_buffer);
                results.push(data);
            }
        }

        self.device = None;
        self.queue = None;
        self.pool = None;

        Ok(results)
    }

    /// 返回当前待处理的任务数量。
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// 清空所有待处理任务（不获取结果，直接丢弃）。
    ///
    /// staging buffer 会归还到 BufferPool 以便复用。
    pub fn clear(&mut self) {
        if let Some(pool) = self.pool.as_ref() {
            for pending in self.pending.drain(..) {
                pool.release_staging(pending.staging_buffer);
            }
        }
        self.pending.clear();
        self.encoder = None;
        self.device = None;
        self.queue = None;
        self.pool = None;
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

// ═══════════════════════════════════════════════════════════════════
// 双 Encoder 流水线：大批量（>4096）时交替使用两个 encoder 隐藏延迟
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "dual-encoder")]
/// 双 Encoder 批量提交器。
///
/// 大批量场景（> batch_threshold）下交替使用两个 CommandEncoder：
/// 当 encoder A 编码满后立即提交到 GPU，CPU 继续编码到 encoder B，
/// 从而重叠 GPU 执行和 CPU 编码，减少总等待时间。
///
/// 小批量场景退化为与 `GpuBatchSubmitter` 相同的单 encoder 行为。
pub struct DualEncoderSubmitter {
    /// encoder A 和其 pending jobs
    encoder_a: Option<CommandEncoder>,
    pending_a: Vec<PendingJob>,
    /// encoder B 和其 pending jobs
    encoder_b: Option<CommandEncoder>,
    pending_b: Vec<PendingJob>,
    /// 当前活跃 encoder（false = A, true = B）
    use_b: bool,
    /// 切换 encoder 的 job 数量阈值
    batch_threshold: usize,
    /// 通用状态
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    pool: Option<Arc<BufferPool>>,
}

#[cfg(feature = "dual-encoder")]
impl DualEncoderSubmitter {
    /// 创建双 encoder 提交器，`batch_threshold` 推荐值 4096。
    pub fn new(batch_threshold: usize) -> Self {
        Self {
            encoder_a: None,
            pending_a: Vec::new(),
            encoder_b: None,
            pending_b: Vec::new(),
            use_b: false,
            batch_threshold,
            device: None,
            queue: None,
            pool: None,
        }
    }

    /// 提交一个计算任务到批量队列。
    ///
    /// 当当前 encoder 的 pending 数量达到 `batch_threshold` 时，
    /// 立即提交该 encoder 到 GPU 并切换到另一个 encoder。
    pub fn submit(&mut self, ctx: &crate::context::GpuContext, job: BatchJob) -> Result<(), GpuError> {
        let device = ctx.device()?.clone();
        let queue = ctx.queue()?.clone();

        if self.device.is_none() {
            self.device = Some(device.clone());
            self.queue = Some(queue.clone());
            self.pool = Some(ctx.buffer_pool_arc());
        }

        // 检查是否需要切换 encoder
        let current_pending = if self.use_b { &self.pending_b } else { &self.pending_a };
        if current_pending.len() >= self.batch_threshold {
            self.flush_current_encoder(&device, &queue)?;
        }

        // 获取或创建当前 encoder
        let (encoder, pending) = self.current_encoder_mut(&device);

        // 编码 dispatch
        let buffer_refs: Vec<&GpuBuffer> = job.buffers.iter().collect();
        assert!(
            buffer_refs.len() == job.pipeline.bindings().len(),
            "BatchJob buffers 数量 ({}) 与管线 bindings 数量 ({}) 不匹配",
            buffer_refs.len(),
            job.pipeline.bindings().len()
        );

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

            pending.push(PendingJob {
                staging_buffer: staging,
            });
        }

        Ok(())
    }

    /// 提交当前活跃 encoder 到 GPU（不等待完成）。
    fn flush_current_encoder(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), GpuError> {
        if self.use_b {
            if let Some(encoder) = self.encoder_b.take() {
                queue.submit(std::iter::once(encoder.finish()));
            }
        } else {
            if let Some(encoder) = self.encoder_a.take() {
                queue.submit(std::iter::once(encoder.finish()));
            }
        }
        // 切换到另一个 encoder
        self.use_b = !self.use_b;
        Ok(())
    }

    /// 获取当前活跃 encoder 和 pending 列表的可变引用。
    fn current_encoder_mut(&mut self, device: &wgpu::Device) -> (&mut CommandEncoder, &mut Vec<PendingJob>) {
        if self.use_b {
            if self.encoder_b.is_none() {
                self.encoder_b = Some(device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("dual_batch_encoder_b"),
                    },
                ));
            }
            (self.encoder_b.as_mut().unwrap(), &mut self.pending_b)
        } else {
            if self.encoder_a.is_none() {
                self.encoder_a = Some(device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor {
                        label: Some("dual_batch_encoder_a"),
                    },
                ));
            }
            (self.encoder_a.as_mut().unwrap(), &mut self.pending_a)
        }
    }

    /// 统一提交所有剩余命令并等待所有已提交任务完成，返回结果数据。
    pub fn wait_all(&mut self) -> Result<Vec<Vec<u8>>, GpuError> {
        let total_pending = self.pending_a.len() + self.pending_b.len();
        if total_pending == 0 {
            return Ok(vec![]);
        }

        let device = self.device.as_ref()
            .ok_or(GpuError::Internal("DualEncoderSubmitter: device 未初始化".into()))?;
        let queue = self.queue.as_ref()
            .ok_or(GpuError::Internal("DualEncoderSubmitter: queue 未初始化".into()))?;

        // 提交两个 encoder 中剩余的命令
        if let Some(encoder) = self.encoder_a.take() {
            queue.submit(std::iter::once(encoder.finish()));
        }
        if let Some(encoder) = self.encoder_b.take() {
            queue.submit(std::iter::once(encoder.finish()));
        }

        // 合并两个 pending 列表
        let all_pending: Vec<PendingJob> = self.pending_a.drain(..)
            .chain(self.pending_b.drain(..))
            .collect();

        // 提交后立即注册所有 map_async 回调，然后单次 poll
        let mut map_rxs: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(all_pending.len());

        for pending in &all_pending {
            let (tx, rx) = std::sync::mpsc::channel();
            pending.staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).ok();
            });
            map_rxs.push(rx);
        }

        // 单次 poll：同时等待 GPU 执行完成和 staging buffer 映射就绪
        device.poll(wgpu::Maintain::Wait);
        crate::poll_counter::increment();

        // 验证所有映射成功
        let mut results = Vec::with_capacity(all_pending.len());
        let mut map_succeeded = Vec::with_capacity(all_pending.len());
        let mut first_error: Option<GpuError> = None;

        for rx in &map_rxs {
            let succeeded = match rx.recv() {
                Ok(Ok(())) => true,
                Ok(Err(ref e)) if first_error.is_none() => {
                    first_error = Some(GpuError::MapFailed(format!("双encoder批量映射失败: {}", e)));
                    false
                }
                Ok(Err(_)) => false,
                Err(_) if first_error.is_none() => {
                    first_error = Some(GpuError::MapFailed("双encoder批量映射通道关闭".into()));
                    false
                }
                Err(_) => false,
            };
            map_succeeded.push(succeeded);
        }

        if let Some(e) = first_error {
            for (i, pending) in all_pending.iter().enumerate() {
                if map_succeeded[i] {
                    pending.staging_buffer.unmap();
                }
            }
            if let Some(pool) = self.pool.as_ref() {
                for pending in all_pending {
                    pool.release_staging(pending.staging_buffer);
                }
            }
            self.device = None;
            self.queue = None;
            self.pool = None;
            return Err(e);
        }

        if let Some(pool) = self.pool.as_ref() {
            for pending in all_pending {
                let view = pending.staging_buffer.slice(..).get_mapped_range();
                let data = view.to_vec();
                drop(view);
                pending.staging_buffer.unmap();
                pool.release_staging(pending.staging_buffer);
                results.push(data);
            }
        }

        self.device = None;
        self.queue = None;
        self.pool = None;

        Ok(results)
    }

    /// 返回当前两个 encoder 的 pending 任务总数。
    pub fn pending_count(&self) -> usize {
        self.pending_a.len() + self.pending_b.len()
    }

    /// 清空所有待处理任务。
    pub fn clear(&mut self) {
        if let Some(pool) = self.pool.as_ref() {
            for pending in self.pending_a.drain(..) {
                pool.release_staging(pending.staging_buffer);
            }
            for pending in self.pending_b.drain(..) {
                pool.release_staging(pending.staging_buffer);
            }
        }
        self.encoder_a = None;
        self.encoder_b = None;
        self.use_b = false;
        self.device = None;
        self.queue = None;
        self.pool = None;
    }
}

#[cfg(feature = "dual-encoder")]
impl Default for DualEncoderSubmitter {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[cfg(feature = "dual-encoder")]
impl Drop for DualEncoderSubmitter {
    fn drop(&mut self) {
        self.clear();
    }
}