use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::buffer_pool::{BufferPool, BufferPoolConfig};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor};
#[cfg(feature = "cpu-fallback")]
use crate::tasks::sha256_cpu::Sha256Cpu;
use crate::ComputeBackend;

const SHA256_WGSL: &str = include_str!("sha256.wgsl");
const SHA256_CHAINED_WGSL: &str = include_str!("sha256_chained.wgsl");

/// SHA-256 计算器的默认 workgroup 大小。
pub const SHA256_DEFAULT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Sha256Params {
    message_count: u32,
    block_mode: u32,
    _padding: [u32; 2],
}

const BLOCK_SIZE: usize = 64;
const BLOCK_U32_COUNT: usize = 16;
const HASH_U32_COUNT: usize = 8;
const MAX_SINGLE_BLOCK_MSG_LEN: usize = 55;

/// SHA-256 并行哈希计算器，基于 GPU compute shader 实现。
///
/// 支持批量处理任意数量的消息，每条消息长度不超过 55 字节时使用单 block 模式，
/// 更长的消息自动切换为多 block 流水线模式。
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let sha256 = Sha256Computer::new(&mut ctx).unwrap();
///
/// let messages = vec![
///     b"hello".to_vec(),
///     b"world".to_vec(),
/// ];
/// let hashes = sha256.compute(&ctx, &messages).unwrap();
/// ```
pub struct Sha256Computer {
    pipeline: Arc<ComputePipeline>,
    pipeline_chained: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    buffer_pool: BufferPool,
    cached_single_block_params: RefCell<HashMap<u32, GpuBuffer>>,
}

/// SHA-256 异步批量提交器，真批量提交实现。
///
/// 所有单 block dispatch + copy 命令编码到同一个 CommandEncoder，
/// 最终通过一次 `queue.submit()` 统一提交。
/// 多 block 消息由于数据依赖仍需同步等待中间结果。
///
/// # 使用方式
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let sha256 = Sha256Computer::new(&mut ctx).unwrap();
/// let mut submitter = sha256.batch_submit(&ctx);
/// // ... 添加任务
/// let hashes = submitter.finish().unwrap();
/// ```
pub struct Sha256BatchSubmitter<'a> {
    computer: &'a Sha256Computer,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: Option<wgpu::CommandEncoder>,
    pending_batches: Vec<PendingBatch>,
}

struct PendingBatch {
    message_count: usize,
    original_indices: Vec<usize>,
    output_buffer: GpuBuffer,
    staging_buffer: wgpu::Buffer,
    is_single_block: bool,
    /// 延迟释放的 input buffer（raw wgpu::Buffer），在 wait_all 提交后才释放
    input_buffers_to_release: Vec<wgpu::Buffer>,
}

impl Sha256Computer {
    /// 创建 SHA-256 计算器，使用默认 workgroup_size [256, 1, 1]。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_workgroup_size(ctx, SHA256_DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建 SHA-256 计算器，指定 workgroup_size。
    pub fn with_workgroup_size(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        Self::with_config(ctx, workgroup_size, BufferPool::new())
    }

    /// 创建 SHA-256 计算器，指定 workgroup_size 和缓冲区池配置。
    ///
    /// 当需要与其他组件共享缓冲区池或精细控制池行为时使用此构造方法。
    pub fn with_buffer_pool_config(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
        pool_config: BufferPoolConfig,
    ) -> Result<Self, GpuError> {
        Self::with_config(ctx, workgroup_size, BufferPool::with_config(pool_config))
    }

    /// 创建 SHA-256 计算器，完全自定义配置。
    fn with_config(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
        buffer_pool: BufferPool,
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor::default_3_binding(SHA256_WGSL, workgroup_size),
        )?;
        let pipeline_chained = ctx.get_or_create_pipeline(
            &PipelineDescriptor::default_3_binding(SHA256_CHAINED_WGSL, workgroup_size),
        )?;
        Ok(Self {
            pipeline,
            pipeline_chained,
            workgroup_size,
            buffer_pool,
            cached_single_block_params: RefCell::new(HashMap::new()),
        })
    }

    /// 批量计算 SHA-256 哈希（同步接口）。
    ///
    /// 当 `GpuContext` 处于 CPU 降级模式时，自动委托到 [`Sha256Cpu`] 计算。
    pub fn compute(
        &self,
        ctx: &GpuContext,
        messages: &[Vec<u8>],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
        if ctx.backend() == ComputeBackend::Cpu {
            #[cfg(feature = "cpu-fallback")]
            {
                let cpu = Sha256Cpu::new();
                return cpu.compute(messages);
            }
            #[cfg(not(feature = "cpu-fallback"))]
            return Err(GpuError::CpuFallback("CPU 降级未启用".to_string()));
        }

        if messages.is_empty() {
            return Ok(vec![]);
        }

        let device = ctx.device();
        let queue = ctx.queue();

        let mut single_block_indices: Vec<usize> = Vec::new();
        let mut multi_block_indices: Vec<usize> = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            if msg.len() <= MAX_SINGLE_BLOCK_MSG_LEN {
                single_block_indices.push(i);
            } else {
                multi_block_indices.push(i);
            }
        }

        let mut results: Vec<Option<[u8; 32]>> = vec![None; messages.len()];

        if !single_block_indices.is_empty() {
            let batch_messages: Vec<&Vec<u8>> =
                single_block_indices.iter().map(|&i| &messages[i]).collect();
            let batch_hashes = self.compute_single_block_batch(device, queue, &batch_messages)?;
            for (batch_i, &orig_i) in single_block_indices.iter().enumerate() {
                results[orig_i] = Some(batch_hashes[batch_i]);
            }
        }

        if !multi_block_indices.is_empty() {
            let multi_messages: Vec<&Vec<u8>> =
                multi_block_indices.iter().map(|&i| &messages[i]).collect();
            let multi_hashes = self.compute_multi_block_batch(device, queue, &multi_messages)?;
            for (batch_i, &orig_i) in multi_block_indices.iter().enumerate() {
                results[orig_i] = Some(multi_hashes[batch_i]);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// 创建异步批量提交器。
    ///
    /// 通过此提交器可以多次 `submit()` 后统一 `wait_all()`，
    /// 将多次 CPU-GPU 同步合并为一次，显著降低调度开销。
    pub fn batch_submitter<'a>(&'a self, ctx: &'a GpuContext) -> Sha256BatchSubmitter<'a> {
        Sha256BatchSubmitter {
            computer: self,
            device: ctx.device(),
            queue: ctx.queue(),
            encoder: None,
            pending_batches: Vec::new(),
        }
    }

    /// 计算单 block 消息批量（≤55 字节）。
    fn compute_single_block_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        messages: &[&Vec<u8>],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
        let count = messages.len();
        let mut all_words = Vec::with_capacity(count * BLOCK_U32_COUNT);

        for msg in messages {
            let padded = pad_single_block(msg);
            let words = bytes_to_be_u32(&padded);
            all_words.extend_from_slice(&words);
        }

        let input_size = (all_words.len() * 4) as u64;
        let output_size = (count * HASH_U32_COUNT * 4) as u64;

        let input_buffer_raw = self.buffer_pool.acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self.buffer_pool.acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_words));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params_buffer = self.get_or_create_single_block_params(device, count as u32);

        let dispatch_x = (count as u32).div_ceil(self.workgroup_size[0]).max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, 1, 1],
        );

        let result = output_buffer.download(device, queue)?;

        self.buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
        self.buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

        let mut hashes = Vec::with_capacity(count);
        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        for i in 0..count {
            let mut hash = [0u8; 32];
            for j in 0..HASH_U32_COUNT {
                let w = raw_u32[i * HASH_U32_COUNT + j];
                hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_be_bytes());
            }
            hashes.push(hash);
        }

        Ok(hashes)
    }

    /// 获取或创建 single block 场景的 params buffer（按 message_count 缓存）。
    fn get_or_create_single_block_params(
        &self,
        device: &wgpu::Device,
        message_count: u32,
    ) -> GpuBuffer {
        let mut cache = self.cached_single_block_params.borrow_mut();
        if let Some(buf) = cache.get(&message_count) {
            buf.clone()
        } else {
            let params = Sha256Params {
                message_count,
                block_mode: 0,
                _padding: [0; 2],
            };
            let buf = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);
            cache.insert(message_count, buf.clone());
            buf
        }
    }

    /// 计算多 block 消息批量（>55 字节）—— 链式着色器，单 dispatch 处理全部 block。
    fn compute_multi_block_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        messages: &[&Vec<u8>],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
        let count = messages.len();

        // 链式着色器输入：[block_counts...][all_blocks...]
        let mut block_counts = Vec::with_capacity(count);
        let mut all_block_words = Vec::new();
        for msg in messages {
            let blocks = split_into_blocks(msg);
            block_counts.push(blocks.len() as u32);
            for block in &blocks {
                all_block_words.extend_from_slice(&bytes_to_be_u32(block));
            }
        }

        let total_u32 = count + all_block_words.len();
        let input_size = (total_u32 * 4) as u64;
        let output_size = (count * HASH_U32_COUNT * 4) as u64;

        let input_buffer_raw = self.buffer_pool.acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self.buffer_pool.acquire(device, output_size, BufferUsage::Storage);

        let mut input_words = Vec::with_capacity(total_u32);
        input_words.extend_from_slice(&block_counts);
        input_words.extend_from_slice(&all_block_words);
        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&input_words));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = Sha256Params { message_count: count as u32, block_mode: 0, _padding: [0; 2] };
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

        let dispatch_x = (count as u32).div_ceil(self.workgroup_size[0]).max(1);
        self.pipeline_chained.dispatch(
            device, queue, &[&input_buffer, &output_buffer, &params_buffer], [dispatch_x, 1, 1],
        );

        let result = output_buffer.download(device, queue)?;
        self.buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
        self.buffer_pool.release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        let mut hashes = Vec::with_capacity(count);
        for i in 0..count {
            let mut hash = [0u8; 32];
            for j in 0..HASH_U32_COUNT {
                let w = raw_u32[i * HASH_U32_COUNT + j];
                hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_be_bytes());
            }
            hashes.push(hash);
        }
        Ok(hashes)
    }
}

impl<'a> Sha256BatchSubmitter<'a> {
    /// 提交一批消息到异步队列。
    pub fn submit(&mut self, messages: &[Vec<u8>]) -> Result<(), GpuError> {
        if messages.is_empty() {
            return Ok(());
        }

        let mut single_block_indices: Vec<usize> = Vec::new();
        let mut multi_block_indices: Vec<usize> = Vec::new();

        for (i, msg) in messages.iter().enumerate() {
            if msg.len() <= MAX_SINGLE_BLOCK_MSG_LEN {
                single_block_indices.push(i);
            } else {
                multi_block_indices.push(i);
            }
        }

        if !single_block_indices.is_empty() {
            let batch_messages: Vec<&Vec<u8>> =
                single_block_indices.iter().map(|&i| &messages[i]).collect();
            self.submit_single_block_batch(&batch_messages, single_block_indices)?;
        }

        for &orig_i in &multi_block_indices {
            self.submit_multi_block_message(&messages[orig_i], orig_i)?;
        }

        Ok(())
    }

    /// 统一等待所有已提交批次完成，返回结果。
    pub fn wait_all(&mut self) -> Result<Vec<(usize, [u8; 32])>, GpuError> {
        if self.pending_batches.is_empty() {
            return Ok(vec![]);
        }

        // 提交累积的 encoder（如果有）
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        self.device.poll(wgpu::Maintain::Wait);

        let mut results = Vec::with_capacity(
            self.pending_batches.iter().map(|b| b.message_count).sum(),
        );

        let mut map_receivers: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>> =
            Vec::with_capacity(self.pending_batches.len());
        for pending in &self.pending_batches {
            let (send_map, recv_map) = std::sync::mpsc::channel();
            pending
                .staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    send_map.send(result).ok();
                });
            map_receivers.push(recv_map);
        }

        self.device.poll(wgpu::Maintain::Wait);

        // 验证所有 staging buffer 映射成功
        for recv in &map_receivers {
            match recv
                .recv()
                .map_err(|_| GpuError::MapFailed("批量映射通道关闭，GPU 设备可能已丢失".into()))?
            {
                Ok(()) => {}
                Err(e) => return Err(GpuError::MapFailed(format!("批量 staging buffer 映射失败: {}", e))),
            }
        }

        for pending in self.pending_batches.drain(..) {
            let view = pending.staging_buffer.slice(..).get_mapped_range();
            let data = view.to_vec();
            drop(view);
            pending.staging_buffer.unmap();

            let raw_u32 = bytemuck::cast_slice::<u8, u32>(&data);
            if pending.is_single_block {
                for (batch_i, &orig_i) in pending.original_indices.iter().enumerate() {
                    let mut hash = [0u8; 32];
                    for j in 0..HASH_U32_COUNT {
                        let w = raw_u32[batch_i * HASH_U32_COUNT + j];
                        hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_be_bytes());
                    }
                    results.push((orig_i, hash));
                }
            } else {
                let mut hash = [0u8; 32];
                for j in 0..HASH_U32_COUNT {
                    let w = raw_u32[j];
                    hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_be_bytes());
                }
                results.push((pending.original_indices[0], hash));
            }

            self.computer.buffer_pool.release(pending.output_buffer.into_raw(), BufferUsage::Storage);
            // 释放延迟持有的 input buffers
            for input_buf in pending.input_buffers_to_release {
                self.computer.buffer_pool.release(input_buf, BufferUsage::Storage);
            }
        }

        Ok(results)
    }

    /// 返回当前待处理批次数量。
    pub fn pending_count(&self) -> usize {
        self.pending_batches.len()
    }

    /// 提交单 block 批次——使用 encode_dispatch_into 编码到共享 encoder。
    fn submit_single_block_batch(
        &mut self,
        messages: &[&Vec<u8>],
        original_indices: Vec<usize>,
    ) -> Result<(), GpuError> {
        let count = messages.len();
        let mut all_words = Vec::with_capacity(count * BLOCK_U32_COUNT);

        for msg in messages {
            let padded = pad_single_block(msg);
            let words = bytes_to_be_u32(&padded);
            all_words.extend_from_slice(&words);
        }

        let input_size = (all_words.len() * 4) as u64;
        let output_size = (count * HASH_U32_COUNT * 4) as u64;

        let input_buffer_raw = self
            .computer
            .buffer_pool
            .acquire(self.device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .computer
            .buffer_pool
            .acquire(self.device, output_size, BufferUsage::Storage);

        self.queue
            .write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_words));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params_buffer = self
            .computer
            .get_or_create_single_block_params(self.device, count as u32);

        let dispatch_x = (count as u32)
            .div_ceil(self.computer.workgroup_size[0])
            .max(1);

        // 惰性创建共享 encoder
        if self.encoder.is_none() {
            self.encoder = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("sha256_batch_encoder"),
                },
            ));
        }
        let encoder = self.encoder.as_mut().unwrap();

        // 编码 dispatch 到共享 encoder（不提交）
        self.computer.pipeline.encode_dispatch_into(
            self.device,
            encoder,
            &[&input_buffer, &output_buffer, &params_buffer],
            [dispatch_x, 1, 1],
        );

        // 编码 copy 到 staging buffer
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sha256_batch_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(output_buffer.raw(), 0, &staging, 0, output_size);

        // 延迟释放 input buffer，防止在 encoder 提交前被 pool 复用
        let input_raw = input_buffer.into_raw();

        self.pending_batches.push(PendingBatch {
            message_count: count,
            original_indices,
            output_buffer,
            staging_buffer: staging,
            is_single_block: true,
            input_buffers_to_release: vec![input_raw],
        });

        Ok(())
    }

    /// 提交多 block 消息——链式着色器，单 dispatch 不中断异步流水线。
    fn submit_multi_block_message(
        &mut self,
        message: &[u8],
        original_index: usize,
    ) -> Result<(), GpuError> {
        let blocks = split_into_blocks(message);
        let block_count = blocks.len() as u32;

        let mut input_words = Vec::with_capacity(1 + block_count as usize * BLOCK_U32_COUNT);
        input_words.push(block_count);
        for block in &blocks {
            input_words.extend_from_slice(&bytes_to_be_u32(block));
        }

        let input_size = (input_words.len() * 4) as u64;
        let output_size = (HASH_U32_COUNT * 4) as u64;

        let input_buf_raw = self.computer.buffer_pool.acquire(self.device, input_size, BufferUsage::Storage);
        let output_buf_raw = self.computer.buffer_pool.acquire(self.device, output_size, BufferUsage::Storage);
        self.queue.write_buffer(&input_buf_raw, 0, bytemuck::cast_slice(&input_words));
        let input_gpu = GpuBuffer::from_raw(input_buf_raw.clone(), input_size);
        let output_gpu = GpuBuffer::from_raw(output_buf_raw, output_size);

        let params = Sha256Params { message_count: 1, block_mode: 0, _padding: [0; 2] };
        let params_buf = GpuBuffer::from_data(self.device, &[params], BufferUsage::Uniform);

        if self.encoder.is_none() {
            self.encoder = Some(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("sha256_batch_encoder") },
            ));
        }
        let encoder = self.encoder.as_mut().unwrap();

        self.computer.pipeline_chained.encode_dispatch_into(
            self.device, encoder, &[&input_gpu, &output_gpu, &params_buf], [1, 1, 1],
        );

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sha256_chained_stg"), size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(output_gpu.raw(), 0, &staging, 0, output_size);

        // 延迟释放 input buffer，防止在 encoder 提交前被 pool 复用
        self.pending_batches.push(PendingBatch {
            message_count: 1, original_indices: vec![original_index],
            output_buffer: output_gpu, staging_buffer: staging, is_single_block: false,
            input_buffers_to_release: vec![input_buf_raw],
        });
        Ok(())
    }
}

fn bytes_to_be_u32(data: &[u8]) -> Vec<u32> {
    let count = data.len() / 4;
    let mut words = Vec::with_capacity(count);
    for chunk in data.chunks_exact(4) {
        words.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    words
}

fn pad_single_block(message: &[u8]) -> Vec<u8> {
    let msg_len = message.len();
    let mut padded = Vec::with_capacity(BLOCK_SIZE);
    padded.extend_from_slice(message);
    padded.push(0x80);

    while padded.len() < 56 {
        padded.push(0x00);
    }

    let bit_len = (msg_len as u64) * 8;
    padded.extend_from_slice(&bit_len.to_be_bytes());

    debug_assert_eq!(padded.len(), BLOCK_SIZE);
    padded
}

fn split_into_blocks(message: &[u8]) -> Vec<Vec<u8>> {
    let msg_len = message.len();
    let mut blocks = Vec::new();

    let full_blocks = msg_len / BLOCK_SIZE;
    for i in 0..full_blocks {
        blocks.push(message[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE].to_vec());
    }

    let remainder_start = full_blocks * BLOCK_SIZE;
    let remainder = &message[remainder_start..];

    let mut last_block = Vec::with_capacity(BLOCK_SIZE);
    last_block.extend_from_slice(remainder);
    last_block.push(0x80);

    if last_block.len() <= 56 {
        while last_block.len() < 56 {
            last_block.push(0x00);
        }
        let bit_len = (msg_len as u64) * 8;
        last_block.extend_from_slice(&bit_len.to_be_bytes());
        blocks.push(last_block);
    } else {
        while last_block.len() < BLOCK_SIZE {
            last_block.push(0x00);
        }
        blocks.push(last_block);

        let mut extra_block = vec![0u8; 56];
        let bit_len = (msg_len as u64) * 8;
        extra_block.extend_from_slice(&bit_len.to_be_bytes());
        blocks.push(extra_block);
    }

    blocks
}
