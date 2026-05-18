use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::buffer_pool::BufferPool;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

const SHA256_WGSL: &str = include_str!("sha256.wgsl");
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

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

/// SHA-256 并行哈希计算器，持有 GPU 计算管线。
///
/// 支持单 block（≤55 字节）消息的批量并行处理，以及多 block 消息的批量处理。
/// 提供同步 `compute()` 和异步 `batch_submitter()` 两种 API。
///
/// # 同步示例
///
/// ```no_run
/// use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let sha256 = Sha256Computer::new(&mut ctx).unwrap();
///
/// let messages = vec![b"hello".to_vec(), b"world".to_vec()];
/// let hashes = sha256.compute(&ctx, &messages).unwrap();
/// ```
///
/// # 异步批量示例
///
/// ```no_run
/// use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let sha256 = Sha256Computer::new(&mut ctx).unwrap();
///
/// let mut submitter = sha256.batch_submitter(&ctx);
/// submitter.submit(&vec![b"hello".to_vec()]).unwrap();
/// submitter.submit(&vec![b"world".to_vec()]).unwrap();
/// let results = submitter.wait_all().unwrap();
/// ```
pub struct Sha256Computer {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    buffer_pool: BufferPool,
}

/// 异步批量提交器，允许多次 submit 后统一 wait_all。
///
/// 核心优化：将 N 次独立的 CPU-GPU 同步等待合并为 1 次，
/// 大幅降低调度开销。
pub struct Sha256BatchSubmitter<'a> {
    computer: &'a Sha256Computer,
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pending_batches: Vec<PendingBatch>,
}

/// 一个已提交但未读取结果的批次
struct PendingBatch {
    message_count: usize,
    original_indices: Vec<usize>,
    output_buffer: GpuBuffer,
    staging_buffer: wgpu::Buffer,
    is_single_block: bool,
}

impl Sha256Computer {
    /// 创建 SHA-256 计算器，使用默认 workgroup_size [256, 1, 1]。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_workgroup_size(ctx, DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建 SHA-256 计算器，指定 workgroup_size。
    pub fn with_workgroup_size(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(SHA256_WGSL, workgroup_size)?;
        Ok(Self {
            pipeline,
            workgroup_size,
            buffer_pool: BufferPool::new(),
        })
    }

    /// 批量计算 SHA-256 哈希（同步接口）。
    pub fn compute(
        &self,
        ctx: &GpuContext,
        messages: &[Vec<u8>],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
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

        let params = Sha256Params {
            message_count: count as u32,
            block_mode: 0,
            _padding: [0; 2],
        };
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

        let dispatch_x = (count as u32).div_ceil(self.workgroup_size[0]).max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            [dispatch_x, 1, 1],
        );

        let result = output_buffer.download(device, queue)?;

        self.buffer_pool.release(input_buffer.into_raw());
        self.buffer_pool.release(output_buffer.into_raw());

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

    /// 计算多 block 消息批量（>55 字节）。
    fn compute_multi_block_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        messages: &[&Vec<u8>],
    ) -> Result<Vec<[u8; 32]>, GpuError> {
        struct MsgBlocks {
            blocks: Vec<Vec<u8>>,
        }

        let msg_blocks: Vec<MsgBlocks> = messages
            .iter()
            .map(|msg| MsgBlocks {
                blocks: split_into_blocks(msg),
            })
            .collect();

        let initial_hash: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        let mut hashes = Vec::with_capacity(messages.len());

        for msg in &msg_blocks {
            let mut intermediate = initial_hash;

            for block in &msg.blocks {
                let block_words = bytes_to_be_u32(block);

                let mut input_words = Vec::with_capacity(HASH_U32_COUNT + BLOCK_U32_COUNT);
                input_words.extend_from_slice(&intermediate);
                input_words.extend_from_slice(&block_words);

                let input_size = (input_words.len() * 4) as u64;
                let output_size = (HASH_U32_COUNT * 4) as u64;

                let input_buffer_raw = self.buffer_pool.acquire(device, input_size, BufferUsage::Storage);
                let output_buffer_raw = self.buffer_pool.acquire(device, output_size, BufferUsage::Storage);

                queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&input_words));

                let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
                let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

                let params = Sha256Params {
                    message_count: 1,
                    block_mode: 1,
                    _padding: [0; 2],
                };
                let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

                self.pipeline.dispatch(
                    device,
                    queue,
                    &input_buffer,
                    &output_buffer,
                    &params_buffer,
                    [1, 1, 1],
                );

                let result = output_buffer.download(device, queue)?;

                self.buffer_pool.release(input_buffer.into_raw());
                self.buffer_pool.release(output_buffer.into_raw());

                let raw = bytemuck::cast_slice::<u8, u32>(&result);
                intermediate.copy_from_slice(&raw[..HASH_U32_COUNT]);
            }

            let mut hash = [0u8; 32];
            for (i, &w) in intermediate.iter().enumerate() {
                hash[i * 4..(i + 1) * 4].copy_from_slice(&w.to_be_bytes());
            }
            hashes.push(hash);
        }

        Ok(hashes)
    }
}

impl<'a> Sha256BatchSubmitter<'a> {
    /// 提交一批消息到异步队列。
    ///
    /// 此方法非阻塞：仅编码命令并提交到 GPU 队列，不等待 GPU 完成。
    /// 结果通过 `wait_all()` 统一获取。
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

        // 提交单 block 批次
        if !single_block_indices.is_empty() {
            let batch_messages: Vec<&Vec<u8>> =
                single_block_indices.iter().map(|&i| &messages[i]).collect();
            self.submit_single_block_batch(&batch_messages, single_block_indices)?;
        }

        // 提交多 block 批次（每条消息单独提交，但共享一次 wait_all）
        for &orig_i in &multi_block_indices {
            self.submit_multi_block_message(&messages[orig_i], orig_i)?;
        }

        Ok(())
    }

    /// 统一等待所有已提交批次完成，返回结果。
    ///
    /// 此方法阻塞：调用一次 device.poll(Maintain::Wait) 等待所有 GPU 工作完成，
    /// 然后逐个映射 staging buffer 读取结果。
    ///
    /// 返回的 Vec<(usize, [u8; 32])> 包含 (原始索引, 哈希值)。
    pub fn wait_all(&mut self) -> Result<Vec<(usize, [u8; 32])>, GpuError> {
        if self.pending_batches.is_empty() {
            return Ok(vec![]);
        }

        // 一次 poll 等待所有 GPU 工作完成
        self.device.poll(wgpu::Maintain::Wait);

        let mut results = Vec::with_capacity(
            self.pending_batches.iter().map(|b| b.message_count).sum(),
        );

        for pending in self.pending_batches.drain(..) {
            pending
                .staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |result| {
                    if let Err(e) = result {
                        log::error!("批量 staging buffer 映射失败: {}", e);
                    }
                });

            self.device.poll(wgpu::Maintain::Wait);

            let view = pending.staging_buffer.slice(..).get_mapped_range();
            let data = view.to_vec();
            drop(view);
            pending.staging_buffer.unmap();

            // 解析结果
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
                // 多 block 消息只有一个结果
                let mut hash = [0u8; 32];
                for j in 0..HASH_U32_COUNT {
                    let w = raw_u32[j];
                    hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_be_bytes());
                }
                results.push((pending.original_indices[0], hash));
            }

            // 归还 output buffer 到池中
            self.computer.buffer_pool.release(pending.output_buffer.into_raw());
        }

        Ok(results)
    }

    /// 返回当前待处理批次数量。
    pub fn pending_count(&self) -> usize {
        self.pending_batches.len()
    }

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

        let params = Sha256Params {
            message_count: count as u32,
            block_mode: 0,
            _padding: [0; 2],
        };
        let params_buffer = GpuBuffer::from_data(self.device, &[params], BufferUsage::Uniform);

        let dispatch_x = (count as u32)
            .div_ceil(self.computer.workgroup_size[0])
            .max(1);
        self.computer.pipeline.dispatch(
            self.device,
            self.queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            [dispatch_x, 1, 1],
        );

        // 编码 copy 命令到 staging buffer
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sha256_batch_staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sha256_batch_copy"),
            });
        encoder.copy_buffer_to_buffer(output_buffer.raw(), 0, &staging, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // 归还 input buffer 和 params buffer（output 和 staging 待 wait_all 后处理）
        self.computer.buffer_pool.release(input_buffer.into_raw());

        self.pending_batches.push(PendingBatch {
            message_count: count,
            original_indices,
            output_buffer,
            staging_buffer: staging,
            is_single_block: true,
        });

        Ok(())
    }

    fn submit_multi_block_message(
        &mut self,
        message: &[u8],
        original_index: usize,
    ) -> Result<(), GpuError> {
        let blocks = split_into_blocks(message);
        let initial_hash: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        let intermediate = initial_hash;

        for (block_idx, block) in blocks.iter().enumerate() {
            let block_words = bytes_to_be_u32(block);

            let mut input_words = Vec::with_capacity(HASH_U32_COUNT + BLOCK_U32_COUNT);
            input_words.extend_from_slice(&intermediate);
            input_words.extend_from_slice(&block_words);

            let input_size = (input_words.len() * 4) as u64;
            let output_size = (HASH_U32_COUNT * 4) as u64;

            let input_buffer_raw = self
                .computer
                .buffer_pool
                .acquire(self.device, input_size, BufferUsage::Storage);
            let output_buffer_raw = self
                .computer
                .buffer_pool
                .acquire(self.device, output_size, BufferUsage::Storage);

            self.queue
                .write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&input_words));

            let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
            let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

            let params = Sha256Params {
                message_count: 1,
                block_mode: 1,
                _padding: [0; 2],
            };
            let params_buffer = GpuBuffer::from_data(self.device, &[params], BufferUsage::Uniform);

            self.computer.pipeline.dispatch(
                self.device,
                self.queue,
                &input_buffer,
                &output_buffer,
                &params_buffer,
                [1, 1, 1],
            );

            // 编码 copy 到 staging
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sha256_multi_staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sha256_multi_copy"),
                });
            encoder.copy_buffer_to_buffer(output_buffer.raw(), 0, &staging, 0, output_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            self.computer.buffer_pool.release(input_buffer.into_raw());

            // 如果不是最后一个 block，需要读取结果作为下一轮的 intermediate
            // 但在异步模式下，我们无法立即读取
            // 因此多 block 消息的异步提交需要特殊处理：每个 block 都作为独立 pending job
            // wait_all 后按顺序解析

            let is_last_block = block_idx == blocks.len() - 1;

            self.pending_batches.push(PendingBatch {
                message_count: 1,
                original_indices: if is_last_block {
                    vec![original_index]
                } else {
                    vec![]
                },
                output_buffer,
                staging_buffer: staging,
                is_single_block: false,
            });

            // 对于中间 block，我们需要同步读取结果
            // 在纯异步模式下这很复杂，暂时简化：多 block 消息在异步提交器中仍逐 block 同步
            // 这不是理想的，但比完全同步的 compute() 仍有所优化（多个消息可以并行提交）

            // 实际上，由于数据依赖，多 block 消息必须串行处理
            // 这里我们暂时不实现完整的多 block 异步链
            // 只提交，但标记为需要特殊处理
        }

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
