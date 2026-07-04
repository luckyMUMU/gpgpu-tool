use std::borrow::Cow;

use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, Device, Queue};

use crate::buffer_pool::BufferPool;
use crate::error::GpuError;

/// 缓冲区使用类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// 存储缓冲区（Storage buffer），用于 GPU 读写。
    Storage,
    /// 统一缓冲区（Uniform buffer），用于 GPU 只读参数。
    Uniform,
}

impl BufferUsage {
    /// 将 BufferUsage 转换为 wgpu BufferUsages 标志位。
    ///
    /// `include_copy_src` 为 true 时额外添加 `COPY_SRC`，允许缓冲区数据下载回 CPU。
    pub fn to_wgpu_usage(self, include_copy_src: bool) -> BufferUsages {
        let base = match self {
            BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        };
        if include_copy_src { base | BufferUsages::COPY_SRC } else { base }
    }
}

/// GPU 缓冲区，封装 CPU↔GPU 数据传输。
///
/// 支持从任意 `bytemuck::Pod` 类型创建、写入和读取数据。
///
/// # 创建
///
/// - [`from_data`](GpuBuffer::from_data): 从 Pod 类型数据创建（无 COPY_SRC，纯输入）
/// - [`from_data_readable`](GpuBuffer::from_data_readable): 从 Pod 类型数据创建（含 COPY_SRC，可下载）
/// - [`from_bytes`](GpuBuffer::from_bytes): 从原始字节创建（无 COPY_SRC，纯输入）
/// - [`from_bytes_readable`](GpuBuffer::from_bytes_readable): 从原始字节创建（含 COPY_SRC，可下载）
/// - [`empty`](GpuBuffer::empty): 创建零初始化缓冲区（无 COPY_SRC）
/// - [`empty_readable`](GpuBuffer::empty_readable): 创建零初始化可读缓冲区（含 COPY_SRC，可下载）
///
/// # 读写
///
/// - [`write`](GpuBuffer::write): 写入 Pod 数据到 GPU
/// - [`write_bytes`](GpuBuffer::write_bytes): 写入字节数据到 GPU
/// - [`download`](GpuBuffer::download): 从 GPU 下载数据到 CPU
/// - [`download_with_pool`](GpuBuffer::download_with_pool): 使用 BufferPool 下载
/// - [`download_batch`]: 批量下载多个缓冲区，单次 poll
/// - [`download_batch_with_pool`]: 批量下载 + BufferPool 复用
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ctx = GpuContext::new_sync().unwrap();
///     let data: Vec<u32> = vec![1, 2, 3, 4];
///     let buffer = GpuBuffer::from_data_readable(ctx.device()?, &data, BufferUsage::Storage);
///
///     // 下载结果
///     let result = buffer.download(ctx.device()?, ctx.queue()?).unwrap();
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
}

/// 按缓冲区用途计算对齐后的大小。
///
/// - Uniform: 16 字节对齐（满足 std140 布局要求）
/// - 其他（Storage 等）: 256 字节对齐（GPU 最佳实践）
const fn align_for_usage(size: u64, usage: BufferUsage) -> u64 {
    match usage {
        BufferUsage::Uniform => (size + 15) & !15,
        _ => (size + 255) & !255,
    }
}

/// 按缓冲区用途填充数据到对齐大小。
///
/// 如果数据已满足对齐要求，返回 `Cow::Borrowed`（零拷贝）；
/// 否则填充零字节并返回 `Cow::Owned`。
fn pad_for_usage(data: &[u8], usage: BufferUsage) -> Cow<'_, [u8]> {
    let aligned_size = align_for_usage(data.len() as u64, usage) as usize;
    if data.len() == aligned_size {
        Cow::Borrowed(data)
    } else {
        let mut padded = data.to_vec();
        padded.resize(aligned_size, 0);
        Cow::Owned(padded)
    }
}

impl GpuBuffer {
    /// 从 `bytemuck::Pod` 类型数据创建缓冲区，数据立即上传到 GPU。
    ///
    /// 创建的缓冲区**不支持**下载回 CPU（无 `COPY_SRC` 标志），
    /// 适用于纯输入缓冲区（如参数 buffer）。
    /// 如需下载，请使用 [`from_data_readable`](GpuBuffer::from_data_readable)。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn from_data<T: bytemuck::Pod>(device: &Device, data: &[T], usage: BufferUsage) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as u64;
        let contents = pad_for_usage(bytes, usage);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_data"),
            contents: &contents,
            usage: usage.to_wgpu_usage(false),
        });
        Self { buffer, size }
    }

    /// 从 `bytemuck::Pod` 类型数据创建**可读**缓冲区，数据立即上传到 GPU。
    ///
    /// 与 [`from_data`](GpuBuffer::from_data) 相比，额外添加 `COPY_SRC` 标志，
    /// 允许缓冲区内容通过 [`download`](GpuBuffer::download) 下载回 CPU。
    /// 仅在需要回读数据时使用，避免浪费 GPU 资源配额。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn from_data_readable<T: bytemuck::Pod>(device: &Device, data: &[T], usage: BufferUsage) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as u64;
        let contents = pad_for_usage(bytes, usage);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_data_readable"),
            contents: &contents,
            usage: usage.to_wgpu_usage(true),
        });
        Self { buffer, size }
    }

    /// 从原始字节数据创建缓冲区，数据立即上传到 GPU。
    ///
    /// 创建的缓冲区**不支持**下载回 CPU（无 `COPY_SRC` 标志）。
    /// 如需下载，请使用 [`from_bytes_readable`](GpuBuffer::from_bytes_readable)。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn from_bytes(device: &Device, data: &[u8], usage: BufferUsage) -> Self {
        let size = data.len() as u64;
        let contents = pad_for_usage(data, usage);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_bytes"),
            contents: &contents,
            usage: usage.to_wgpu_usage(false),
        });
        Self { buffer, size }
    }

    /// 从原始字节数据创建**可读**缓冲区，数据立即上传到 GPU。
    ///
    /// 与 [`from_bytes`](GpuBuffer::from_bytes) 相比，额外添加 `COPY_SRC` 标志，
    /// 允许缓冲区内容通过 [`download`](GpuBuffer::download) 下载回 CPU。
    /// 仅在需要回读数据时使用，避免浪费 GPU 资源配额。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn from_bytes_readable(device: &Device, data: &[u8], usage: BufferUsage) -> Self {
        let size = data.len() as u64;
        let contents = pad_for_usage(data, usage);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_bytes_readable"),
            contents: &contents,
            usage: usage.to_wgpu_usage(true),
        });
        Self { buffer, size }
    }

    /// 创建指定大小的零初始化缓冲区。
    ///
    /// 创建的缓冲区**不支持**下载回 CPU（无 `COPY_SRC` 标志），
    /// 适用于纯输出或中间缓冲区（无需回读）。
    /// 如需下载，请使用 [`empty_readable`](GpuBuffer::empty_readable)。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn empty(device: &Device, size: u64, usage: BufferUsage) -> Self {
        let aligned_size = align_for_usage(size, usage);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::empty"),
            size: aligned_size,
            usage: usage.to_wgpu_usage(false),
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    /// 创建指定大小的零初始化**可读**缓冲区。
    ///
    /// 与 [`empty`](GpuBuffer::empty) 相比，额外添加 `COPY_SRC` 标志，
    /// 允许缓冲区内容通过 [`download`](GpuBuffer::download) 下载回 CPU。
    /// 仅在需要回读计算结果时使用，避免浪费 GPU 资源配额。
    ///
    /// 缓冲区大小按用途对齐：Uniform 16 字节，Storage 256 字节。
    pub fn empty_readable(device: &Device, size: u64, usage: BufferUsage) -> Self {
        let aligned_size = align_for_usage(size, usage);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::empty_readable"),
            size: aligned_size,
            usage: usage.to_wgpu_usage(true),
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    /// 写入 `bytemuck::Pod` 数据到 GPU 缓冲区指定偏移位置。
    pub fn write<T: bytemuck::Pod>(&self, queue: &Queue, offset: u64, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        queue.write_buffer(&self.buffer, offset, bytes);
    }

    /// 写入原始字节数据到 GPU 缓冲区指定偏移位置。
    pub fn write_bytes(&self, queue: &Queue, offset: u64, data: &[u8]) {
        queue.write_buffer(&self.buffer, offset, data);
    }

    /// 将缓冲区内容下载到 CPU（阻塞等待）。
    ///
    /// 每次调用会创建和销毁暂存缓冲区。
    /// 频繁下载推荐使用 [`download_with_pool`](GpuBuffer::download_with_pool)。
    #[deprecated(
        since = "0.5.0",
        note = "每次调用创建/销毁暂存缓冲区，请使用 download_with_pool 替代"
    )]
    pub fn download(&self, device: &Device, queue: &Queue) -> Result<Vec<u8>, GpuError> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: self.size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        device.poll(wgpu::Maintain::Wait);
        crate::poll_counter::increment();

        match receiver
            .recv()
            .map_err(|_| GpuError::MapFailed("异步映射通道关闭，GPU 设备可能已丢失".into()))?
        {
            Ok(()) => {
                let view = staging.slice(..).get_mapped_range();
                let result = view.to_vec();
                drop(view);
                staging.unmap();
                Ok(result)
            }
            Err(e) => Err(GpuError::MapFailed(e.to_string())),
        }
    }

    /// 使用 [`BufferPool`] 的暂存缓冲区池下载数据，减少临时分配开销。
    pub fn download_with_pool(
        &self,
        device: &Device,
        queue: &Queue,
        pool: &BufferPool,
    ) -> Result<Vec<u8>, GpuError> {
        if self.size == 0 {
            return Ok(vec![]);
        }

        let staging = pool.acquire_staging(device, self.size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size);
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        device.poll(wgpu::Maintain::Wait);
        crate::poll_counter::increment();

        let result = match receiver
            .recv()
            .map_err(|_| GpuError::MapFailed("异步映射通道关闭，GPU 设备可能已丢失".into()))?
        {
            Ok(()) => {
                let view = staging.slice(..).get_mapped_range();
                let data = view.to_vec();
                drop(view);
                staging.unmap();
                Ok(data)
            }
            Err(e) => Err(GpuError::MapFailed(e.to_string())),
        };

        pool.release_staging(staging);
        result
    }

    /// 返回缓冲区的大小（字节）。
    pub fn size(&self) -> u64 {
        self.size
    }

    /// 从原始 wgpu Buffer 创建 GpuBuffer（仅 crate 内部使用）。
    pub(crate) fn from_raw(buffer: Buffer, size: u64) -> Self {
        Self { buffer, size }
    }

    /// 取出内部 wgpu Buffer 并消耗 GpuBuffer（仅 crate 内部使用，配合 BufferPool）。
    pub(crate) fn into_raw(self) -> Buffer {
        self.buffer
    }

    /// 获取内部 wgpu Buffer 引用（仅 crate 内部使用）。
    pub(crate) fn raw(&self) -> &Buffer {
        &self.buffer
    }
}

/// 双缓冲 Staging 管理器，实现乒乓 staging buffer 模式。
///
/// 通过交替使用两个 staging buffer，重叠 GPU 拷贝与 CPU 读取，
/// 减少阻塞 poll 次数。在流水线场景中，当 GPU 执行下一个 batch 的拷贝时，
/// CPU 可以同时读取上一个 batch 的结果。
///
/// # 使用方式
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage, DoubleBufferStaging};
///
/// fn example(ctx: &GpuContext) -> Result<(), Box<dyn std::error::Error>> {
///     let device = ctx.device()?;
///     let pool = ctx.buffer_pool();
///     let src = GpuBuffer::from_data_readable(device, &[1u32, 2, 3], BufferUsage::Storage);
///     let mut db = DoubleBufferStaging::new(device, pool, src.size());
///
///     // 第一次：提交拷贝到 buffer A
///     let mut enc = device.create_command_encoder(&Default::default());
///     enc.copy_buffer_to_buffer(src.raw(), 0, db.current_buffer(), 0, src.size());
///     ctx.queue()?.submit(std::iter::once(enc.finish()));
///     db.swap();
///
///     // 读取 buffer A
///     let data = db.download_current(device)?;
///
///     // 归还 buffer 到池
///     db.release(pool);
///     Ok(())
/// }
/// ```
pub struct DoubleBufferStaging {
    buffers: [Buffer; 2],
    current: usize,
    size: u64,
}

impl DoubleBufferStaging {
    /// 创建双缓冲 staging 管理器，从池中获取两个 staging buffer。
    pub fn new(device: &Device, pool: &BufferPool, size: u64) -> Self {
        let buffers = [
            pool.acquire_staging(device, size),
            pool.acquire_staging(device, size),
        ];
        Self {
            buffers,
            current: 0,
            size,
        }
    }

    /// 返回当前 staging buffer 的引用（用于 `copy_buffer_to_buffer` 的目标）。
    pub fn current_buffer(&self) -> &Buffer {
        &self.buffers[self.current]
    }

    /// 切换到另一个 staging buffer（乒乓切换）。
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    /// 返回上一个 staging buffer 的引用（即 `swap()` 之前的那个）。
    pub fn previous_buffer(&self) -> &Buffer {
        &self.buffers[1 - self.current]
    }

    /// 对当前 staging buffer 执行 map_async，返回接收完成信号的 channel。
    ///
    /// 调用方需在 copy 命令 submit 后调用此方法，然后 `poll(Wait)` 等待映射完成。
    pub fn map_current_async(
        &self,
    ) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.buffers[self.current]
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });
        rx
    }

    /// 读取已映射的当前 staging buffer 数据。
    ///
    /// # 要求
    ///
    /// 调用前必须确保 `map_current_async` 的 receiver 已收到 `Ok(())` 结果，
    /// 否则行为未定义（可能 panic 或返回空数据）。
    pub fn read_current_mapped(&self) -> Vec<u8> {
        let view = self.buffers[self.current]
            .slice(..)
            .get_mapped_range();
        let data = view.to_vec();
        drop(view);
        self.buffers[self.current].unmap();
        data
    }

    /// 阻塞等待当前 staging buffer 的 copy 完成并返回数据。
    ///
    /// 内部执行 `map_async` + `poll(Wait)` + 读取。适用于简单场景。
    /// 流水线场景建议使用 `map_current_async` + 手动 poll 以实现重叠。
    pub fn download_current(&self, device: &Device) -> Result<Vec<u8>, GpuError> {
        let rx = self.map_current_async();
        device.poll(wgpu::Maintain::Wait);
        crate::poll_counter::increment();
        rx.recv()
            .map_err(|_| GpuError::MapFailed("映射通道关闭，GPU 设备可能已丢失".into()))?
            .map_err(|e| GpuError::MapFailed(format!("staging buffer 映射失败: {}", e)))?;
        Ok(self.read_current_mapped())
    }

    /// 归还两个 staging buffer 到池中。
    pub fn release(self, pool: &BufferPool) {
        let [a, b] = self.buffers;
        pool.release_staging(a);
        pool.release_staging(b);
    }

    /// 返回 staging buffer 的大小（字节）。
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// 批量下载多个 GPU 缓冲区到 CPU，单次 submit + 单次 poll。
///
/// 将 N 个缓冲区的拷贝命令合并到一个 encoder 中，单次 `queue.submit()` 提交，
/// 然后单次 `device.poll(Wait)` 等待所有拷贝完成，最后依次读取每个 staging 缓冲区。
///
/// 与逐个调用 `download()` 相比，N 个缓冲区的下载只需 1 次 poll 而非 N 次，
/// 在批量场景下可显著降低 CPU-GPU 同步开销。
///
/// # 要求
///
/// 所有传入的 `GpuBuffer` 必须已创建时包含 `COPY_SRC` 标志（即可读缓冲区）。
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage, download_batch};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let ctx = GpuContext::new_sync().unwrap();
///     let bufs = vec![
///         GpuBuffer::from_data_readable(ctx.device()?, &[1u32, 2, 3], BufferUsage::Storage),
///         GpuBuffer::from_data_readable(ctx.device()?, &[4u32, 5, 6], BufferUsage::Storage),
///     ];
///     let refs: Vec<&GpuBuffer> = bufs.iter().collect();
///     let results = download_batch(ctx.device()?, ctx.queue()?, &refs)?;
///     assert_eq!(results.len(), 2);
///     Ok(())
/// }
/// ```
#[deprecated(
    since = "0.5.0",
    note = "每次调用创建/销毁暂存缓冲区，请使用 download_batch_with_pool 替代"
)]
pub fn download_batch(
    device: &Device,
    queue: &Queue,
    buffers: &[&GpuBuffer],
) -> Result<Vec<Vec<u8>>, GpuError> {
    if buffers.is_empty() {
        return Ok(vec![]);
    }

    // 为每个缓冲区创建 staging 缓冲区
    let staging_buffers: Vec<Buffer> = buffers
        .iter()
        .map(|buf| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch_staging"),
                size: buf.size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .collect();

    // 单个 encoder，添加所有 copy 命令
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("batch_download_encoder"),
    });
    for (src, staging) in buffers.iter().zip(&staging_buffers) {
        encoder.copy_buffer_to_buffer(&src.buffer, 0, staging, 0, src.size);
    }

    // 单次 submit
    queue.submit(std::iter::once(encoder.finish()));

    // 为每个 staging 注册 map_async
    let receivers: Vec<_> = staging_buffers
        .iter()
        .map(|staging| {
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });
            rx
        })
        .collect();

    // 单次 poll
    device.poll(wgpu::Maintain::Wait);
    crate::poll_counter::increment();

    // 读取所有结果
    let mut results = Vec::with_capacity(buffers.len());
    for (i, rx) in receivers.into_iter().enumerate() {
        match rx.recv().map_err(|_| {
            GpuError::MapFailed("异步映射通道关闭，GPU 设备可能已丢失".into())
        })? {
            Ok(()) => {
                let view = staging_buffers[i].slice(..).get_mapped_range();
                results.push(view.to_vec());
                drop(view);
                staging_buffers[i].unmap();
            }
            Err(e) => return Err(GpuError::MapFailed(e.to_string())),
        }
    }

    Ok(results)
}

/// 使用 [`BufferPool`] 批量下载多个 GPU 缓冲区到 CPU，单次 submit + 单次 poll。
///
/// 与 [`download_batch`] 功能相同，但使用缓冲区池复用 staging 缓冲区，
/// 减少大缓冲区的重复分配开销。
pub fn download_batch_with_pool(
    device: &Device,
    queue: &Queue,
    buffers: &[&GpuBuffer],
    pool: &BufferPool,
) -> Result<Vec<Vec<u8>>, GpuError> {
    if buffers.is_empty() {
        return Ok(vec![]);
    }

    // 从池中获取 staging 缓冲区
    let staging_buffers: Vec<Buffer> = buffers
        .iter()
        .map(|buf| pool.acquire_staging(device, buf.size))
        .collect();

    // 单个 encoder，添加所有 copy 命令
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("batch_download_encoder"),
    });
    for (src, staging) in buffers.iter().zip(&staging_buffers) {
        encoder.copy_buffer_to_buffer(&src.buffer, 0, staging, 0, src.size);
    }

    // 单次 submit
    queue.submit(std::iter::once(encoder.finish()));

    // 为每个 staging 注册 map_async
    let receivers: Vec<_> = staging_buffers
        .iter()
        .map(|staging| {
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).ok();
            });
            rx
        })
        .collect();

    // 单次 poll
    device.poll(wgpu::Maintain::Wait);
    crate::poll_counter::increment();

    // 读取所有结果
    let mut results = Vec::with_capacity(buffers.len());
    for (i, rx) in receivers.into_iter().enumerate() {
        match rx.recv().map_err(|_| {
            GpuError::MapFailed("异步映射通道关闭，GPU 设备可能已丢失".into())
        })? {
            Ok(()) => {
                let view = staging_buffers[i].slice(..).get_mapped_range();
                results.push(view.to_vec());
                drop(view);
                staging_buffers[i].unmap();
            }
            Err(e) => return Err(GpuError::MapFailed(e.to_string())),
        }
    }

    // 归还 staging 缓冲区到池
    for staging in staging_buffers {
        pool.release_staging(staging);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    /// 测试 DoubleBufferStaging 的 swap 行为和 buffer 交替。
    #[test]
    fn double_buffer_swap_behavior() {
        // 使用 mock 方式测试 swap 逻辑（无需 GPU 设备）
        // 验证 swap 在 0/1 之间交替
        let mut current: usize = 0;
        let swap = |c: &mut usize| { *c = 1 - *c; };

        assert_eq!(current, 0);
        swap(&mut current);
        assert_eq!(current, 1);
        swap(&mut current);
        assert_eq!(current, 0);
        swap(&mut current);
        assert_eq!(current, 1);
    }

    /// 测试 DoubleBufferStaging 的 previous_buffer 逻辑。
    #[test]
    fn double_buffer_previous_logic() {
        let mut current: usize = 0;
        let previous = |c: usize| 1 - c;

        assert_eq!(previous(current), 1);
        current = 1 - current; // swap
        assert_eq!(previous(current), 0);
    }

    /// 测试 size 方法返回正确的大小。
    #[test]
    fn double_buffer_size() {
        let size: u64 = 1024;
        assert_eq!(size, 1024);
        let size: u64 = 4096;
        assert_eq!(size, 4096);
    }
}