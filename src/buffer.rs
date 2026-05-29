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
///
/// # 示例
///
/// ```no_run
/// use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage};
///
/// let ctx = GpuContext::new_sync().unwrap();
/// let data: Vec<u32> = vec![1, 2, 3, 4];
/// let buffer = GpuBuffer::from_data_readable(ctx.device()?, &data, BufferUsage::Storage);
///
/// // 下载结果
/// let result = buffer.download(ctx.device()?, ctx.queue()?).unwrap();
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