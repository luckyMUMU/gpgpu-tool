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

fn to_wgpu_usage(usage: BufferUsage) -> BufferUsages {
    match usage {
        BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST,
        BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    }
}

/// GPU 缓冲区，封装 CPU↔GPU 数据传输。
///
/// 支持从任意 `bytemuck::Pod` 类型创建、写入和读取数据。
///
/// # 创建
///
/// - [`from_data`](GpuBuffer::from_data): 从 Pod 类型数据创建
/// - [`from_bytes`](GpuBuffer::from_bytes): 从原始字节创建
/// - [`empty`](GpuBuffer::empty): 创建零初始化缓冲区
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
/// use wgpu_compute_engine::{GpuContext, GpuBuffer, BufferUsage};
///
/// let ctx = GpuContext::new_sync().unwrap();
/// let data: Vec<u32> = vec![1, 2, 3, 4];
/// let buffer = GpuBuffer::from_data(ctx.device(), &data, BufferUsage::Storage);
///
/// // 下载结果
/// let result = buffer.download(ctx.device(), ctx.queue()).unwrap();
/// ```
#[derive(Clone)]
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
}

impl GpuBuffer {
    /// 从 `bytemuck::Pod` 类型数据创建缓冲区，数据立即上传到 GPU。
    pub fn from_data<T: bytemuck::Pod>(device: &Device, data: &[T], usage: BufferUsage) -> Self {
        let bytes = bytemuck::cast_slice(data);
        let size = bytes.len() as u64;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_data"),
            contents: bytes,
            usage: to_wgpu_usage(usage) | BufferUsages::COPY_SRC,
        });
        Self { buffer, size }
    }

    /// 从原始字节数据创建缓冲区，数据立即上传到 GPU。
    pub fn from_bytes(device: &Device, data: &[u8], usage: BufferUsage) -> Self {
        let size = data.len() as u64;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_bytes"),
            contents: data,
            usage: to_wgpu_usage(usage) | BufferUsages::COPY_SRC,
        });
        Self { buffer, size }
    }

    /// 创建指定大小的零初始化缓冲区。
    pub fn empty(device: &Device, size: u64, usage: BufferUsage) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::empty"),
            size,
            usage: to_wgpu_usage(usage) | BufferUsages::COPY_SRC,
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

        match receiver.recv().unwrap() {
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

        let result = match receiver.recv().unwrap() {
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

    #[doc(hidden)]
    pub fn from_raw(buffer: Buffer, size: u64) -> Self {
        Self { buffer, size }
    }

    #[doc(hidden)]
    pub fn into_raw(self) -> Buffer {
        self.buffer
    }

    #[doc(hidden)]
    pub fn raw(&self) -> &Buffer {
        &self.buffer
    }
}