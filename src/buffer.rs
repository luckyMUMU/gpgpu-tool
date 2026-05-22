use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages, Device, Queue};

use crate::buffer_pool::BufferPool;
use crate::error::GpuError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    Storage,
    Uniform,
}

fn to_wgpu_usage(usage: BufferUsage) -> BufferUsages {
    match usage {
        BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST,
        BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    }
}

#[derive(Clone)]
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
}

impl GpuBuffer {
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

    pub fn from_bytes(device: &Device, data: &[u8], usage: BufferUsage) -> Self {
        let size = data.len() as u64;
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuBuffer::from_bytes"),
            contents: data,
            usage: to_wgpu_usage(usage) | BufferUsages::COPY_SRC,
        });
        Self { buffer, size }
    }

    pub fn empty(device: &Device, size: u64, usage: BufferUsage) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::empty"),
            size,
            usage: to_wgpu_usage(usage) | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self { buffer, size }
    }

    pub fn write<T: bytemuck::Pod>(&self, queue: &Queue, offset: u64, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        queue.write_buffer(&self.buffer, offset, bytes);
    }

    pub fn write_bytes(&self, queue: &Queue, offset: u64, data: &[u8]) {
        queue.write_buffer(&self.buffer, offset, data);
    }

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

    pub fn download_with_pool(&self, device: &Device, queue: &Queue, pool: &BufferPool) -> Result<Vec<u8>, GpuError> {
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

    pub fn from_raw(buffer: Buffer, size: u64) -> Self {
        Self { buffer, size }
    }

    pub fn into_raw(self) -> Buffer {
        self.buffer
    }

    pub fn raw(&self) -> &Buffer {
        &self.buffer
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}
