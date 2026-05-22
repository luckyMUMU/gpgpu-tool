use std::sync::Arc;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::buffer_pool::BufferPool;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;

const RESIZE_WGSL: &str = include_str!("resize.wgsl");
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParams {
    image_count: u32,
    src_w: u32,
    src_h: u32,
    packed_dst: u32,
}

pub struct GpuResize {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
    buffer_pool: BufferPool,
}

impl GpuResize {
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_workgroup_size(ctx, DEFAULT_WORKGROUP_SIZE)
    }

    pub fn with_workgroup_size(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(RESIZE_WGSL, workgroup_size)?;
        Ok(Self {
            pipeline,
            workgroup_size,
            buffer_pool: BufferPool::new(),
        })
    }

    pub fn resize_batch(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        target_width: u32,
        target_height: u32,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        if images.len() != dimensions.len() {
            return Err(GpuError::InvalidInput(
                "图像数量与尺寸数量不匹配".to_string(),
            ));
        }

        let (src_w, src_h) = dimensions[0];
        let src_pixels = (src_w * src_h) as usize;
        for (i, img) in images.iter().enumerate() {
            if dimensions[i] != (src_w, src_h) {
                return Err(GpuError::InvalidInput(
                    "批量缩放要求所有图像尺寸相同".to_string(),
                ));
            }
            if img.len() != src_pixels {
                return Err(GpuError::InvalidInput(format!(
                    "第 {} 张图像像素数量与声明尺寸不符",
                    i
                )));
            }
        }

        let src_u32_per_image = src_pixels.div_ceil(4) as u64;
        let dst_pixels = (target_width * target_height) as usize;
        let dst_u32_per_image = dst_pixels as u64;
        let u32_per_image = src_u32_per_image + dst_u32_per_image;

        let max_batch = if u32_per_image > 0 {
            ((MAX_BUFFER_SIZE / 2) / (u32_per_image * 4)).max(1) as usize
        } else {
            images.len()
        };

        let mut all_resized = Vec::with_capacity(images.len());

        for chunk_start in (0..images.len()).step_by(max_batch) {
            let chunk_end = (chunk_start + max_batch).min(images.len());
            let chunk = &images[chunk_start..chunk_end];
            let resized = self.resize_batch_inner(
                ctx, chunk, src_w, src_h, target_width, target_height,
            )?;
            all_resized.extend(resized);
        }

        Ok(all_resized)
    }

    pub fn resize_batch_gpu(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        if images.is_empty() {
            return Err(GpuError::InvalidInput("图像列表为空".to_string()));
        }

        if images.len() != dimensions.len() {
            return Err(GpuError::InvalidInput(
                "图像数量与尺寸数量不匹配".to_string(),
            ));
        }

        let (src_w, src_h) = dimensions[0];
        let src_pixels = (src_w * src_h) as usize;
        for (i, img) in images.iter().enumerate() {
            if dimensions[i] != (src_w, src_h) {
                return Err(GpuError::InvalidInput(
                    "批量缩放要求所有图像尺寸相同".to_string(),
                ));
            }
            if img.len() != src_pixels {
                return Err(GpuError::InvalidInput(format!(
                    "第 {} 张图像像素数量与声明尺寸不符",
                    i
                )));
            }
        }

        self.resize_batch_gpu_inner(
            ctx, images, src_w, src_h, target_width, target_height,
        )
    }

    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    fn resize_batch_gpu_inner(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<(GpuBuffer, usize), GpuError> {
        let device = ctx.device();
        let queue = ctx.queue();
        let image_count = images.len();
        let src_pixels = (src_w * src_h) as usize;
        let dst_pixels = (target_width * target_height) as usize;

        let src_u32_per_image = src_pixels.div_ceil(4);
        let total_src_u32 = image_count * src_u32_per_image;
        let mut all_pixels: Vec<u32> = vec![0u32; total_src_u32];
        for (img_idx, img) in images.iter().enumerate() {
            let dst_slice = &mut all_pixels[img_idx * src_u32_per_image..(img_idx + 1) * src_u32_per_image];
            let dst_bytes = bytemuck::cast_slice_mut::<u32, u8>(dst_slice);
            let copy_len = img.len().min(dst_bytes.len());
            dst_bytes[..copy_len].copy_from_slice(&img[..copy_len]);
        }

        let input_size = (all_pixels.len() * 4) as u64;
        let output_u32_count = image_count * dst_pixels;
        let output_size = (output_u32_count * 4) as u64;

        let input_buffer_raw = self
            .buffer_pool
            .acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .buffer_pool
            .acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = ResizeParams {
            image_count: image_count as u32,
            src_w,
            src_h,
            packed_dst: (target_width << 16) | target_height,
        };
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

        let dispatch_x = (image_count as u32)
            .div_ceil(self.workgroup_size[0])
            .max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            [dispatch_x, 1, 1],
        );

        self.buffer_pool.release(input_buffer.into_raw());

        Ok((output_buffer, output_u32_count))
    }

    fn resize_batch_inner(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        src_w: u32,
        src_h: u32,
        target_width: u32,
        target_height: u32,
    ) -> Result<Vec<Vec<u8>>, GpuError> {
        let device = ctx.device();
        let queue = ctx.queue();
        let image_count = images.len();
        let src_pixels = (src_w * src_h) as usize;
        let dst_pixels = (target_width * target_height) as usize;

        let src_u32_per_image = src_pixels.div_ceil(4);
        let total_src_u32 = image_count * src_u32_per_image;
        let mut all_pixels: Vec<u32> = vec![0u32; total_src_u32];
        for (img_idx, img) in images.iter().enumerate() {
            let dst_slice = &mut all_pixels[img_idx * src_u32_per_image..(img_idx + 1) * src_u32_per_image];
            let dst_bytes = bytemuck::cast_slice_mut::<u32, u8>(dst_slice);
            let copy_len = img.len().min(dst_bytes.len());
            dst_bytes[..copy_len].copy_from_slice(&img[..copy_len]);
        }

        let input_size = (all_pixels.len() * 4) as u64;
        let output_u32_count = image_count * dst_pixels;
        let output_size = (output_u32_count * 4) as u64;

        let input_buffer_raw = self
            .buffer_pool
            .acquire(device, input_size, BufferUsage::Storage);
        let output_buffer_raw = self
            .buffer_pool
            .acquire(device, output_size, BufferUsage::Storage);

        queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&all_pixels));

        let input_buffer = GpuBuffer::from_raw(input_buffer_raw, input_size);
        let output_buffer = GpuBuffer::from_raw(output_buffer_raw, output_size);

        let params = ResizeParams {
            image_count: image_count as u32,
            src_w,
            src_h,
            packed_dst: (target_width << 16) | target_height,
        };
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

        let dispatch_x = (image_count as u32)
            .div_ceil(self.workgroup_size[0])
            .max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            [dispatch_x, 1, 1],
        );

        let result = output_buffer.download_with_pool(device, queue, &self.buffer_pool)?;

        self.buffer_pool.release(input_buffer.into_raw());
        self.buffer_pool.release(output_buffer.into_raw());

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        let mut resized_images = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let mut pixels = Vec::with_capacity(dst_pixels);
            let offset = i * dst_pixels;
            for j in 0..dst_pixels {
                pixels.push((raw_u32[offset + j] & 0xFF) as u8);
            }
            resized_images.push(pixels);
        }

        Ok(resized_images)
    }
}
