use crate::batch::GpuBatchSubmitter;
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;

/// CPU 线程预打包的分块数据。
///
/// 用于双缓冲流水线：CPU 线程预打包像素数据（u8 → u32），
/// 主线程同步执行 GPU dispatch。
pub(crate) struct PackedChunk {
    pub(crate) packed: Vec<Vec<u32>>,
    pub(crate) dims: Vec<(u32, u32)>,
}

/// 将灰度图像数据上传到 GPU 存储缓冲区。
///
/// 使用 pixel_pack 将 u8 像素扩展为 u32，再通过 buffer_pool 分配并写入 GPU。
pub(crate) fn upload_image_to_gpu(
    ctx: &GpuContext,
    image: &[u8],
    width: u32,
    height: u32,
) -> Result<GpuBuffer, GpuError> {
    let expected_len = (width as usize) * (height as usize);
    if image.len() != expected_len {
        return Err(GpuError::InvalidInput(format!(
            "图像数据长度 ({}) 与声明的尺寸 ({}x{}={}) 不匹配",
            image.len(),
            width,
            height,
            expected_len
        )));
    }
    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let packed = crate::pixel_pack::pack_u8_to_u32(image);
    let input_size = (packed.len() * 4) as u64;
    let input_buffer_raw = ctx
        .buffer_pool()
        .acquire(device, input_size, BufferUsage::Storage)?;
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(&packed));
    Ok(GpuBuffer::from_raw(input_buffer_raw, input_size))
}

/// 将预打包的 u32 像素数据上传到 GPU 存储缓冲区。
///
/// 跳过 u8→u32 转换步骤，直接将已打包的像素数据写入 GPU。
/// 用于双缓冲流水线中 CPU 线程已预先完成像素打包的场景。
pub(crate) fn upload_packed_to_gpu(
    ctx: &GpuContext,
    packed: &[u32],
) -> Result<GpuBuffer, GpuError> {
    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let input_size = (packed.len() * 4) as u64;
    let input_buffer_raw = ctx
        .buffer_pool()
        .acquire(device, input_size, BufferUsage::Storage)?;
    queue.write_buffer(&input_buffer_raw, 0, bytemuck::cast_slice(packed));
    Ok(GpuBuffer::from_raw(input_buffer_raw, input_size))
}

/// 将多个 GPU 缓冲区合并为一个连续存储缓冲区。
///
/// 使用 GPU 端 copy_buffer_to_buffer 实现零拷贝合并，
/// 避免将数据下载到 CPU 再重新上传。
pub(crate) fn merge_gpu_buffers(
    ctx: &GpuContext,
    buffers: &[GpuBuffer],
    pixels_per_image: usize,
) -> Result<GpuBuffer, GpuError> {
    let device = ctx.device()?;
    let queue = ctx.queue()?;
    let byte_size_per_image = (pixels_per_image * 4) as u64;
    let total_bytes = byte_size_per_image * buffers.len() as u64;

    let merged_raw = ctx
        .buffer_pool()
        .acquire(device, total_bytes, BufferUsage::Storage)?;
    let merged = GpuBuffer::from_raw(merged_raw, total_bytes);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("merge_buffers_encoder"),
    });
    for (i, buf) in buffers.iter().enumerate() {
        let offset = (i as u64) * byte_size_per_image;
        encoder.copy_buffer_to_buffer(buf.raw(), 0, merged.raw(), offset, byte_size_per_image);
    }
    queue.submit(std::iter::once(encoder.finish()));

    Ok(merged)
}

/// 将多个 GPU 缓冲区合并为一个连续存储缓冲区（批量编码模式）。
///
/// 与 `merge_gpu_buffers()` 功能相同，但将 copy 命令编码到
/// `GpuBatchSubmitter` 的共享 CommandEncoder 中，而非独立提交。
pub(crate) fn merge_gpu_buffers_batch(
    ctx: &GpuContext,
    batch: &mut GpuBatchSubmitter,
    buffers: &[GpuBuffer],
    pixels_per_image: usize,
) -> Result<GpuBuffer, GpuError> {
    let device = ctx.device()?;
    let byte_size_per_image = (pixels_per_image * 4) as u64;
    let total_bytes = byte_size_per_image * buffers.len() as u64;

    let merged_raw = ctx
        .buffer_pool()
        .acquire(device, total_bytes, BufferUsage::Storage)?;
    let merged = GpuBuffer::from_raw(merged_raw, total_bytes);

    for (i, buf) in buffers.iter().enumerate() {
        let offset = (i as u64) * byte_size_per_image;
        batch.encode_copy(
            ctx,
            buf.raw(),
            0,
            merged.raw(),
            offset,
            byte_size_per_image,
        )?;
    }

    Ok(merged)
}

/// CPU 侧灰度图像盒式滤波下采样（零依赖，适合感知哈希场景）。
///
/// 与 GPU `resize.wgsl` 着色器逻辑一致：使用 f32 比率计算源区域，
/// 对映射到每个目标像素的源像素区域取平均（box filter）。
pub(crate) fn resize_grayscale(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return pixels.to_vec();
    }

    let sw = src_w as usize;
    let sh = src_h as usize;
    let stride = sw + 1;

    let mut sat = vec![0u64; (sh + 1) * stride];
    for y in 0..sh {
        let mut row_sum = 0u64;
        for x in 0..sw {
            row_sum += pixels[y * sw + x] as u64;
            sat[(y + 1) * stride + (x + 1)] = row_sum + sat[y * stride + (x + 1)];
        }
    }

    // 使用 f32 比率，与 GPU resize.wgsl 着色器一致
    let x_ratio = src_w as f32 / dst_w as f32;
    let y_ratio = src_h as f32 / dst_h as f32;
    let mut output = Vec::with_capacity((dst_w as u64 * dst_h as u64) as usize);

    for dy in 0..dst_h {
        let y0 = (dy as f32 * y_ratio) as usize;
        let y1 = ((dy as f32 + 1.0) * y_ratio).min(src_h as f32) as usize;
        for dx in 0..dst_w {
            let x0 = (dx as f32 * x_ratio) as usize;
            let x1 = ((dx as f32 + 1.0) * x_ratio).min(src_w as f32) as usize;
            let top_right = sat[y1 * stride + x1] - sat[y0 * stride + x1];
            let bottom_right = sat[y1 * stride + x0] - sat[y0 * stride + x0];
            let sum = top_right - bottom_right;
            let area = ((x1 - x0) * (y1 - y0)).max(1);
            output.push((sum / area as u64) as u8);
        }
    }

    output
}
