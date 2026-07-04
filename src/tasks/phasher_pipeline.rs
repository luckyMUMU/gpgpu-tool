use crate::batch::GpuBatchSubmitter;
use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::hash_common::{
    compute_phash_from_gpu_buffer_batch,
    compute_phash_from_gpu_buffer_with_thresholds, parse_batch_hash_results,
};
use crate::tasks::phasher::PerceptualHasher;
use crate::tasks::phasher_util::{
    merge_gpu_buffers, merge_gpu_buffers_batch, PackedChunk,
};

impl PerceptualHasher {
    /// GPU 批量管线：所有图像同尺寸时的 预处理 → 缩放 → 哈希。
    ///
    /// 含分块逻辑，避免单次 GPU dispatch 超出缓冲区限制。
    /// 使用 CPU/GPU 双缓冲流水线：CPU 线程预打包下一个分块的像素数据时，
    /// 主线程同步执行当前分块的 GPU dispatch（上传 + 模糊 + 缩放 + 哈希）。
    pub(super) fn compute_gpu_batch_pipeline(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        all_target_size: bool,
    ) -> Result<Vec<u64>, GpuError> {
        let (src_w, src_h) = dimensions[0];
        let src_pixels = (src_w as u64 * src_h as u64) as usize;
        let dst_pixels = (self.target_width as u64 * self.target_height as u64) as usize;
        let src_u32_per_image = src_pixels as u64;
        let dst_u32_per_image = dst_pixels as u64;
        let u32_per_image = if all_target_size {
            src_u32_per_image
        } else {
            src_u32_per_image + dst_u32_per_image
        };

        // 检查单张图像是否超过 max_storage_buffer_binding_size
        // u32 对齐后每像素 4 字节，加上缩放输出缓冲区的额外需求
        let per_image_bytes = src_u32_per_image * 4;
        let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;
        if per_image_bytes > max_binding {
            return Err(GpuError::InvalidInput(format!(
                "单张图像大小 ({src_w}×{src_h} = {per_image_bytes} 字节 u32 对齐) 超过 \
                 max_storage_buffer_binding_size ({max_binding} 字节)。\
                 请减小图像尺寸或使用支持更大绑定大小的 GPU 适配器。"
            )));
        }

        let max_batch = if u32_per_image > 0 {
            (self.max_batch_size / (u32_per_image * 4)).max(1) as usize
        } else {
            images.len()
        };

        // 构建分块边界
        let chunks: Vec<(usize, usize)> = (0..images.len())
            .step_by(max_batch)
            .map(|start| (start, (start + max_batch).min(images.len())))
            .collect();

        // 单个分块或无分块：流水线开销不值得，直接串行处理
        if chunks.len() <= 1 {
            let mut all_hashes = Vec::with_capacity(images.len());
            self.process_chunks_serial(
                ctx, images, dimensions, all_target_size,
                src_pixels, src_w, src_h, &chunks, &mut all_hashes,
            )?;
            return Ok(all_hashes);
        }

        // ========== 双缓冲流水线 ==========
        // CPU 线程：预打包像素数据（u8 → u32）
        // 主线程：GPU dispatch（上传 + 模糊 + 缩放 + 哈希）
        // sync_channel 容量为 1，实现背压：CPU 可提前准备一个分块
        use std::sync::mpsc;

        let (packed_tx, packed_rx) = mpsc::sync_channel::<PackedChunk>(1);
        let mut all_hashes = Vec::with_capacity(images.len());

        std::thread::scope(|s| {
            // CPU 工作线程：预打包像素数据（u8 → u32 转换）
            // 仅做纯 CPU 计算，不访问 GpuContext
            s.spawn(|| {
                for &(start, end) in &chunks {
                    let chunk_images = &images[start..end];
                    let chunk_dims = &dimensions[start..end];
                    let packed: Vec<Vec<u32>> = chunk_images
                        .iter()
                        .map(|img| crate::pixel_pack::pack_u8_to_u32(img))
                        .collect();
                    let dims = chunk_dims.to_vec();
                    if packed_tx.send(PackedChunk { packed, dims }).is_err() {
                        break; // 接收端已关闭
                    }
                }
            });

            // 主线程：接收预打包数据并执行 GPU dispatch
            for &(start, end) in &chunks {
                let chunk_data = packed_rx.recv().map_err(|_| {
                    GpuError::Internal(
                        "GPU 流水线：CPU 工作线程意外断开连接".to_string(),
                    )
                })?;

                let chunk_len = end - start;

                // 使用 GpuBatchSubmitter 将所有 dispatch 编码到单个 encoder
                let mut batch = GpuBatchSubmitter::new();

                // 阶段一：预处理（上传已打包数据 + 可选模糊，编码到 batch）
                let gpu_images = self.preprocess_gpu_from_packed_batch(
                    ctx, &mut batch, &chunk_data.packed, &chunk_data.dims,
                )?;

                if all_target_size {
                    // 已缩放（有模糊），合并后直接计算哈希
                    let merged = merge_gpu_buffers_batch(ctx, &mut batch, &gpu_images, src_pixels)?;
                    release_buffers(ctx, gpu_images);
                    let (hash_output, img_count, hs) = compute_phash_from_gpu_buffer_batch(
                        self.computer.pipeline(), ctx, &mut batch, &merged,
                        chunk_len, self.target_width, self.target_height,
                        self.computer.workgroup_size(), self.hash_size,
                    )?;
                    let results = batch.wait_all()?;
                    if let Some(raw_data) = results.first() {
                        let hashes = parse_batch_hash_results(raw_data, img_count, hs);
                        all_hashes.extend(hashes);
                    }
                    ctx.buffer_pool().release(merged.into_raw(), BufferUsage::Storage);
                    ctx.buffer_pool().release(hash_output.into_raw(), BufferUsage::Storage);
                } else {
                    // 阶段二：缩放（编码到 batch）
                    let (resized, _u32_count) = self.resize_gpu_batch(ctx, &mut batch, &gpu_images, src_w, src_h)?;
                    release_buffers(ctx, gpu_images);
                    // 阶段三：哈希（编码到 batch + staging buffer）
                    let (hash_output, img_count, hs) = compute_phash_from_gpu_buffer_batch(
                        self.computer.pipeline(), ctx, &mut batch, &resized,
                        chunk_len, self.target_width, self.target_height,
                        self.computer.workgroup_size(), self.hash_size,
                    )?;
                    let results = batch.wait_all()?;
                    if let Some(raw_data) = results.first() {
                        let hashes = parse_batch_hash_results(raw_data, img_count, hs);
                        all_hashes.extend(hashes);
                    }
                    ctx.buffer_pool().release(resized.into_raw(), BufferUsage::Storage);
                    ctx.buffer_pool().release(hash_output.into_raw(), BufferUsage::Storage);
                }
            }

            Ok::<(), GpuError>(())
        })
        .map_err(|e| {
            GpuError::Internal(format!("GPU 流水线工作线程 panic: {:?}", e))
        })?;

        Ok(all_hashes)
    }

    /// 串行处理分块（无流水线开销的回退路径）。
    ///
    /// 使用 GpuBatchSubmitter 将所有 dispatch 命令编码到单个 CommandEncoder，
    /// 通过单次 submit + 单次 poll 完成整个流水线，大幅减少 GPU 提交开销。
    pub(super) fn process_chunks_serial(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
        all_target_size: bool,
        src_pixels: usize,
        src_w: u32,
        src_h: u32,
        chunks: &[(usize, usize)],
        all_hashes: &mut Vec<u64>,
    ) -> Result<(), GpuError> {
        let needs_threshold = matches!(self.algorithm, crate::tasks::phasher::HashAlgorithm::Mean | crate::tasks::phasher::HashAlgorithm::Median);

        for &(start, end) in chunks {
            let chunk_images = &images[start..end];
            let chunk_dims = &dimensions[start..end];
            let chunk_len = chunk_images.len();

            if needs_threshold {
                // Mean/Median Hash 路径：先 GPU blur+resize，再 CPU 计算阈值，再 GPU hash
                let gpu_images = self.preprocess_gpu(ctx, chunk_images, chunk_dims)?;

                let resized_buffer = if all_target_size {
                    let merged = merge_gpu_buffers(ctx, &gpu_images, src_pixels)?;
                    release_buffers(ctx, gpu_images);
                    merged
                } else {
                    let (resized, _u32_count) = self.resize_gpu(ctx, &gpu_images, src_w, src_h)?;
                    release_buffers(ctx, gpu_images);
                    resized
                };

                // 从 GPU buffer 下载像素数据，计算阈值
                let thresholds = self.compute_thresholds_from_gpu_buffer(
                    ctx, &resized_buffer,
                    (self.target_width * self.target_height) as usize * chunk_len,
                    chunk_len,
                )?;

                // 使用带阈值的哈希计算
                let hashes = compute_phash_from_gpu_buffer_with_thresholds(
                    self.computer.pipeline(), ctx, &resized_buffer,
                    (self.target_width * self.target_height) as usize * chunk_len,
                    chunk_len, self.target_width, self.target_height,
                    self.computer.workgroup_size(), self.hash_size, &thresholds,
                )?;
                ctx.buffer_pool().release(resized_buffer.into_raw(), BufferUsage::Storage);
                all_hashes.extend(hashes);
            } else {
                // 其他算法：使用批量编码模式
                let mut batch = GpuBatchSubmitter::new();

                // 阶段一：预处理（上传 + 可选模糊，编码到 batch）
                let gpu_images = self.preprocess_gpu_batch(ctx, &mut batch, chunk_images, chunk_dims)?;

                if all_target_size {
                    let merged = merge_gpu_buffers_batch(ctx, &mut batch, &gpu_images, src_pixels)?;
                    release_buffers(ctx, gpu_images);
                    let (hash_output, img_count, hs) = compute_phash_from_gpu_buffer_batch(
                        self.computer.pipeline(), ctx, &mut batch, &merged,
                        chunk_len, self.target_width, self.target_height,
                        self.computer.workgroup_size(), self.hash_size,
                    )?;
                    let results = batch.wait_all()?;
                    if let Some(raw_data) = results.first() {
                        let hashes = parse_batch_hash_results(raw_data, img_count, hs);
                        all_hashes.extend(hashes);
                    }
                    ctx.buffer_pool().release(merged.into_raw(), BufferUsage::Storage);
                    ctx.buffer_pool().release(hash_output.into_raw(), BufferUsage::Storage);
                } else {
                    let (resized, _u32_count) = self.resize_gpu_batch(ctx, &mut batch, &gpu_images, src_w, src_h)?;
                    release_buffers(ctx, gpu_images);
                    let (hash_output, img_count, hs) = compute_phash_from_gpu_buffer_batch(
                        self.computer.pipeline(), ctx, &mut batch, &resized,
                        chunk_len, self.target_width, self.target_height,
                        self.computer.workgroup_size(), self.hash_size,
                    )?;
                    let results = batch.wait_all()?;
                    if let Some(raw_data) = results.first() {
                        let hashes = parse_batch_hash_results(raw_data, img_count, hs);
                        all_hashes.extend(hashes);
                    }
                    ctx.buffer_pool().release(resized.into_raw(), BufferUsage::Storage);
                    ctx.buffer_pool().release(hash_output.into_raw(), BufferUsage::Storage);
                }
            }
        }
        Ok(())
    }

    /// GPU 逐图管线：图像尺寸不同时的 预处理 → 缩放 → 哈希。
    pub(super) fn compute_gpu_per_image_pipeline(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        dimensions: &[(u32, u32)],
    ) -> Result<Vec<u64>, GpuError> {
        let mut all_hashes = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let (w, h) = dimensions[i];
            let dims = [(w, h)];

            // 阶段一：预处理
            let gpu_images = self.preprocess_gpu(ctx, std::slice::from_ref(image), &dims)?;

            if w == self.target_width && h == self.target_height {
                // 已缩放（有模糊），直接计算哈希
                let src_pixels = (w as u64 * h as u64) as usize;
                let hashes = self.compute_hash_gpu(ctx, &gpu_images[0], src_pixels, 1)?;
                release_buffers(ctx, gpu_images);
                all_hashes.extend(hashes);
            } else {
                // 阶段二：缩放
                let (resized, u32_count) = self.resize_gpu(ctx, &gpu_images, w, h)?;
                release_buffers(ctx, gpu_images);
                // 阶段三：哈希
                let hashes = self.compute_hash_gpu(ctx, &resized, u32_count, 1)?;
                ctx.buffer_pool().release(resized.into_raw(), BufferUsage::Storage);
                all_hashes.extend(hashes);
            }
        }
        Ok(all_hashes)
    }

    /// 从 GPU buffer 下载像素数据并计算阈值（均值或中位数）。
    ///
    /// 用于 Mean/Median Hash 的零拷贝管线场景。
    pub(super) fn compute_thresholds_from_gpu_buffer(
        &self,
        ctx: &GpuContext,
        gpu_buffer: &GpuBuffer,
        _u32_count: usize,
        image_count: usize,
    ) -> Result<Vec<u32>, GpuError> {
        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let buffer_pool = ctx.buffer_pool();

        let raw_pixels = gpu_buffer.download_with_pool(device, queue, buffer_pool)?;
        let raw_u32: &[u32] = bytemuck::cast_slice(&raw_pixels);
        let pixels_per_image = (self.target_width * self.target_height) as usize;

        let mut thresholds = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let base = i * pixels_per_image;
            let end = (base + pixels_per_image).min(raw_u32.len());
            let pixel_slice = &raw_u32[base..end];

            let threshold = match self.algorithm {
                crate::tasks::phasher::HashAlgorithm::Mean => {
                    // f32 均值的位模式（与 compute_means_as_u32 一致）
                    if pixel_slice.is_empty() {
                        0u32
                    } else {
                        let sum: f32 = pixel_slice.iter().map(|&p| p as f32).sum();
                        let mean = sum / pixel_slice.len() as f32;
                        mean.to_bits()
                    }
                }
                crate::tasks::phasher::HashAlgorithm::Median => {
                    // 直方图中位数（与 compute_medians 一致）
                    if pixel_slice.is_empty() {
                        0u32
                    } else {
                        let mut histogram = [0u32; 256];
                        for &pixel in pixel_slice {
                            let v = (pixel & 0xFF) as usize;
                            histogram[v] += 1;
                        }
                        let half = pixel_slice.len() as u32 / 2;
                        let mut count = 0u32;
                        let mut median = 128u32;
                        for v in 0..256u32 {
                            count += histogram[v as usize];
                            if count > half {
                                median = v;
                                break;
                            }
                        }
                        median
                    }
                }
                _ => 0u32, // 其他算法不应调用此方法
            };
            thresholds.push(threshold);
        }
        Ok(thresholds)
    }
}

/// 释放 GPU 缓冲区列表到缓冲池。
pub(crate) fn release_buffers(ctx: &GpuContext, buffers: Vec<GpuBuffer>) {
    for buf in buffers {
        ctx.buffer_pool().release(buf.into_raw(), BufferUsage::Storage);
    }
}
