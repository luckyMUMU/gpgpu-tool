use std::sync::Arc;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::{ComputePipeline, PipelineDescriptor};
use crate::tasks::hash_common::{HashSize, PerceptualHashComputer};

const PDQ_DCT_WGSL: &str = include_str!("pdq_hash.wgsl");
const PDQ_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
const PDQ_IMAGE_SIZE: u32 = 64;
const PDQ_DCT_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DctParams {
    width: u32,
    height: u32,
    dct_size: u32,
    pass_mode: u32,
    image_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// PDQ 哈希结果，包含 256-bit 哈希和质量评分。
pub struct PdqHashResult {
    /// 256-bit 哈希值（4 个 u64）。
    pub hash: [u64; 4],
    /// 质量评分（0.0-1.0，越高表示哈希越可靠）。
    pub quality: f32,
}

/// PDQ 哈希 CPU 计算器。
///
/// 基于 DCT 频域变换的感知哈希算法（Meta/Facebook PDQ），
/// 输入为 64×64 灰度图像，输出 256-bit 哈希 + 质量评分。
pub struct PdqHashCpu;

impl PdqHashCpu {
    pub fn new() -> Self {
        Self
    }

    /// 计算已缩放到 64×64 的灰度图像的 PDQ 哈希。
    pub fn compute(&self, pixels_64x64: &[u8]) -> Result<PdqHashResult, GpuError> {
        if pixels_64x64.len() != 64 * 64 {
            return Err(GpuError::InvalidInput(format!(
                "PDQ 需要 64×64 灰度输入（4096 字节），实际 {} 字节",
                pixels_64x64.len()
            )));
        }

        let data: Vec<f64> = pixels_64x64.iter().map(|&p| p as f64).collect();

        let dct_result = dct2d_64x64(&data);

        let mut coefficients = Vec::with_capacity(256);
        for row in 0..16 {
            for col in 0..16 {
                coefficients.push(dct_result[row * 64 + col]);
            }
        }

        let mut sorted = coefficients.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = (sorted[127] + sorted[128]) / 2.0;

        let bits: Vec<bool> = coefficients.iter().map(|&c| c >= median).collect();

        let hash = pack_bits_to_u64(&bits);

        let quality = compute_quality(&coefficients, median);

        Ok(PdqHashResult { hash, quality })
    }
}

impl Default for PdqHashCpu {
    fn default() -> Self {
        Self::new()
    }
}

/// PDQ 哈希 GPU 计算器。
///
/// GPU 执行两趟 DCT-II 变换（水平 + 垂直），
/// CPU 完成中值量化和哈希打包。
pub struct PdqHashGpu {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
}

impl PdqHashGpu {
    /// 创建 PDQ GPU 计算器。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(
            &PipelineDescriptor::default_3_binding(PDQ_DCT_WGSL, PDQ_WORKGROUP_SIZE),
        )?;
        Ok(Self {
            pipeline,
            workgroup_size: PDQ_WORKGROUP_SIZE,
        })
    }

    /// 对已缩放到 64×64 的灰度图像批量计算 PDQ 哈希。
    ///
    /// GPU 执行两趟 DCT-II 变换，CPU 完成中值量化和哈希打包。
    /// 每张图像输出 `hash_size.u64s_per_image()` 个 u64（PDQ 固定为 4 个 u64 = 256-bit）。
    /// 当 `hash_size != 16` 时返回错误，因为 PDQ 算法固定输出 256-bit 哈希。
    pub fn compute(&self, ctx: &GpuContext, images: &[Vec<u8>], hash_size: HashSize) -> Result<Vec<u64>, GpuError> {
        if hash_size.size() != 16 {
            return Err(GpuError::InvalidInput(format!(
                "PDQ 哈希固定为 16×16=256 bit，不支持 hash_size={}（期望 16）",
                hash_size.size()
            )));
        }
        if images.is_empty() {
            return Ok(vec![]);
        }

        let device = ctx.device()?;
        let queue = ctx.queue()?;
        let image_count = images.len();
        let pixels_per_image = (PDQ_IMAGE_SIZE * PDQ_IMAGE_SIZE) as usize;

        for img in images {
            if img.len() != pixels_per_image {
                return Err(GpuError::InvalidInput(format!(
                    "PDQ 需要 64×64 灰度输入（4096 字节），实际 {} 字节",
                    img.len()
                )));
            }
        }

        let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * pixels_per_image);
        for img in images {
            all_pixels.extend(crate::pixel_pack::pack_u8_to_u32(img));
        }

        let buffer_size = (all_pixels.len() * 4) as u64;

        let input_raw = ctx.buffer_pool().acquire(device, buffer_size, BufferUsage::Storage);
        let intermediate_raw = ctx.buffer_pool().acquire(device, buffer_size, BufferUsage::Storage);
        let output_raw = ctx.buffer_pool().acquire(device, buffer_size, BufferUsage::Storage);

        queue.write_buffer(&input_raw, 0, bytemuck::cast_slice(&all_pixels));

        let input_buffer = GpuBuffer::from_raw(input_raw, buffer_size);
        let intermediate_buffer = GpuBuffer::from_raw(intermediate_raw, buffer_size);
        let output_buffer = GpuBuffer::from_raw(output_raw, buffer_size);

        let total_elements = (image_count * pixels_per_image) as u32;
        let dispatch_x = total_elements.div_ceil(self.workgroup_size[0]).max(1);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pdq_dct_encoder"),
        });

        let params_h = DctParams {
            width: PDQ_IMAGE_SIZE,
            height: PDQ_IMAGE_SIZE,
            dct_size: PDQ_DCT_SIZE,
            pass_mode: 0,
            image_count: image_count as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer_h = GpuBuffer::from_data(device, &[params_h], BufferUsage::Uniform);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&input_buffer, &intermediate_buffer, &params_buffer_h],
            [dispatch_x, 1, 1],
            None,
            ctx.compute_units(),
        );

        let params_v = DctParams {
            pass_mode: 1,
            ..params_h
        };
        let params_buffer_v = GpuBuffer::from_data(device, &[params_v], BufferUsage::Uniform);
        self.pipeline.encode_dispatch_into(
            device,
            &mut encoder,
            &[&intermediate_buffer, &output_buffer, &params_buffer_v],
            [dispatch_x, 1, 1],
            None,
            ctx.compute_units(),
        );

        queue.submit(std::iter::once(encoder.finish()));

        let result = output_buffer.download_with_pool(device, queue, ctx.buffer_pool())?;

        ctx.buffer_pool().release(input_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(intermediate_buffer.into_raw(), BufferUsage::Storage);
        ctx.buffer_pool().release(output_buffer.into_raw(), BufferUsage::Storage);

        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);
        let dct_f32: Vec<f32> = raw_u32.iter().map(|&v| f32::from_bits(v)).collect();

        let mut hashes = Vec::with_capacity(image_count * 4);
        for img_idx in 0..image_count {
            let offset = img_idx * pixels_per_image;

            let mut coefficients = Vec::with_capacity(256);
            for row in 0..16usize {
                for col in 0..16usize {
                    coefficients.push(dct_f32[offset + row * 64 + col] as f64);
                }
            }

            let mut sorted = coefficients.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = (sorted[127] + sorted[128]) / 2.0;

            let bits: Vec<bool> = coefficients.iter().map(|&c| c >= median).collect();
            let hash = pack_bits_to_u64(&bits);
            hashes.extend_from_slice(&hash);
        }

        Ok(hashes)
    }
}

impl PerceptualHashComputer for PdqHashGpu {
    fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        self.compute(ctx, images, HashSize::new(16))
    }

    fn compute_sized(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
        hash_size: HashSize,
    ) -> Result<Vec<u64>, GpuError> {
        self.compute(ctx, images, hash_size)
    }

    fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    fn hash_size(&self) -> HashSize {
        HashSize::new(16)
    }
}

/// 1D DCT-II 变换。
fn dct1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            sum += x
                * (std::f64::consts::PI * (2.0 * i as f64 + 1.0) * k as f64 / (2.0 * n as f64)).cos();
        }
        *out = sum;
    }
    output
}

/// 2D DCT-II 变换（可分离：先水平，再垂直）。
fn dct2d_64x64(data: &[f64]) -> Vec<f64> {
    let n = 64usize;
    let mut intermediate = vec![0.0; n * n];
    for row in 0..n {
        let row_data: Vec<f64> = (0..n).map(|col| data[row * n + col]).collect();
        let row_dct = dct1d(&row_data);
        for col in 0..n {
            intermediate[row * n + col] = row_dct[col];
        }
    }
    let mut result = vec![0.0; n * n];
    for col in 0..n {
        let col_data: Vec<f64> = (0..n).map(|row| intermediate[row * n + col]).collect();
        let col_dct = dct1d(&col_data);
        for row in 0..n {
            result[row * n + col] = col_dct[row];
        }
    }
    result
}

/// 将 256 个 bit 打包为 4 个 u64。
fn pack_bits_to_u64(bits: &[bool]) -> [u64; 4] {
    let mut hash = [0u64; 4];
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            hash[i / 64] |= 1u64 << (i % 64);
        }
    }
    hash
}

/// 计算质量评分。
/// 基于 DCT 系数与中值的偏差：偏差越大，哈希区分度越高，质量越好。
fn compute_quality(coefficients: &[f64], median: f64) -> f32 {
    let total_deviation: f64 = coefficients.iter().map(|&c| (c - median).abs()).sum();
    let avg_deviation = total_deviation / coefficients.len() as f64;
    (avg_deviation / 100.0).min(1.0) as f32
}
