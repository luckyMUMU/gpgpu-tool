use std::sync::Arc;

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::pipeline::ComputePipeline;
use crate::tasks::hash_common::{PerceptualHashComputer, PhashParams};

const DOUBLE_GRADIENT_HASH_WGSL: &str = include_str!("double_gradient_hash.wgsl");
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Double Gradient Hash（双梯度哈希）GPU 计算器。
///
/// 对每幅 9x9 灰度图像同时计算水平和垂直方向的相邻像素差值，
/// 各取 32bit 组合为 64bit 哈希值。
pub struct DoubleGradientHashComputer {
    pipeline: Arc<ComputePipeline>,
    workgroup_size: [u32; 3],
}

impl DoubleGradientHashComputer {
    /// 创建 Double Gradient Hash 计算器，使用默认 workgroup_size [256, 1, 1]。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        Self::with_workgroup_size(ctx, DEFAULT_WORKGROUP_SIZE)
    }

    /// 创建 Double Gradient Hash 计算器，指定 workgroup_size。
    pub fn with_workgroup_size(
        ctx: &mut GpuContext,
        workgroup_size: [u32; 3],
    ) -> Result<Self, GpuError> {
        let pipeline = ctx.get_or_create_pipeline(DOUBLE_GRADIENT_HASH_WGSL, workgroup_size)?;
        Ok(Self {
            pipeline,
            workgroup_size,
        })
    }
}

impl PerceptualHashComputer for DoubleGradientHashComputer {
    fn compute(
        &self,
        ctx: &GpuContext,
        images: &[Vec<u8>],
    ) -> Result<Vec<u64>, GpuError> {
        if images.is_empty() {
            return Ok(vec![]);
        }

        let device = ctx.device();
        let queue = ctx.queue();

        let image_count = images.len();
        let pixels_per_image = images[0].len();
        let width = (pixels_per_image as u32).isqrt();
        let height = width;

        let mut all_pixels: Vec<u32> = Vec::with_capacity(image_count * pixels_per_image);
        for img in images {
            if img.len() != pixels_per_image {
                return Err(GpuError::InvalidInput(
                    "所有图像尺寸必须一致".to_string(),
                ));
            }
            for &pixel in img {
                all_pixels.push(pixel as u32);
            }
        }

        let input_buffer = GpuBuffer::from_data(device, &all_pixels, BufferUsage::Storage);
        let output_size = (image_count * 2 * 4) as u64;
        let output_buffer = GpuBuffer::empty(device, output_size, BufferUsage::Storage);

        let params = PhashParams {
            width,
            height,
            image_count: image_count as u32,
            _padding: 0,
        };
        let params_buffer = GpuBuffer::from_data(device, &[params], BufferUsage::Uniform);

        let dispatch_x = (image_count as u32).div_ceil(self.workgroup_size[0]).max(1);
        self.pipeline.dispatch(
            device,
            queue,
            &input_buffer,
            &output_buffer,
            &params_buffer,
            [dispatch_x, 1, 1],
        );

        let result = output_buffer.download(device, queue)?;
        let raw_u32 = bytemuck::cast_slice::<u8, u32>(&result);

        let mut hashes = Vec::with_capacity(image_count);
        for i in 0..image_count {
            let low = raw_u32[i * 2] as u64;
            let high = raw_u32[i * 2 + 1] as u64;
            hashes.push(low | (high << 32));
        }

        Ok(hashes)
    }
}
