use crate::buffer::GpuBuffer;
use crate::buffer_pool::BufferPool;
use crate::context::GpuContext;
use crate::error::GpuError;
use crate::tasks::convolution::{BorderMode, GpuConvolution};

/// GPU 高斯模糊预处理模块。
///
/// 基于可分离卷积实现高斯模糊，用于感知哈希的图像降噪。
pub struct GpuGaussianBlur {
    convolution: GpuConvolution,
}

impl GpuGaussianBlur {
    /// 创建 GPU 高斯模糊计算器。
    pub fn new(ctx: &mut GpuContext) -> Result<Self, GpuError> {
        let convolution = GpuConvolution::new(ctx)?;
        Ok(Self { convolution })
    }

    /// 对灰度图像执行高斯模糊。
    ///
    /// kernel_size 必须为正奇数（3/5/7/9/11），sigma 为高斯核标准差。
    /// sigma <= 0 时自动计算为 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8（OpenCV 默认公式）。
    #[allow(clippy::too_many_arguments)]
    pub fn blur(
        &self,
        ctx: &GpuContext,
        pixels: &[u8],
        width: u32,
        height: u32,
        kernel_size: u32,
        sigma: f32,
    ) -> Result<Vec<u8>, GpuError> {
        let kernel_1d = generate_gaussian_kernel_1d(kernel_size, sigma)?;
        self.convolution.convolve_separable(
            ctx,
            pixels,
            width,
            height,
            &kernel_1d,
            kernel_size,
            BorderMode::Clamp,
        )
    }

    /// 对 GPU buffer 中的图像执行高斯模糊，返回 GPU buffer（零拷贝流水线）。
    #[allow(clippy::too_many_arguments)]
    pub fn blur_gpu(
        &self,
        ctx: &GpuContext,
        input_buffer: &GpuBuffer,
        input_u32_count: usize,
        width: u32,
        height: u32,
        kernel_size: u32,
        sigma: f32,
    ) -> Result<GpuBuffer, GpuError> {
        let kernel_1d = generate_gaussian_kernel_1d(kernel_size, sigma)?;
        self.convolution.convolve_separable_gpu(
            ctx,
            input_buffer,
            input_u32_count,
            width,
            height,
            &kernel_1d,
            kernel_size,
            BorderMode::Clamp,
        )
    }

    /// 返回内部缓冲区复用池的引用。
    pub fn buffer_pool(&self) -> &BufferPool {
        self.convolution.buffer_pool()
    }
}

/// 生成 1D 高斯核。
///
/// 使用高斯函数 G(x) = exp(-x²/(2σ²))，归一化后使核元素之和为 1。
/// sigma <= 0 时使用 OpenCV 默认公式自动计算。
fn generate_gaussian_kernel_1d(kernel_size: u32, sigma: f32) -> Result<Vec<f32>, GpuError> {
    if kernel_size == 0 || kernel_size.is_multiple_of(2) {
        return Err(GpuError::InvalidInput(
            "核大小必须为正奇数".to_string(),
        ));
    }
    if kernel_size > 11 {
        return Err(GpuError::InvalidInput(
            "核大小不能超过 11".to_string(),
        ));
    }

    let sigma = if sigma <= 0.0 {
        0.3 * ((kernel_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    let radius = (kernel_size / 2) as i32;
    let mut kernel = Vec::with_capacity(kernel_size as usize);
    let mut sum = 0.0f32;

    for x in -radius..=radius {
        let x_f = x as f32;
        let val = (-x_f * x_f / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }

    for v in &mut kernel {
        *v /= sum;
    }

    Ok(kernel)
}
