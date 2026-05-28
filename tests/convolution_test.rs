use gpgpu_tool::{
    tasks::convolution::{BorderMode, GpuConvolution},
    tasks::gaussian_blur::GpuGaussianBlur,
    tasks::hash_common::PerceptualHashComputer,
    tasks::mean_hash::MeanHashComputer,
    GpuContext,
};

mod common;
use common::test_data;

// === CPU 参考实现 ===

/// CPU 2D 卷积参考实现（不可分离）。
fn cpu_convolve_2d(
    pixels: &[u8],
    width: u32,
    height: u32,
    kernel: &[f32],
    kernel_size: u32,
    border_mode: BorderMode,
) -> Vec<u8> {
    let radius = (kernel_size / 2) as i32;
    let mut output = vec![0u8; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for ky in -radius..=radius {
                for kx in -radius..=radius {
                    let sx = apply_border(x as i32 + kx, width as i32, border_mode);
                    let sy = apply_border(y as i32 + ky, height as i32, border_mode);
                    // BorderMode::Zero 时越界坐标返回 -1，像素值视为 0
                    let pixel = if sx < 0 || sy < 0 {
                        0.0f32
                    } else {
                        pixels[(sy * width as i32 + sx) as usize] as f32
                    };
                    let k = kernel[((ky + radius) * kernel_size as i32 + (kx + radius)) as usize];
                    sum += pixel * k;
                }
            }
            output[(y * width + x) as usize] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }
    output
}

/// CPU 可分离卷积参考实现。
fn cpu_convolve_separable(
    pixels: &[u8],
    width: u32,
    height: u32,
    kernel_1d: &[f32],
    kernel_size: u32,
    border_mode: BorderMode,
) -> Vec<u8> {
    let radius = (kernel_size / 2) as i32;
    let pixel_count = (width * height) as usize;

    // 水平 1D
    let mut intermediate = vec![0.0f32; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for k in -radius..=radius {
                let sx = apply_border(x as i32 + k, width as i32, border_mode);
                // BorderMode::Zero 时越界坐标返回 -1，像素值视为 0
                let pixel = if sx < 0 {
                    0.0f32
                } else {
                    pixels[(y as i32 * width as i32 + sx) as usize] as f32
                };
                sum += pixel * kernel_1d[(k + radius) as usize];
            }
            intermediate[(y * width + x) as usize] = sum;
        }
    }

    // 垂直 1D
    let mut output = vec![0u8; pixel_count];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for k in -radius..=radius {
                let sy = apply_border(y as i32 + k, height as i32, border_mode);
                // BorderMode::Zero 时越界坐标返回 -1，像素值视为 0
                let pixel = if sy < 0 {
                    0.0f32
                } else {
                    intermediate[(sy * width as i32 + x as i32) as usize]
                };
                sum += pixel * kernel_1d[(k + radius) as usize];
            }
            output[(y * width + x) as usize] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }
    output
}

/// CPU 高斯模糊参考实现。
fn cpu_gaussian_blur(
    pixels: &[u8],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
) -> Vec<u8> {
    let kernel_1d = generate_gaussian_kernel_1d(kernel_size, sigma);
    cpu_convolve_separable(pixels, width, height, &kernel_1d, kernel_size, BorderMode::Clamp)
}

fn generate_gaussian_kernel_1d(kernel_size: u32, sigma: f32) -> Vec<f32> {
    let sigma = if sigma <= 0.0 {
        0.3 * ((kernel_size as f32 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };
    let radius = (kernel_size / 2) as i32;
    let mut kernel = Vec::with_capacity(kernel_size as usize);
    let mut sum = 0.0f32;
    for x in -radius..=radius {
        let val = (-(x as f32).powi(2) / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// 边界坐标映射。
/// BorderMode::Zero 时越界返回 -1，调用者需将对应像素值视为 0。
fn apply_border(coord: i32, size: i32, mode: BorderMode) -> i32 {
    if coord >= 0 && coord < size {
        return coord;
    }
    match mode {
        BorderMode::Zero => -1,
        BorderMode::Clamp => coord.clamp(0, size - 1),
        BorderMode::Reflect => {
            let period = 2 * size;
            let mut c = coord % period;
            if c < 0 {
                c += period;
            }
            if c >= size {
                c = period - 1 - c;
            }
            c
        }
    }
}

// === 测试用例 ===

#[test]
fn test_separable_convolution_clamp() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let conv = GpuConvolution::new(&mut ctx).expect("创建失败");

    let width = 32u32;
    let height = 32u32;
    let pixels = test_data::gradient_image((width * height) as usize);
    // 近似高斯 5 点核
    let kernel_1d = [0.0625f32, 0.25f32, 0.375f32, 0.25f32, 0.0625f32];
    let kernel_size = 5u32;

    let gpu_result = conv
        .convolve_separable(&ctx, &pixels, width, height, &kernel_1d, kernel_size, BorderMode::Clamp)
        .expect("GPU 卷积失败");

    let cpu_result = cpu_convolve_separable(&pixels, width, height, &kernel_1d, kernel_size, BorderMode::Clamp);

    // 允许 ±1 灰度值误差（GPU f32 精度差异）
    for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        let diff = (*g as i32 - *c as i32).abs();
        assert!(
            diff <= 1,
            "可分离卷积 Clamp 模式: 像素 {} 不匹配 (gpu={}, cpu={}, diff={})",
            i, g, c, diff
        );
    }
}

#[test]
fn test_full_2d_convolution() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let conv = GpuConvolution::new(&mut ctx).expect("创建失败");

    let width = 16u32;
    let height = 16u32;
    let pixels = test_data::random_image((width * height) as usize);

    // 3×3 均值核
    let kernel_2d = [1.0f32 / 9.0; 9];
    let kernel_size = 3u32;

    let gpu_result = conv
        .convolve_2d(&ctx, &pixels, width, height, &kernel_2d, kernel_size, BorderMode::Clamp)
        .expect("GPU 卷积失败");

    let cpu_result = cpu_convolve_2d(&pixels, width, height, &kernel_2d, kernel_size, BorderMode::Clamp);

    for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        let diff = (*g as i32 - *c as i32).abs();
        assert!(
            diff <= 1,
            "2D 卷积: 像素 {} 不匹配 (gpu={}, cpu={}, diff={})",
            i, g, c, diff
        );
    }
}

#[test]
fn test_border_modes() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let conv = GpuConvolution::new(&mut ctx).expect("创建失败");

    let width = 8u32;
    let height = 8u32;
    let pixels = test_data::gradient_image((width * height) as usize);
    let kernel_1d = [0.25f32, 0.5f32, 0.25f32];
    let kernel_size = 3u32;

    for mode in [BorderMode::Zero, BorderMode::Clamp, BorderMode::Reflect] {
        let gpu_result = conv
            .convolve_separable(&ctx, &pixels, width, height, &kernel_1d, kernel_size, mode)
            .expect("GPU 卷积失败");

        let cpu_result = cpu_convolve_separable(&pixels, width, height, &kernel_1d, kernel_size, mode);

        for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            let diff = (*g as i32 - *c as i32).abs();
            assert!(
                diff <= 1,
                "边界模式 {:?}: 像素 {} 不匹配 (gpu={}, cpu={}, diff={})",
                mode, i, g, c, diff
            );
        }
    }
}

#[test]
fn test_gaussian_blur() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let blur = GpuGaussianBlur::new(&mut ctx).expect("创建失败");

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::random_image((width * height) as usize);

    let gpu_result = blur.blur(&ctx, &pixels, width, height, 5, 1.5).expect("GPU 高斯模糊失败");
    let cpu_result = cpu_gaussian_blur(&pixels, width, height, 5, 1.5);

    for (i, (g, c)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
        let diff = (*g as i32 - *c as i32).abs();
        assert!(
            diff <= 2,
            "高斯模糊: 像素 {} 不匹配 (gpu={}, cpu={}, diff={})",
            i, g, c, diff
        );
    }
}

#[test]
fn test_convolution_to_hash_pipeline() {
    // 卷积 → 哈希流水线测试：验证 GPU 模糊和 CPU 模糊后哈希结果一致
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let blur = GpuGaussianBlur::new(&mut ctx).expect("创建失败");
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);

    // CPU 路径：先高斯模糊，再计算哈希
    let cpu_blurred = cpu_gaussian_blur(&pixels, width, height, 5, 1.5);
    let cpu_hash = hasher.compute(&ctx, &[cpu_blurred]).expect("计算失败")[0];

    // GPU 路径：先高斯模糊，再计算哈希
    let gpu_blurred = blur.blur(&ctx, &pixels, width, height, 5, 1.5).expect("GPU 模糊失败");
    let gpu_hash = hasher.compute(&ctx, &[gpu_blurred]).expect("计算失败")[0];

    assert_eq!(cpu_hash, gpu_hash, "卷积→哈希流水线: CPU 和 GPU 路径结果不一致");
}
