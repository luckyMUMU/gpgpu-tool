//! GPU Lanczos3 缩放一致性测试。
//!
//! 验证 GPU Lanczos3（2-pass 可分离卷积）与 CPU Lanczos3（`image` crate）
//! 的缩放结果一致性，以及 GPU Lanczos3 + 感知哈希的端到端正确性。

use gpgpu_tool::GpuContext;
use gpgpu_tool::ResizeFilter;
use gpgpu_tool::tasks::gpu_resize::{GpuResize, GpuResizeConfig};
use gpgpu_tool::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use gpgpu_tool::tasks::hash_common::HashSize;

// ═════════════════════════════════════════════════════════════════
// 辅助函数
// ═════════════════════════════════════════════════════════════════

/// 创建渐变测试图像（水平+垂直渐变混合）。
fn make_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let v = ((x * 255 / width.max(1)) + (y * 255 / height.max(1))) / 2;
            pixels.push(v.min(255) as u8);
        }
    }
    pixels
}

/// 创建棋盘格测试图像。
fn make_checkerboard_image(width: u32, height: u32, cell_size: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let cell_x = x / cell_size.max(1);
            let cell_y = y / cell_size.max(1);
            let v = if (cell_x + cell_y) % 2 == 0 { 200 } else { 50 };
            pixels.push(v);
        }
    }
    pixels
}

/// 创建随机噪声测试图像（确定性种子）。
fn make_noise_image(width: u32, height: u32, seed: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    let mut state = seed;
    for _ in 0..(width * height) {
        // 简单 LCG 随机数
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        pixels.push((state >> 16) as u8);
    }
    pixels
}

/// 使用 CPU Lanczos3（`image` crate）缩放灰度图像。
fn cpu_lanczos3_resize(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> Vec<u8> {
    let img = image::GrayImage::from_raw(src_w, src_h, pixels.to_vec())
        .expect("无效的图像尺寸");
    let dyn_img = image::DynamicImage::ImageLuma8(img);
    let resized = dyn_img.resize_exact(
        dst_w,
        dst_h,
        image::imageops::FilterType::Lanczos3,
    );
    resized.to_luma8().into_raw()
}

/// 计算两个像素数组间的最大绝对误差和平均绝对误差。
fn compute_error(gpu: &[u8], cpu: &[u8]) -> (u32, f64) {
    assert_eq!(gpu.len(), cpu.len(), "像素数组长度不匹配");
    let mut max_err = 0u32;
    let mut sum_err = 0u64;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        let err = (*g as i32 - *c as i32).unsigned_abs();
        max_err = max_err.max(err);
        sum_err += err as u64;
    }
    let avg_err = sum_err as f64 / gpu.len() as f64;
    (max_err, avg_err)
}

// ═════════════════════════════════════════════════════════════════
// 像素级一致性测试
// ═════════════════════════════════════════════════════════════════

/// 测试 GPU Lanczos3 与 CPU Lanczos3 在渐变图像上的像素一致性。
#[test]
fn test_gpu_lanczos3_gradient_pixel_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 128u32;
    let src_h = 128u32;
    let dst_w = 9u32;  // Gradient hash 目标宽度
    let dst_h = 8u32;  // Gradient hash 目标高度

    let pixels = make_gradient_image(src_w, src_h);
    let images = vec![pixels.clone()];
    let dims = vec![(src_w, src_h)];

    // GPU Lanczos3 缩放
    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();
    let gpu_result = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();

    // CPU Lanczos3 缩放
    let cpu_result = cpu_lanczos3_resize(&pixels, src_w, src_h, dst_w, dst_h);

    let (max_err, avg_err) = compute_error(&gpu_result[0], &cpu_result);
    println!("渐变图像: GPU vs CPU Lanczos3 — 最大误差={}, 平均误差={:.2}", max_err, avg_err);

    // Lanczos3 GPU/CPU 应该非常接近（f32 vs f64 精度差异）
    assert!(max_err <= 5, "最大像素误差 {} 超过阈值 5", max_err);
    assert!(avg_err <= 1.5, "平均像素误差 {:.2} 超过阈值 1.5", avg_err);
}

/// 测试 GPU Lanczos3 与 CPU Lanczos3 在棋盘格图像上的像素一致性。
#[test]
fn test_gpu_lanczos3_checkerboard_pixel_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 64u32;
    let src_h = 64u32;
    let dst_w = 16u32;
    let dst_h = 16u32;

    let pixels = make_checkerboard_image(src_w, src_h, 8);
    let images = vec![pixels.clone()];
    let dims = vec![(src_w, src_h)];

    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();
    let gpu_result = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();

    let cpu_result = cpu_lanczos3_resize(&pixels, src_w, src_h, dst_w, dst_h);

    let (max_err, avg_err) = compute_error(&gpu_result[0], &cpu_result);
    println!("棋盘格: GPU vs CPU Lanczos3 — 最大误差={}, 平均误差={:.2}", max_err, avg_err);

    // 棋盘格高频内容，Lanczos3 可能有振铃效应，允许稍大误差
    assert!(max_err <= 10, "最大像素误差 {} 超过阈值 10", max_err);
    assert!(avg_err < 3.0, "平均像素误差 {:.2} 超过阈值 3.0", avg_err);
}

/// 测试 GPU Lanczos3 与 CPU Lanczos3 在噪声图像上的像素一致性。
#[test]
fn test_gpu_lanczos3_noise_pixel_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 256u32;
    let src_h = 256u32;
    let dst_w = 32u32;
    let dst_h = 32u32;

    let pixels = make_noise_image(src_w, src_h, 42);
    let images = vec![pixels.clone()];
    let dims = vec![(src_w, src_h)];

    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();
    let gpu_result = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();

    let cpu_result = cpu_lanczos3_resize(&pixels, src_w, src_h, dst_w, dst_h);

    let (max_err, avg_err) = compute_error(&gpu_result[0], &cpu_result);
    println!("噪声图像: GPU vs CPU Lanczos3 — 最大误差={}, 平均误差={:.2}", max_err, avg_err);

    assert!(max_err <= 5, "最大像素误差 {} 超过阈值 5", max_err);
    assert!(avg_err <= 1.5, "平均像素误差 {:.2} 超过阈值 1.5", avg_err);
}

// ═════════════════════════════════════════════════════════════════
// 批量处理测试
// ═════════════════════════════════════════════════════════════════

/// 测试 GPU Lanczos3 批量缩放多张图像的一致性。
#[test]
fn test_gpu_lanczos3_batch_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 128u32;
    let src_h = 128u32;
    let dst_w = 16u32;
    let dst_h = 16u32;

    // 创建 4 张不同种子的噪声图像
    let images: Vec<Vec<u8>> = (0..4u32)
        .map(|seed| make_noise_image(src_w, src_h, seed * 1000 + 1))
        .collect();
    let dims = vec![(src_w, src_h); 4];

    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();
    let gpu_results = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();

    assert_eq!(gpu_results.len(), 4, "批量结果数量不匹配");

    // 逐张与 CPU 对比
    for (i, (gpu_img, src_img)) in gpu_results.iter().zip(images.iter()).enumerate() {
        let cpu_img = cpu_lanczos3_resize(src_img, src_w, src_h, dst_w, dst_h);
        let (max_err, avg_err) = compute_error(gpu_img, &cpu_img);
        println!("批量图像 {}: 最大误差={}, 平均误差={:.2}", i, max_err, avg_err);
        assert!(max_err <= 5, "图像 {} 最大像素误差 {} 超过阈值 5", i, max_err);
    }
}

// ═════════════════════════════════════════════════════════════════
// 哈希一致性测试
// ═════════════════════════════════════════════════════════════════

/// 测试 GPU Lanczos3 + Gradient Hash 与 CPU Lanczos3 + Gradient Hash 的哈希一致性。
#[test]
fn test_gpu_lanczos3_gradient_hash_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 128u32;
    let src_h = 128u32;

    // 创建测试图像
    let images = vec![
        make_gradient_image(src_w, src_h),
        make_checkerboard_image(src_w, src_h, 8),
        make_noise_image(src_w, src_h, 42),
    ];
    let dims = vec![(src_w, src_h); 3];

    // GPU Lanczos3 hasher（通过 with_lanczos3 便捷构造）
    let hasher_gpu = PerceptualHasher::with_lanczos3(&mut ctx, HashAlgorithm::Gradient).unwrap();
    let hashes_gpu = hasher_gpu.compute(&ctx, &images, &dims).unwrap();

    // CPU Lanczos3 hasher（默认，不启用 GPU 缩放，compute_images 用 CPU Lanczos3）
    // 但 compute() 方法使用 CPU box filter，不是 CPU Lanczos3
    // 为了对比 CPU Lanczos3，需要用 compute_images (需要 image feature) 或手动流程
    // 这里用 GpuResize + compute_resized 模拟
    let hasher_cpu = PerceptualHasher::new(&mut ctx, HashAlgorithm::Gradient).unwrap();

    // CPU Lanczos3 缩放后计算哈希
    let resized_cpu: Vec<Vec<u8>> = images.iter()
        .map(|img| cpu_lanczos3_resize(img, src_w, src_h, hasher_cpu.target_size().0, hasher_cpu.target_size().1))
        .collect();
    let hashes_cpu = hasher_cpu.compute_resized(&ctx, &resized_cpu).unwrap();

    // 对比哈希
    assert_eq!(hashes_gpu.len(), hashes_cpu.len(), "哈希数量不匹配");
    for (i, (h_gpu, h_cpu)) in hashes_gpu.iter().zip(hashes_cpu.iter()).enumerate() {
        let dist = (h_gpu ^ h_cpu).count_ones();
        println!("图像 {}: GPU Lanczos3 hash={:#018x}, CPU Lanczos3 hash={:#018x}, Hamming 距离={}",
                 i, h_gpu, h_cpu, dist);
        // 由于 GPU/CPU 精度差异和灰度-缩放顺序差异，允许少量 bit 差异
        assert!(dist <= 20, "图像 {} 哈希 Hamming 距离 {} 超过阈值 20", i, dist);
    }
}

/// 测试 GPU Lanczos3 + Mean Hash（32×32 = 1024-bit）的一致性。
#[test]
fn test_gpu_lanczos3_mean_hash_1024bit_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 256u32;
    let src_h = 256u32;
    let hash_size = HashSize::new(32); // 32×32 = 1024-bit

    let images = vec![
        make_gradient_image(src_w, src_h),
        make_noise_image(src_w, src_h, 99),
    ];
    let dims = vec![(src_w, src_h); 2];

    // GPU Lanczos3 + Mean Hash
    let hasher_gpu = PerceptualHasher::with_lanczos3_and_hash_size(
        &mut ctx, HashAlgorithm::Mean, hash_size,
    ).unwrap();
    let hashes_gpu = hasher_gpu.compute(&ctx, &images, &dims).unwrap();

    // CPU Lanczos3 + Mean Hash
    let hasher_cpu = PerceptualHasher::with_hash_size(&mut ctx, HashAlgorithm::Mean, hash_size).unwrap();
    let resized_cpu: Vec<Vec<u8>> = images.iter()
        .map(|img| cpu_lanczos3_resize(img, src_w, src_h, hasher_cpu.target_size().0, hasher_cpu.target_size().1))
        .collect();
    let hashes_cpu = hasher_cpu.compute_resized(&ctx, &resized_cpu).unwrap();

    assert_eq!(hashes_gpu.len(), hashes_cpu.len());
    // 1024-bit 哈希 = 16 个 u64
    for i in 0..images.len() {
        let gpu_chunk = &hashes_gpu[i * 16..(i + 1) * 16];
        let cpu_chunk = &hashes_cpu[i * 16..(i + 1) * 16];
        let total_dist: u32 = gpu_chunk.iter()
            .zip(cpu_chunk.iter())
            .map(|(g, c)| (g ^ c).count_ones())
            .sum();
        println!("图像 {}: 1024-bit Mean Hash Hamming 距离={}/1024", i, total_dist);
        // 允许 2% 的 bit 差异
        assert!(total_dist <= 30, "图像 {} 哈希 Hamming 距离 {} 超过阈值 30", i, total_dist);
    }
}

/// 测试 GPU Lanczos3 + Block Hash（16×16 = 256-bit）的一致性。
#[test]
fn test_gpu_lanczos3_block_hash_256bit_consistency() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 128u32;
    let src_h = 128u32;
    let hash_size = HashSize::new(16); // 16×16 = 256-bit

    let images = vec![
        make_gradient_image(src_w, src_h),
        make_checkerboard_image(src_w, src_h, 16),
        make_noise_image(src_w, src_h, 7),
    ];
    let dims = vec![(src_w, src_h); 3];

    // GPU Lanczos3 + Block Hash
    let hasher_gpu = PerceptualHasher::with_lanczos3_and_hash_size(
        &mut ctx, HashAlgorithm::Block, hash_size,
    ).unwrap();
    let hashes_gpu = hasher_gpu.compute(&ctx, &images, &dims).unwrap();

    // CPU Lanczos3 + Block Hash
    let hasher_cpu = PerceptualHasher::with_hash_size(&mut ctx, HashAlgorithm::Block, hash_size).unwrap();
    let resized_cpu: Vec<Vec<u8>> = images.iter()
        .map(|img| cpu_lanczos3_resize(img, src_w, src_h, hasher_cpu.target_size().0, hasher_cpu.target_size().1))
        .collect();
    let hashes_cpu = hasher_cpu.compute_resized(&ctx, &resized_cpu).unwrap();

    assert_eq!(hashes_gpu.len(), hashes_cpu.len());
    // 256-bit 哈希 = 4 个 u64
    for i in 0..images.len() {
        let gpu_chunk = &hashes_gpu[i * 4..(i + 1) * 4];
        let cpu_chunk = &hashes_cpu[i * 4..(i + 1) * 4];
        let total_dist: u32 = gpu_chunk.iter()
            .zip(cpu_chunk.iter())
            .map(|(g, c)| (g ^ c).count_ones())
            .sum();
        println!("图像 {}: 256-bit Block Hash Hamming 距离={}/256", i, total_dist);
        assert!(total_dist <= 20, "图像 {} 哈希 Hamming 距离 {} 超过阈值 20", i, total_dist);
    }
}

// ═════════════════════════════════════════════════════════════════
// 不同缩放比例测试
// ═════════════════════════════════════════════════════════════════

/// 测试不同缩放比例下 GPU Lanczos3 的正确性。
#[test]
fn test_gpu_lanczos3_various_scales() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();

    let src_w = 128u32;
    let src_h = 128u32;
    let pixels = make_gradient_image(src_w, src_h);

    // 测试多种目标尺寸
    let targets = [
        (8, 8),    // 大幅下采样
        (9, 8),    // Gradient hash 尺寸
        (16, 16),  // Block hash 尺寸
        (32, 32),  // 中等尺寸
        (64, 64),  // 较大尺寸
    ];

    for (dst_w, dst_h) in targets {
        let images = vec![pixels.clone()];
        let dims = vec![(src_w, src_h)];

        let gpu_result = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();
        let cpu_result = cpu_lanczos3_resize(&pixels, src_w, src_h, dst_w, dst_h);

        let (max_err, avg_err) = compute_error(&gpu_result[0], &cpu_result);
        println!("缩放 {}x{} → {}x{}: 最大误差={}, 平均误差={:.2}",
                 src_w, src_h, dst_w, dst_h, max_err, avg_err);

        assert!(max_err <= 5, "缩放到 {}x{} 最大像素误差 {} 超过阈值 5", dst_w, dst_h, max_err);
        assert!(avg_err <= 1.5, "缩放到 {}x{} 平均像素误差 {:.2} 超过阈值 1.5", dst_w, dst_h, avg_err);
    }
}

/// 测试上采样场景下 GPU Lanczos3 的正确性。
#[test]
fn test_gpu_lanczos3_upsampling() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let src_w = 16u32;
    let src_h = 16u32;
    let dst_w = 64u32;
    let dst_h = 64u32;

    let pixels = make_gradient_image(src_w, src_h);
    let images = vec![pixels.clone()];
    let dims = vec![(src_w, src_h)];

    let gpu_resize = GpuResize::with_config(
        &mut ctx,
        GpuResizeConfig::default().filter(ResizeFilter::Lanczos3),
    ).unwrap();
    let gpu_result = gpu_resize.resize_batch(&ctx, &images, &dims, dst_w, dst_h).unwrap();

    let cpu_result = cpu_lanczos3_resize(&pixels, src_w, src_h, dst_w, dst_h);

    let (max_err, avg_err) = compute_error(&gpu_result[0], &cpu_result);
    println!("上采样 {}x{} → {}x{}: 最大误差={}, 平均误差={:.2}",
             src_w, src_h, dst_w, dst_h, max_err, avg_err);

    assert!(max_err <= 5, "上采样最大像素误差 {} 超过阈值 5", max_err);
    assert!(avg_err <= 1.5, "上采样平均像素误差 {:.2} 超过阈值 1.5", avg_err);
}
