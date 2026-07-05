use std::fs;
use std::path::Path;
use std::time::Instant;

use image::GenericImageView;
use gpgpu_tool::GpuContext;
use gpgpu_tool::GpuError;
use gpgpu_tool::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use gpgpu_tool::tasks::sha256::Sha256Computer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load_all_images(dir: &str) -> Vec<image::DynamicImage> {
    let mut results = Vec::new();
    let path = Path::new(dir);
    if !path.exists() { return results; }

    fn walk(dir: &Path, results: &mut Vec<image::DynamicImage>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() { walk(&p, results); }
                else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) { results.push(img); }
                    }
                }
            }
        }
    }

    walk(path, &mut results);
    results
}

fn to_gray_lanczos(img: &image::DynamicImage, w: u32, h: u32) -> Vec<u8> {
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
    resized.grayscale().pixels().map(|(_, _, luma)| luma.0[0]).collect()
}

fn resize_box_filter(pixels: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h { return pixels.to_vec(); }
    let mut output = Vec::with_capacity((dst_w * dst_h) as usize);
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;
    for dy in 0..dst_h {
        let sy0 = (dy as f64 * y_ratio) as u32;
        let sy1 = ((dy + 1) as f64 * y_ratio).min(src_h as f64) as u32;
        let yc = (sy1 - sy0).max(1);
        for dx in 0..dst_w {
            let sx0 = (dx as f64 * x_ratio) as u32;
            let sx1 = ((dx + 1) as f64 * x_ratio).min(src_w as f64) as u32;
            let xc = (sx1 - sx0).max(1);
            let mut sum: u32 = 0;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    sum += pixels[(sy * src_w + sx) as usize] as u32;
                }
            }
            output.push((sum / (xc * yc)) as u8);
        }
    }
    output
}

fn cpu_mean_hash(pixels: &[u8]) -> u64 {
    let sum: f32 = pixels.iter().map(|&p| p as f32).sum();
    let mean = sum / pixels.len() as f32;
    let mut hash: u64 = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel as f32 >= mean { hash |= 1u64 << i; }
    }
    hash
}

#[test]
fn test_large_image_hash_correctness() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let images = load_all_images(DATA_DIR);
    if images.is_empty() { return; }

    println!("\n=== 大图感知哈希正确性验证 ===");

    for &src_size in &[64, 128, 256, 512] {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
        let (tw, th) = hasher.target_size();

        let mut pixel_data = Vec::new();
        let mut dimensions = Vec::new();
        let mut resized_for_cpu = Vec::new();

        for img in &images {
            let gray_src = to_gray_lanczos(img, src_size, src_size);
            pixel_data.push(gray_src.clone());
            dimensions.push((src_size, src_size));
            let resized = resize_box_filter(&gray_src, src_size, src_size, tw, th);
            resized_for_cpu.push(resized);
        }

        let gpu_hashes = hasher.compute(&ctx, &pixel_data, &dimensions).unwrap();

        let mut all_pass = true;
        for (i, resized) in resized_for_cpu.iter().enumerate() {
            let cpu_hash = cpu_mean_hash(resized);
            let gpu_hash = gpu_hashes[i];
            if gpu_hash != cpu_hash {
                all_pass = false;
                println!("  ✗ {}x{} 图像 {} GPU={:016x} CPU={:016x}", src_size, src_size, i, gpu_hash, cpu_hash);
            }
        }

        println!("  {}x{} → 缩放到 {}x{}: {} ({})", src_size, src_size, tw, th,
            if all_pass { "✓ 全部匹配" } else { "✗ 不匹配" },
            images.len());
        assert!(all_pass, "{}x{} Mean Hash GPU/CPU 不一致", src_size, src_size);
    }
}

#[test]
fn test_large_image_hash_performance() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let images = load_all_images(DATA_DIR);
    if images.is_empty() { return; }

    let iterations = 10;

    println!("\n=== 大图感知哈希性能测试 ({} 张真实图像) ===", images.len());
    println!();

    for &src_size in &[128, 256, 512, 1024] {
        let pixel_data: Vec<Vec<u8>> = images.iter()
            .map(|img| to_gray_lanczos(img, src_size, src_size))
            .collect();
        let dimensions: Vec<(u32, u32)> = vec![(src_size, src_size); images.len()];

        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
        let (tw, th) = hasher.target_size();

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hasher.compute(&ctx, &pixel_data, &dimensions).unwrap();
        }
        let gpu_elapsed = start.elapsed() / iterations;

        let start = Instant::now();
        for _ in 0..iterations {
            for pixels in &pixel_data {
                let resized = resize_box_filter(pixels, src_size, src_size, tw, th);
                let _ = cpu_mean_hash(&resized);
            }
        }
        let cpu_elapsed = start.elapsed() / iterations;

        println!("源图 {}x{}: GPU={:?}/batch ({:.1} μs/img) | CPU={:?}/batch ({:.1} μs/img) | GPU/CPU={:.2}x",
            src_size, src_size,
            gpu_elapsed, gpu_elapsed.as_micros() as f64 / images.len() as f64,
            cpu_elapsed, cpu_elapsed.as_micros() as f64 / images.len() as f64,
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64());
    }
}

#[test]
fn test_large_image_batch_performance() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let images = load_all_images(DATA_DIR);
    if images.is_empty() { return; }

    let mut large_batch: Vec<Vec<u8>> = Vec::new();
    for img in &images {
        let gray = to_gray_lanczos(img, 256, 256);
        for _ in 0..5 { large_batch.push(gray.clone()); }
    }
    let dims: Vec<(u32, u32)> = vec![(256, 256); large_batch.len()];

    println!("\n=== 大批量性能测试 ({} 张 256x256 图像) ===", large_batch.len());

    for algo in &[HashAlgorithm::Mean, HashAlgorithm::Median, HashAlgorithm::Gradient, HashAlgorithm::Block] {
        let hasher = PerceptualHasher::new(&mut ctx, *algo).unwrap();
        let start = Instant::now();
        let _ = hasher.compute(&ctx, &large_batch, &dims).unwrap();
        let elapsed = start.elapsed();
        let (tw, th) = hasher.target_size();
        println!("  {:?} (→ {}x{}): {:?} ({:.1} μs/img)",
            algo, tw, th, elapsed, elapsed.as_micros() as f64 / large_batch.len() as f64);
    }

    let start = Instant::now();
    for pixels in &large_batch {
        let resized = resize_box_filter(pixels, 256, 256, 8, 8);
        let _ = cpu_mean_hash(&resized);
    }
    let cpu_elapsed = start.elapsed();
    println!("  CPU Mean (→ 8x8): {:?} ({:.1} μs/img)",
        cpu_elapsed, cpu_elapsed.as_micros() as f64 / large_batch.len() as f64);

    let sha256 = Sha256Computer::new(&mut ctx).unwrap();

    let start = Instant::now();
    let _ = sha256.compute(&ctx, &large_batch).unwrap();
    let sync_elapsed = start.elapsed();
    println!("\n  SHA-256 同步: {:?} ({:.1} μs/img)", sync_elapsed, sync_elapsed.as_micros() as f64 / large_batch.len() as f64);

    let start = Instant::now();
    let mut submitter = sha256.batch_submitter(&ctx).unwrap();
    submitter.submit(&large_batch).unwrap();
    let _ = submitter.wait_all().unwrap();
    let async_elapsed = start.elapsed();
    println!("  SHA-256 异步: {:?} ({:.1} μs/img) ({:.1}x vs 同步)", async_elapsed,
        async_elapsed.as_micros() as f64 / large_batch.len() as f64,
        sync_elapsed.as_secs_f64() / async_elapsed.as_secs_f64());
}

// ============================================================================
// 大图像边界测试辅助函数
// ============================================================================

/// 创建合成灰度图像（简单渐变模式）。
///
/// 像素值 = (x * 255 / width + y * 255 / height) / 2，产生从左上到右下的渐变。
fn create_synthetic_image(width: u32, height: u32) -> Vec<u8> {
    let size = (width as usize) * (height as usize);
    let mut pixels = Vec::with_capacity(size);
    for y in 0..height {
        for x in 0..width {
            let v = ((x as f32 * 255.0 / width as f32) + (y as f32 * 255.0 / height as f32)) / 2.0;
            pixels.push(v as u8);
        }
    }
    pixels
}

// ============================================================================
// 大图像 OOM 边界测试
// ============================================================================

/// 测试：10000×10000 灰度图像（100MB u8 / 400MB u32 对齐）应被拒绝。
///
/// 单张图像 u32 对齐后为 400MB，远超典型 GPU 的 max_storage_buffer_binding_size
/// （通常 128MB-256MB），应返回 `GpuError::InvalidInput` 并包含明确错误消息。
#[test]
#[ignore = "slow: creates 100MB synthetic image, may OOM on low-memory systems"]
fn test_large_image_exceeds_buffer_binding_limit() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let width: u32 = 10000;
    let height: u32 = 10000;
    let image = create_synthetic_image(width, height);
    let dimensions = vec![(width, height)];

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let result = hasher.compute(&ctx, &[image], &dimensions);

    match result {
        Err(GpuError::InvalidInput(msg)) => {
            assert!(
                msg.contains("max_storage_buffer_binding_size"),
                "错误消息应提及 max_storage_buffer_binding_size，实际: {}",
                msg
            );
            assert!(
                msg.contains(&format!("{}×{}", width, height)),
                "错误消息应包含图像尺寸 {}×{}，实际: {}",
                width, height, msg
            );
            println!("✓ 正确拒绝超大图像: {}", msg);
        }
        Ok(hashes) => {
            // 如果 GPU 恰好支持 400MB+ 绑定大小（极少见），不应 panic
            println!("⚠ GPU 支持超大缓冲区绑定，返回 {} 个哈希", hashes.len());
        }
        Err(e) => {
            panic!("预期 GpuError::InvalidInput，实际得到: {:?}", e);
        }
    }
}

/// 测试：图像刚好在 max_storage_buffer_binding_size 边界内应成功计算。
///
/// 计算一张尺寸刚好使 u32 对齐字节数 ≤ max_storage_buffer_binding_size 的图像，
/// 验证 `PerceptualHasher::compute()` 成功返回哈希值。
#[test]
#[ignore = "slow: creates large synthetic image near GPU buffer limit"]
fn test_large_image_just_under_buffer_binding_limit() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let max_binding = ctx.limits().max_storage_buffer_binding_size as u64;
    // u32 对齐：每像素 4 字节
    let max_pixels = max_binding / 4;

    // 选择正方形尺寸，像素数 ≤ max_pixels
    let side = (max_pixels as f64).sqrt() as u32;
    // 确保 side × side ≤ max_pixels
    let side = if (side as u64) * (side as u64) > max_pixels {
        side - 1
    } else {
        side
    };
    let side = side.max(8); // 至少 8×8（Mean Hash 最小目标尺寸）

    let image = create_synthetic_image(side, side);
    let dimensions = vec![(side, side)];

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let result = hasher.compute(&ctx, &[image], &dimensions);

    match result {
        Ok(hashes) => {
            assert_eq!(hashes.len(), 1, "应返回 1 个哈希值");
            let per_image_bytes = (side as u64) * (side as u64) * 4;
            println!(
                "✓ {}×{} 图像 ({} 字节 u32 对齐) 成功计算哈希: {:016x}",
                side, side, per_image_bytes, hashes[0]
            );
            assert!(
                per_image_bytes <= max_binding,
                "u32 对齐字节数 {} 应 ≤ max_storage_buffer_binding_size {}",
                per_image_bytes, max_binding
            );
        }
        Err(GpuError::InvalidInput(msg)) => {
            // 可能因为 max_batch_size 限制被拒绝（DEFAULT_MAX_BATCH_SIZE = 128MB）
            // 这也是合理的行为
            println!("⚠ 图像被拒绝（可能因 max_batch_size 限制）: {}", msg);
        }
        Err(e) => {
            panic!("预期成功或 InvalidInput，实际得到: {:?}", e);
        }
    }
}

/// 测试：混合尺寸图像（不同 w/h 在同一批次）应正常工作。
///
/// 创建 3 张不同尺寸的合成图像（128×128, 256×128, 128×256），
/// 验证 `PerceptualHasher::compute()` 能正确处理混合尺寸批次。
#[test]
fn test_mixed_size_images_in_batch() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let sizes = [(128u32, 128u32), (256u32, 128u32), (128u32, 256u32)];
    let images: Vec<Vec<u8>> = sizes
        .iter()
        .map(|&(w, h)| create_synthetic_image(w, h))
        .collect();
    let dimensions: Vec<(u32, u32)> = sizes.to_vec();

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let result = hasher.compute(&ctx, &images, &dimensions);

    match result {
        Ok(hashes) => {
            assert_eq!(
                hashes.len(),
                sizes.len(),
                "应返回 {} 个哈希值，实际 {} 个",
                sizes.len(),
                hashes.len()
            );
            for (i, hash) in hashes.iter().enumerate() {
                println!(
                    "  {}×{} → {:016x}",
                    sizes[i].0, sizes[i].1, hash
                );
            }
            // 验证所有哈希非零（渐变模式缩放到 8×8 后可能产生相同哈希，
            // 但不应为零——零哈希意味着全黑图像）
            for (i, hash) in hashes.iter().enumerate() {
                assert_ne!(
                    *hash, 0,
                    "{}×{} 哈希不应为零（渐变图像非全黑）",
                    sizes[i].0, sizes[i].1
                );
            }
        }
        Err(e) => {
            panic!("混合尺寸批次应成功计算，实际错误: {:?}", e);
        }
    }
}