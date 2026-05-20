use std::fs;
use std::path::Path;
use std::time::Instant;

use image::GenericImageView;
use wgpu_compute_engine::GpuContext;
use wgpu_compute_engine::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use wgpu_compute_engine::tasks::sha256::Sha256Computer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

/// 加载目录下所有图像为 DynamicImage
fn load_all_images(dir: &str) -> Vec<image::DynamicImage> {
    let mut results = Vec::new();
    let path = Path::new(dir);
    if !path.exists() { return results; }

    fn walk(dir: &Path, results: &mut Vec<image::DynamicImage>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, results);
                } else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) {
                            results.push(img);
                        }
                    }
                }
            }
        }
    }

    walk(path, &mut results);
    results
}

/// 将 DynamicImage 转为指定尺寸的灰度像素 + 尺寸信息
fn to_grayscale_pixels(img: &image::DynamicImage, w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let resized = img.resize_exact(w, h, image::imageops::FilterType::Lanczos3);
    let gray: Vec<u8> = resized.grayscale().pixels().map(|(_, _, luma)| luma.0[0]).collect();
    (gray, w, h)
}

/// CPU 参考实现：Mean Hash
fn cpu_mean_hash(pixels: &[u8], _w: u32, _h: u32) -> u64 {
    let sum: f32 = pixels.iter().map(|&p| p as f32).sum();
    let mean = sum / pixels.len() as f32;
    let mut hash: u64 = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel as f32 >= mean {
            hash |= 1u64 << i;
        }
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

    for &size in &[64, 128, 256, 512] {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
        let (tw, th) = hasher.target_size();

        let mut pixel_data = Vec::new();
        let mut dimensions = Vec::new();
        let mut resized_data = Vec::new();

        for img in &images {
            let (gray, w, h) = to_grayscale_pixels(img, size, size);
            pixel_data.push(gray.clone());
            dimensions.push((w, h));
            let (resized, _, _) = to_grayscale_pixels(img, tw, th);
            resized_data.push(resized);
        }

        // GPU: 通过 PerceptualHasher 自动缩放
        let gpu_hashes = hasher.compute(&ctx, &pixel_data, &dimensions).unwrap();

        // CPU: 手动缩放后计算
        let mut all_pass = true;
        for (i, resized) in resized_data.iter().enumerate() {
            let cpu_hash = cpu_mean_hash(resized, tw, th);
            let gpu_hash = gpu_hashes[i];
            if gpu_hash != cpu_hash {
                all_pass = false;
                println!("  ✗ {}x{} 图像 {} GPU={:016x} CPU={:016x}", size, size, i, gpu_hash, cpu_hash);
            }
        }

        println!("  {}x{} → 缩放到 {}x{}: {} ({})", size, size, tw, th,
            if all_pass { "✓ 全部匹配" } else { "✗ 不匹配" },
            images.len());
        assert!(all_pass, "{}x{} Mean Hash GPU/CPU 不一致", size, size);
    }
}

#[test]
fn test_large_image_hash_performance() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let images = load_all_images(DATA_DIR);
    if images.is_empty() { return; }

    let iterations = 20;

    println!("\n=== 大图感知哈希性能测试 ({} 张真实图像) ===", images.len());
    println!();

    // 测试不同原始图像尺寸
    for &src_size in &[64, 128, 256, 512, 1024] {
        let pixel_data: Vec<Vec<u8>> = images.iter()
            .map(|img| to_grayscale_pixels(img, src_size, src_size).0)
            .collect();
        let dimensions: Vec<(u32, u32)> = vec![(src_size, src_size); images.len()];

        // GPU Mean Hash（含 CPU 缩放）
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = hasher.compute(&ctx, &pixel_data, &dimensions).unwrap();
        }
        let gpu_elapsed = start.elapsed() / iterations;

        // CPU Mean Hash（含 CPU 缩放）
        let (tw, th) = hasher.target_size();
        let start = Instant::now();
        for _ in 0..iterations {
            for pixels in &pixel_data {
                let resized = resize_grayscale_simple(pixels, src_size, src_size, tw, th);
                let _ = cpu_mean_hash(&resized, tw, th);
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

    // 通过复制生成大批量
    let mut large_batch_256: Vec<Vec<u8>> = Vec::new();
    for img in &images {
        let (gray, _, _) = to_grayscale_pixels(img, 256, 256);
        for _ in 0..20 {
            large_batch_256.push(gray.clone());
        }
    }

    let dimensions_256: Vec<(u32, u32)> = vec![(256, 256); large_batch_256.len()];

    println!("\n=== 大批量性能测试 ({} 张 256x256 图像) ===", large_batch_256.len());

    // GPU 各算法
    for algo in &[HashAlgorithm::Mean, HashAlgorithm::Median, HashAlgorithm::Gradient, HashAlgorithm::Block] {
        let hasher = PerceptualHasher::new(&mut ctx, *algo).unwrap();
        let start = Instant::now();
        let _ = hasher.compute(&ctx, &large_batch_256, &dimensions_256).unwrap();
        let elapsed = start.elapsed();

        let (tw, th) = hasher.target_size();
        println!("  {:?} (→ {}x{}): {:?} ({:.1} μs/img)",
            algo, tw, th, elapsed, elapsed.as_micros() as f64 / large_batch_256.len() as f64);
    }

    // CPU Mean Hash 对比
    let start = Instant::now();
    for pixels in &large_batch_256 {
        let resized = resize_grayscale_simple(pixels, 256, 256, 8, 8);
        let _ = cpu_mean_hash(&resized, 8, 8);
    }
    let cpu_elapsed = start.elapsed();
    println!("  CPU Mean (→ 8x8): {:?} ({:.1} μs/img)",
        cpu_elapsed, cpu_elapsed.as_micros() as f64 / large_batch_256.len() as f64);

    // SHA-256 大图批量
    let sha256 = Sha256Computer::new(&mut ctx).unwrap();

    // 同步
    let start = Instant::now();
    let _ = sha256.compute(&ctx, &large_batch_256).unwrap();
    let sync_elapsed = start.elapsed();
    println!("\n  SHA-256 同步: {:?} ({:.1} μs/img)", sync_elapsed, sync_elapsed.as_micros() as f64 / large_batch_256.len() as f64);

    // 异步批量
    let start = Instant::now();
    let mut submitter = sha256.batch_submitter(&ctx);
    submitter.submit(&large_batch_256).unwrap();
    let _ = submitter.wait_all().unwrap();
    let async_elapsed = start.elapsed();
    println!("  SHA-256 异步: {:?} ({:.1} μs/img) ({:.1}x vs 同步)", async_elapsed,
        async_elapsed.as_micros() as f64 / large_batch_256.len() as f64,
        sync_elapsed.as_secs_f64() / async_elapsed.as_secs_f64());
}

/// 简单盒式下采样（与 phasher.rs 中的实现一致）
fn resize_grayscale_simple(pixels: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> Vec<u8> {
    if src_w == dst_w && src_h == dst_h {
        return pixels.to_vec();
    }
    let mut output = Vec::with_capacity((dst_w * dst_h) as usize);
    let x_ratio = src_w as f64 / dst_w as f64;
    let y_ratio = src_h as f64 / dst_h as f64;
    for dy in 0..dst_h {
        let src_y_start = (dy as f64 * y_ratio) as u32;
        let src_y_end = ((dy + 1) as f64 * y_ratio).min(src_h as f64) as u32;
        let y_count = (src_y_end - src_y_start).max(1);
        for dx in 0..dst_w {
            let src_x_start = (dx as f64 * x_ratio) as u32;
            let src_x_end = ((dx + 1) as f64 * x_ratio).min(src_w as f64) as u32;
            let x_count = (src_x_end - src_x_start).max(1);
            let mut sum: u32 = 0;
            for sy in src_y_start..src_y_end {
                for sx in src_x_start..src_x_end {
                    sum += pixels[(sy * src_w + sx) as usize] as u32;
                }
            }
            output.push((sum / (x_count * y_count)) as u8);
        }
    }
    output
}