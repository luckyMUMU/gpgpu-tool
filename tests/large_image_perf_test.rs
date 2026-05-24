use std::fs;
use std::path::Path;
use std::time::Instant;

use image::GenericImageView;
use gpgpu_tool::GpuContext;
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
    let mut submitter = sha256.batch_submitter(&ctx);
    submitter.submit(&large_batch).unwrap();
    let _ = submitter.wait_all().unwrap();
    let async_elapsed = start.elapsed();
    println!("  SHA-256 异步: {:?} ({:.1} μs/img) ({:.1}x vs 同步)", async_elapsed,
        async_elapsed.as_micros() as f64 / large_batch.len() as f64,
        sync_elapsed.as_secs_f64() / async_elapsed.as_secs_f64());
}