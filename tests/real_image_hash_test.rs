use std::fs;
use std::path::Path;
use std::time::Instant;

use image::GenericImageView;
use gpgpu_tool::GpuContext;
use gpgpu_tool::tasks::hash_common::PerceptualHashComputer;
use gpgpu_tool::tasks::mean_hash::MeanHashComputer;
use gpgpu_tool::tasks::median_hash::MedianHashComputer;
use gpgpu_tool::tasks::gradient_hash::GradientHashComputer;
use gpgpu_tool::tasks::sha256::Sha256Computer;

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load_images_as_grayscale(dir: &str, target_w: u32, target_h: u32) -> Vec<(String, Vec<u8>)> {
    let mut results = Vec::new();
    let path = Path::new(dir);
    if !path.exists() {
        eprintln!("测试数据目录不存在: {}", dir);
        return results;
    }

    fn walk_dir(dir: &Path, results: &mut Vec<(String, Vec<u8>)>, tw: u32, th: u32) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk_dir(&p, results, tw, th);
                } else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) {
                            let resized = img.resize_exact(tw, th, image::imageops::FilterType::Lanczos3);
                            let luma_img = resized.grayscale();
                            let gray: Vec<u8> = luma_img.pixels()
                                .map(|(_, _, luma)| luma.0[0])
                                .collect();
                            let name = p.file_name().unwrap().to_string_lossy().to_string();
                            results.push((name, gray));
                        }
                    }
                }
            }
        }
    }

    walk_dir(path, &mut results, target_w, target_h);
    results
}

/// CPU 参考实现：Mean Hash（与 GPU WGSL 算法一致，使用 f32 均值）
fn cpu_mean_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
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

/// CPU 参考实现：Gradient Hash（水平梯度，与 GPU WGSL 算法一致）
fn cpu_gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let mut hash: u64 = 0;
    let mut bit = 0u32;
    for row in 0..height {
        for col in 0..(width - 1) {
            let idx = (row * width + col) as usize;
            let idx_next = (row * width + col + 1) as usize;
            if pixels[idx_next] > pixels[idx] {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }
    hash
}

/// CPU 参考实现：Median Hash（与 GPU WGSL 直方图中值算法一致）
fn cpu_median_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    let mut histogram = [0u32; 256];
    for &pixel in pixels {
        histogram[pixel as usize] += 1;
    }
    let half = pixels.len() as u32 / 2;
    let mut count = 0u32;
    let mut median = 128u32;
    for v in 0..256u32 {
        count += histogram[v as usize];
        if count > half {
            median = v;
            break;
        }
    }
    let mut hash: u64 = 0;
    for (i, &pixel) in pixels.iter().enumerate() {
        if pixel as u32 > median {
            hash |= 1u64 << i;
        }
    }
    hash
}

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[test]
fn test_real_image_mean_hash() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");
    let computer = MeanHashComputer::new(&mut ctx).expect("MeanHash 创建失败");

    let images = load_images_as_grayscale(DATA_DIR, 8, 8);
    if images.is_empty() {
        eprintln!("未找到测试图像，跳过");
        return;
    }

    let pixel_data: Vec<Vec<u8>> = images.iter().map(|(_, px)| px.clone()).collect();
    let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();

    let gpu_hashes = computer.compute(&ctx, &pixel_data).expect("GPU 计算失败");

    println!("\n=== Mean Hash (8x8) 真实图像验证 ===");
    println!("{:<30} {:<18} {:<18} {:<10}", "图像", "GPU Hash", "CPU Hash", "汉明距离");

    let mut all_pass = true;
    for (i, name) in names.iter().enumerate() {
        let cpu_hash = cpu_mean_hash(&pixel_data[i], 8, 8);
        let gpu_hash = gpu_hashes[i];
        let dist = hamming_distance(gpu_hash, cpu_hash);
        if dist > 0 { all_pass = false; }
        println!("{:<30} {:016x}   {:016x}   {} ({})", name, gpu_hash, cpu_hash, dist, if dist == 0 { "✓" } else { "✗" });
    }

    assert!(all_pass, "Mean Hash GPU/CPU 结果不一致");
}

#[test]
fn test_real_image_median_hash() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");
    let computer = MedianHashComputer::new(&mut ctx).expect("MedianHash 创建失败");

    let images = load_images_as_grayscale(DATA_DIR, 8, 8);
    if images.is_empty() { return; }

    let pixel_data: Vec<Vec<u8>> = images.iter().map(|(_, px)| px.clone()).collect();
    let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();

    let gpu_hashes = computer.compute(&ctx, &pixel_data).expect("GPU 计算失败");

    println!("\n=== Median Hash (8x8) 真实图像验证 ===");
    println!("{:<30} {:<18} {:<18} {:<10}", "图像", "GPU Hash", "CPU Hash", "汉明距离");

    let mut all_pass = true;
    for (i, name) in names.iter().enumerate() {
        let cpu_hash = cpu_median_hash(&pixel_data[i], 8, 8);
        let gpu_hash = gpu_hashes[i];
        let dist = hamming_distance(gpu_hash, cpu_hash);
        if dist > 0 { all_pass = false; }
        println!("{:<30} {:016x}   {:016x}   {} ({})", name, gpu_hash, cpu_hash, dist, if dist == 0 { "✓" } else { "✗" });
    }

    assert!(all_pass, "Median Hash GPU/CPU 结果不一致");
}

#[test]
fn test_real_image_gradient_hash() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");
    let computer = GradientHashComputer::new(&mut ctx).expect("GradientHash 创建失败");

    let images = load_images_as_grayscale(DATA_DIR, 8, 9);
    if images.is_empty() { return; }

    let pixel_data: Vec<Vec<u8>> = images.iter().map(|(_, px)| px.clone()).collect();
    let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();

    let gpu_hashes = computer.compute(&ctx, &pixel_data).expect("GPU 计算失败");

    println!("\n=== Gradient Hash (8x9) 真实图像验证 ===");
    println!("{:<30} {:<18} {:<18} {:<10}", "图像", "GPU Hash", "CPU Hash", "汉明距离");

    let mut all_pass = true;
    for (i, name) in names.iter().enumerate() {
        let cpu_hash = cpu_gradient_hash(&pixel_data[i], 8, 9);
        let gpu_hash = gpu_hashes[i];
        let dist = hamming_distance(gpu_hash, cpu_hash);
        if dist > 0 { all_pass = false; }
        println!("{:<30} {:016x}   {:016x}   {} ({})", name, gpu_hash, cpu_hash, dist, if dist == 0 { "✓" } else { "✗" });
    }

    assert!(all_pass, "Gradient Hash GPU/CPU 结果不一致");
}

#[test]
fn test_real_image_sha256() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("SHA256 创建失败");

    let images = load_images_as_grayscale(DATA_DIR, 8, 8);
    if images.is_empty() { return; }

    let messages: Vec<Vec<u8>> = images.iter().map(|(_, px)| px.clone()).collect();
    let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();

    let gpu_hashes = sha256.compute(&ctx, &messages).expect("GPU SHA-256 计算失败");

    use sha2::{Sha256, Digest};

    println!("\n=== SHA-256 真实图像验证 ===");
    println!("{:<30} {:<10}", "图像", "匹配");

    let mut all_pass = true;
    for (i, name) in names.iter().enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(&messages[i]);
        let cpu_hash = hasher.finalize();

        let match_result = gpu_hashes[i] == cpu_hash.as_slice();
        if !match_result { all_pass = false; }
        println!("{:<30} {}", name, if match_result { "✓" } else { "✗" });
    }

    assert!(all_pass, "SHA-256 GPU/CPU 结果不一致");
}

#[test]
fn test_real_image_hash_performance() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");

    let images_8x8 = load_images_as_grayscale(DATA_DIR, 8, 8);
    if images_8x8.is_empty() { return; }

    let pixel_data_8x8: Vec<Vec<u8>> = images_8x8.iter().map(|(_, px)| px.clone()).collect();
    let image_count = pixel_data_8x8.len();

    println!("\n=== 感知哈希性能测试 ({} 张图像) ===", image_count);

    let iterations = 50;

    // Mean Hash
    let mean_computer = MeanHashComputer::new(&mut ctx).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = mean_computer.compute(&ctx, &pixel_data_8x8).unwrap();
    }
    let mean_per_batch = start.elapsed() / iterations;
    println!("Mean Hash:   {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, mean_per_batch, mean_per_batch.as_micros() as f64 / image_count as f64);

    // Median Hash
    let median_computer = MedianHashComputer::new(&mut ctx).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = median_computer.compute(&ctx, &pixel_data_8x8).unwrap();
    }
    let median_per_batch = start.elapsed() / iterations;
    println!("Median Hash: {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, median_per_batch, median_per_batch.as_micros() as f64 / image_count as f64);

    // CPU Mean Hash 对比
    let start = Instant::now();
    for _ in 0..iterations {
        for px in &pixel_data_8x8 {
            let _ = cpu_mean_hash(px, 8, 8);
        }
    }
    let cpu_per_batch = start.elapsed() / iterations;
    println!("CPU Mean:    {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, cpu_per_batch, cpu_per_batch.as_micros() as f64 / image_count as f64);

    // SHA-256 异步批量 vs 同步
    let sha256 = Sha256Computer::new(&mut ctx).unwrap();
    let messages: Vec<Vec<u8>> = pixel_data_8x8.clone();

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = sha256.compute(&ctx, &messages).unwrap();
    }
    let sync_per_batch = start.elapsed() / iterations;
    println!("\nSHA-256 同步: {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, sync_per_batch, sync_per_batch.as_micros() as f64 / image_count as f64);

    let start = Instant::now();
    for _ in 0..iterations {
        let mut submitter = sha256.batch_submitter(&ctx).unwrap();
        submitter.submit(&messages).unwrap();
        let _ = submitter.wait_all().unwrap();
    }
    let async_per_batch = start.elapsed() / iterations;
    println!("SHA-256 异步: {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, async_per_batch, async_per_batch.as_micros() as f64 / image_count as f64);
    println!("SHA-256 加速比: {:.2}x (异步/同步)", sync_per_batch.as_secs_f64() / async_per_batch.as_secs_f64());

    // CPU SHA-256 对比
    use sha2::{Sha256, Digest};
    let start = Instant::now();
    for _ in 0..iterations {
        for msg in &messages {
            let mut hasher = Sha256::new();
            hasher.update(msg);
            let _ = hasher.finalize();
        }
    }
    let cpu_sha_per_batch = start.elapsed() / iterations;
    println!("CPU SHA-256:  {} 次迭代, 平均 {:?}/batch ({:.2} μs/image)",
        iterations, cpu_sha_per_batch, cpu_sha_per_batch.as_micros() as f64 / image_count as f64);
}

#[test]
fn test_real_image_hash_similarity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = GpuContext::new_sync().expect("GPU 上下文创建失败");
    let mean_computer = MeanHashComputer::new(&mut ctx).unwrap();

    let images = load_images_as_grayscale(DATA_DIR, 8, 8);
    if images.is_empty() { return; }

    let pixel_data: Vec<Vec<u8>> = images.iter().map(|(_, px)| px.clone()).collect();
    let names: Vec<&str> = images.iter().map(|(n, _)| n.as_str()).collect();

    let hashes = mean_computer.compute(&ctx, &pixel_data).unwrap();

    println!("\n=== Mean Hash 相似度矩阵 (汉明距离) ===");
    let display_count = names.len().min(6);
    print!("{:<8}", "");
    for i in 0..display_count {
        print!("{:>8}", &names[i][..names[i].len().min(6)]);
    }
    println!();

    for i in 0..display_count {
        print!("{:<8}", &names[i][..names[i].len().min(6)]);
        for j in 0..display_count {
            let dist = hamming_distance(hashes[i], hashes[j]);
            print!("{:>8}", dist);
        }
        println!();
    }

    for &hash in hashes.iter() {
        assert_eq!(hamming_distance(hash, hash), 0, "自身汉明距离应为 0");
    }
}