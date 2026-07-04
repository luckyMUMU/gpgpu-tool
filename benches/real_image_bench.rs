use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::GenericImageView;
use gpgpu_tool::{
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    tasks::sha256::Sha256Computer,
    GpuContext,
};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load_all_images(dir: &str) -> Vec<(String, Vec<u8>, u32, u32)> {
    let mut results = Vec::new();
    let path = Path::new(dir);
    if !path.exists() {
        return results;
    }

    fn walk_dir(dir: &Path, results: &mut Vec<(String, Vec<u8>, u32, u32)>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk_dir(&p, results);
                } else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) {
                            let (w, h) = img.dimensions();
                            let luma = img.grayscale();
                            let gray: Vec<u8> = luma.pixels()
                                .map(|(_, _, luma)| luma.0[0])
                                .collect();
                            let name = p.file_name().unwrap().to_string_lossy().to_string();
                            results.push((name, gray, w, h));
                        }
                    }
                }
            }
        }
    }

    walk_dir(path, &mut results);
    results
}

fn bench_phash_real_images(c: &mut Criterion) {
    let images = load_all_images(DATA_DIR);
    if images.is_empty() {
        eprintln!("没有找到测试图像，跳过真实图像 bench");
        return;
    }

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let mut group = c.benchmark_group("real_image_phash");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    let total_images = images.len();
    println!("\n加载了 {} 张真实图像", total_images);

    let all_pixels: Vec<Vec<u8>> = images.iter().map(|(_, pixels, _, _)| pixels.clone()).collect();
    let all_dims: Vec<(u32, u32)> = images.iter().map(|(_, _, w, h)| (*w, *h)).collect();

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    group.bench_function(format!("mean_all_{}_images", total_images), |b| {
        b.iter(|| {
            let _ = hasher.compute(black_box(&ctx), black_box(&all_pixels), black_box(&all_dims));
        })
    });

    let hasher_median = PerceptualHasher::new(&mut ctx, HashAlgorithm::Median).unwrap();
    group.bench_function(format!("median_all_{}_images", total_images), |b| {
        b.iter(|| {
            let _ = hasher_median.compute(black_box(&ctx), black_box(&all_pixels), black_box(&all_dims));
        })
    });

    let hasher_gradient = PerceptualHasher::new(&mut ctx, HashAlgorithm::Gradient).unwrap();
    group.bench_function(format!("gradient_all_{}_images", total_images), |b| {
        b.iter(|| {
            let _ = hasher_gradient.compute(black_box(&ctx), black_box(&all_pixels), black_box(&all_dims));
        })
    });

    let hasher_block = PerceptualHasher::new(&mut ctx, HashAlgorithm::Block).unwrap();
    group.bench_function(format!("block_all_{}_images", total_images), |b| {
        b.iter(|| {
            let _ = hasher_block.compute(black_box(&ctx), black_box(&all_pixels), black_box(&all_dims));
        })
    });

    group.finish();
}

fn bench_phash_real_images_repeated(c: &mut Criterion) {
    let images = load_all_images(DATA_DIR);
    if images.is_empty() {
        return;
    }

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let all_pixels: Vec<Vec<u8>> = images.iter().map(|(_, pixels, _, _)| pixels.clone()).collect();
    let all_dims: Vec<(u32, u32)> = images.iter().map(|(_, _, w, h)| (*w, *h)).collect();

    let mut group = c.benchmark_group("real_image_phash_repeated");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    group.bench_function("mean_10x_batch", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                for _ in 0..10 {
                    let _ = hasher.compute(black_box(&ctx), black_box(&all_pixels), black_box(&all_dims));
                }
            }
            start.elapsed()
        })
    });

    group.finish();
}

fn bench_sha256_real_images(c: &mut Criterion) {
    let images = load_all_images(DATA_DIR);
    if images.is_empty() {
        return;
    }

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("创建失败");

    let mut group = c.benchmark_group("real_image_sha256");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    let all_pixels: Vec<Vec<u8>> = images.iter().map(|(_, pixels, _, _)| pixels.clone()).collect();

    group.bench_function(format!("sha256_all_{}_images", images.len()), |b| {
        b.iter(|| {
            let _ = sha256.compute(black_box(&ctx), black_box(&all_pixels));
        })
    });

    group.bench_function(format!("sha256_async_all_{}_images", images.len()), |b| {
        b.iter(|| {
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            submitter.submit(black_box(&all_pixels)).unwrap();
            let _ = submitter.wait_all().unwrap();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_phash_real_images,
    bench_phash_real_images_repeated,
    bench_sha256_real_images,
);
criterion_main!(benches);
