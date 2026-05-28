use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::GenericImageView;
use gpgpu_tool::{
    tasks::bktree::{BkTree, hamming_distance},
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    GpuContext,
};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load_real_images(dir: &str) -> Vec<(Vec<u8>, u32, u32)> {
    let mut results = Vec::new();
    let path = std::path::Path::new(dir);
    if !path.exists() { return results; }

    fn walk(dir: &std::path::Path, results: &mut Vec<(Vec<u8>, u32, u32)>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() { walk(&p, results); }
                else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) {
                            let (w, h) = img.dimensions();
                            let luma = img.grayscale();
                            let gray: Vec<u8> = luma.pixels()
                                .map(|(_, _, luma)| luma.0[0])
                                .collect();
                            results.push((gray, w, h));
                        }
                    }
                }
            }
        }
    }

    walk(path, &mut results);
    results
}

fn bench_phash_gpu_vs_cpu_resize(c: &mut Criterion) {
    let real_images = load_real_images(DATA_DIR);
    if real_images.is_empty() { return; }

    let real_count = real_images.len();
    let copies = (5000 + real_count - 1) / real_count;
    let total = real_count * copies;

    println!("\n=== GPU vs CPU 缩放: {} 张真实大图 ({} 张 × {} copies) ===", total, real_count, copies);

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let mut group = c.benchmark_group("phash_gpu_vs_cpu_resize");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for algo in &[HashAlgorithm::Mean, HashAlgorithm::Block] {
        let pixels: Vec<Vec<u8>> = real_images.iter().map(|(p, _, _)| p.clone()).collect();
        let dims: Vec<(u32, u32)> = real_images.iter().map(|(_, w, h)| (*w, *h)).collect();

        let hasher_cpu = PerceptualHasher::with_resize_mode(&mut ctx, *algo, false).unwrap();
        group.bench_function(format!("{:?}_cpu_resize_{}imgs", algo, real_count), |b| {
            b.iter(|| {
                let _ = hasher_cpu.compute(black_box(&ctx), black_box(&pixels), black_box(&dims));
            })
        });

        let hasher_gpu = PerceptualHasher::with_resize_mode(&mut ctx, *algo, true).unwrap();
        group.bench_function(format!("{:?}_gpu_resize_{}imgs", algo, real_count), |b| {
            b.iter(|| {
                let _ = hasher_gpu.compute(black_box(&ctx), black_box(&pixels), black_box(&dims));
            })
        });
    }

    group.finish();
}

fn bench_phash_large_batch_gpu_resize(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let mut group = c.benchmark_group("phash_large_batch_gpu");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    for (img_size, batch_size) in [(256, 5000), (512, 2000), (1024, 500)] {
        let image = vec![128u8; img_size * img_size];
        let pixels: Vec<Vec<u8>> = (0..batch_size).map(|_| image.clone()).collect();
        let dims: Vec<(u32, u32)> = (0..batch_size).map(|_| (img_size as u32, img_size as u32)).collect();

        let hasher_cpu = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, false).unwrap();
        group.bench_function(format!("mean_cpu_{}x{}_x{}", img_size, img_size, batch_size), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = hasher_cpu.compute(black_box(&ctx), black_box(&pixels), black_box(&dims));
                }
                start.elapsed()
            })
        });

        let hasher_gpu = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, true).unwrap();
        group.bench_function(format!("mean_gpu_{}x{}_x{}", img_size, img_size, batch_size), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = hasher_gpu.compute(black_box(&ctx), black_box(&pixels), black_box(&dims));
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_phash_and_bktree_gpu(c: &mut Criterion) {
    let real_images = load_real_images(DATA_DIR);
    if real_images.is_empty() { return; }

    let real_count = real_images.len();

    println!("\n=== {} 张真实大图 GPU 缩放 + BK-Tree ===", real_count);

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, true).unwrap();

    let pixels: Vec<Vec<u8>> = real_images.iter().map(|(p, _, _)| p.clone()).collect();
    let dims: Vec<(u32, u32)> = real_images.iter().map(|(_, w, h)| (*w, *h)).collect();

    let mut group = c.benchmark_group("phash_bktree_gpu");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    group.bench_function(format!("gpu_hash_{}imgs", real_count), |b| {
        b.iter(|| {
            let hashes = hasher.compute(black_box(&ctx), black_box(&pixels), black_box(&dims)).unwrap();
            black_box(hashes);
        })
    });

    let hashes = hasher.compute(&ctx, &pixels, &dims).unwrap();

    group.bench_function(format!("bktree_build_{}imgs", real_count), |b| {
        b.iter(|| {
            let tree = BkTree::from_hashes(black_box(hashes.iter().copied()));
            black_box(tree);
        })
    });

    let tree = BkTree::from_hashes(hashes.iter().copied());

    group.bench_function(format!("bktree_find_threshold5_{}imgs", real_count), |b| {
        let query = hashes[0];
        b.iter(|| {
            let results = tree.find(black_box(query), 5);
            black_box(results);
        })
    });

    group.bench_function(format!("bktree_find_nearest_{}imgs", real_count), |b| {
        let query = hashes[0];
        b.iter(|| {
            let result = tree.find_nearest(black_box(query));
            black_box(result);
        })
    });

    group.bench_function(format!("brute_force_threshold5_{}imgs", real_count), |b| {
        let query = hashes[0];
        b.iter(|| {
            let results: Vec<(u64, u32)> = hashes.iter()
                .filter_map(|&h| {
                    let d = hamming_distance(h, query);
                    if d <= 5 { Some((h, d)) } else { None }
                })
                .collect();
            black_box(results);
        })
    });

    group.finish();
}

fn bench_bktree_large_scale(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("bktree_large_scale");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for size in [10000, 50000, 100000] {
        let clusters = size / 10;
        let per_cluster = 10;
        let mut hashes = Vec::with_capacity(clusters * per_cluster);
        for _ in 0..clusters {
            let center: u64 = rng.random();
            for _ in 0..per_cluster {
                let mut h = center;
                for _ in 0..5u32 {
                    let bit = 1u64 << (rng.random_range(0..64));
                    h ^= bit;
                }
                hashes.push(h);
            }
        }

        group.bench_function(format!("bktree_build_{}", size), |b| {
            b.iter(|| {
                let tree = BkTree::from_hashes(black_box(hashes.iter().copied()));
                black_box(tree);
            })
        });

        let tree = BkTree::from_hashes(hashes.iter().copied());

        group.bench_function(format!("bktree_find_nearest_{}", size), |b| {
            let query = hashes[0];
            b.iter(|| {
                let result = tree.find_nearest(black_box(query));
                black_box(result);
            })
        });

        group.bench_function(format!("bktree_find_threshold5_{}", size), |b| {
            let query = hashes[0];
            b.iter(|| {
                let results = tree.find(black_box(query), 5);
                black_box(results);
            })
        });

        group.bench_function(format!("brute_force_threshold5_{}", size), |b| {
            let query = hashes[0];
            b.iter(|| {
                let results: Vec<(u64, u32)> = hashes.iter()
                    .filter_map(|&h| {
                        let d = hamming_distance(h, query);
                        if d <= 5 { Some((h, d)) } else { None }
                    })
                    .collect();
                black_box(results);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_phash_gpu_vs_cpu_resize,
    bench_phash_large_batch_gpu_resize,
    bench_phash_and_bktree_gpu,
    bench_bktree_large_scale,
);
criterion_main!(benches);
