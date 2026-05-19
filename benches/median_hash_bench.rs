use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use wgpu_compute_engine::{
    tasks::hash_common::PerceptualHashComputer,
    tasks::median_hash::MedianHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::test_data;

fn bench_median_hash_sizes(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let mut group = c.benchmark_group("median_hash_sizes");

    for (w, h) in [(8u32, 8u32), (16, 16), (32, 32)] {
        let size = format!("{}x{}", w, h);
        let image = test_data::gradient_image((w * h) as usize);
        let batch: Vec<Vec<u8>> = (0..100).map(|_| image.clone()).collect();

        group.bench_with_input(BenchmarkId::new("gpu", &size), &batch, |b, imgs| {
            b.iter(|| hasher.compute(black_box(&ctx), black_box(imgs)))
        });

        group.bench_with_input(BenchmarkId::new("cpu", &size), &batch, |b, imgs| {
            b.iter(|| {
                for img in imgs {
                    let _ = hash_reference::median_hash(black_box(img), w, h);
                }
            })
        });
    }

    group.finish();
}

fn bench_median_hash_batch(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let mut group = c.benchmark_group("median_hash_batch");

    let image = test_data::gradient_image(64);
    for batch_size in [1, 10, 100, 1000] {
        let batch: Vec<Vec<u8>> = (0..batch_size).map(|_| image.clone()).collect();

        group.bench_with_input(BenchmarkId::new("gpu", batch_size), &batch, |b, imgs| {
            b.iter(|| hasher.compute(black_box(&ctx), black_box(imgs)))
        });

        group.bench_with_input(BenchmarkId::new("cpu", batch_size), &batch, |b, imgs| {
            b.iter(|| {
                for img in imgs {
                    let _ = hash_reference::median_hash(black_box(img), 8, 8);
                }
            })
        });
    }

    group.finish();
}

fn bench_median_hash_workgroup(c: &mut Criterion) {
    let mut group = c.benchmark_group("median_hash_workgroup");

    let image = test_data::gradient_image(64);
    let batch: Vec<Vec<u8>> = (0..100).map(|_| image.clone()).collect();

    for wg_size in [64u32, 128, 256, 512] {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let hasher =
            MedianHashComputer::with_workgroup_size(&mut ctx, [wg_size, 1, 1]).expect("创建失败");

        group.bench_with_input(
            BenchmarkId::new(format!("wg{}", wg_size), 100),
            &batch,
            |b, imgs| b.iter(|| hasher.compute(black_box(&ctx), black_box(imgs))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_median_hash_sizes, bench_median_hash_batch, bench_median_hash_workgroup);
criterion_main!(benches);
