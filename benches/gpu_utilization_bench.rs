use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpgpu_tool::{
    tasks::hash_common::PerceptualHashComputer,
    tasks::mean_hash::MeanHashComputer,
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    tasks::sha256::Sha256Computer,
    GpuContext,
};

fn bench_sha256_sustained_throughput(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("创建失败");

    let messages_10k: Vec<Vec<u8>> = (0..10000).map(|i| vec![i as u8; 32]).collect();
    let messages_50k: Vec<Vec<u8>> = (0..50000).map(|i| vec![i as u8; 32]).collect();
    let messages_100k: Vec<Vec<u8>> = (0..100000).map(|i| vec![i as u8; 32]).collect();

    let mut group = c.benchmark_group("sha256_sustained");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("sync_10k", |b| {
        b.iter(|| sha256.compute(black_box(&ctx), black_box(&messages_10k)))
    });

    group.bench_function("sync_50k", |b| {
        b.iter(|| sha256.compute(black_box(&ctx), black_box(&messages_50k)))
    });

    group.bench_function("sync_100k", |b| {
        b.iter(|| sha256.compute(black_box(&ctx), black_box(&messages_100k)))
    });

    group.bench_function("async_10k", |b| {
        b.iter(|| {
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            submitter.submit(black_box(&messages_10k)).unwrap();
            let _ = submitter.wait_all().unwrap();
        })
    });

    group.bench_function("async_50k", |b| {
        b.iter(|| {
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            submitter.submit(black_box(&messages_50k)).unwrap();
            let _ = submitter.wait_all().unwrap();
        })
    });

    group.bench_function("async_100k", |b| {
        b.iter(|| {
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            submitter.submit(black_box(&messages_100k)).unwrap();
            let _ = submitter.wait_all().unwrap();
        })
    });

    group.finish();
}

fn bench_sha256_continuous_submit(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("创建失败");

    let messages_10k: Vec<Vec<u8>> = (0..10000).map(|i| vec![i as u8; 32]).collect();

    let mut group = c.benchmark_group("sha256_continuous");
    group.measurement_time(Duration::from_secs(20));
    group.sample_size(10);

    group.bench_function("submit_10x10k", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut submitter = sha256.batch_submitter(&ctx).unwrap();
                for _ in 0..10 {
                    submitter.submit(black_box(&messages_10k)).unwrap();
                }
                let _ = submitter.wait_all().unwrap();
            }
            start.elapsed()
        })
    });

    group.finish();
}

fn bench_phash_sustained(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let image = vec![128u8; 64];
    let batch_10k: Vec<Vec<u8>> = (0..10000).map(|_| image.clone()).collect();
    let batch_50k: Vec<Vec<u8>> = (0..50000).map(|_| image.clone()).collect();

    let mut group = c.benchmark_group("phash_sustained");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(20);

    group.bench_function("mean_10k", |b| {
        b.iter(|| hasher.compute(black_box(&ctx), black_box(&batch_10k)))
    });

    group.bench_function("mean_50k", |b| {
        b.iter(|| hasher.compute(black_box(&ctx), black_box(&batch_50k)))
    });

    group.finish();
}

fn bench_phash_large_image(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let mut group = c.benchmark_group("phash_large_image");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(15);

    for (img_size, batch_size) in [(256, 1000), (512, 500), (1024, 200)] {
        let image = vec![128u8; img_size * img_size];
        let batch: Vec<Vec<u8>> = (0..batch_size).map(|_| image.clone()).collect();
        let dims: Vec<(u32, u32)> = (0..batch_size).map(|_| (img_size as u32, img_size as u32)).collect();

        group.bench_function(format!("{}x{}_x{}", img_size, img_size, batch_size), |b| {
            b.iter(|| hasher.compute(black_box(&ctx), black_box(&batch), black_box(&dims)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sha256_sustained_throughput,
    bench_sha256_continuous_submit,
    bench_phash_sustained,
    bench_phash_large_image,
);
criterion_main!(benches);
