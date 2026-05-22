use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wgpu_compute_engine::{
    tasks::sha256::Sha256Computer,
    GpuContext,
};

fn bench_sha256_max_throughput(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("创建失败");

    let mut group = c.benchmark_group("sha256_max_throughput");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    for batch_size in [50000, 100000, 200000, 500000] {
        let messages: Vec<Vec<u8>> = (0..batch_size).map(|i| vec![i as u8; 32]).collect();

        group.bench_function(format!("async_{}", batch_size), |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let mut submitter = sha256.batch_submitter(&ctx);
                    submitter.submit(black_box(&messages)).unwrap();
                    let _ = submitter.wait_all().unwrap();
                }
                start.elapsed()
            })
        });
    }

    group.finish();
}

fn bench_sha256_continuous_stream(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("创建失败");

    let messages_100k: Vec<Vec<u8>> = (0..100000).map(|i| vec![i as u8; 32]).collect();

    let mut group = c.benchmark_group("sha256_continuous_stream");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(5);

    group.bench_function("stream_10x100k", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut submitter = sha256.batch_submitter(&ctx);
                for _ in 0..10 {
                    submitter.submit(black_box(&messages_100k)).unwrap();
                }
                let _ = submitter.wait_all().unwrap();
            }
            start.elapsed()
        })
    });

    group.bench_function("stream_50x100k", |b| {
        b.iter_custom(|iters| {
            let start = Instant::now();
            for _ in 0..iters {
                let mut submitter = sha256.batch_submitter(&ctx);
                for _ in 0..50 {
                    submitter.submit(black_box(&messages_100k)).unwrap();
                }
                let _ = submitter.wait_all().unwrap();
            }
            start.elapsed()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sha256_max_throughput,
    bench_sha256_continuous_stream,
);
criterion_main!(benches);
