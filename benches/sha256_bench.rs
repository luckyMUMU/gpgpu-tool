use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sha2::{Digest, Sha256};
use wgpu_compute_engine::tasks::sha256::Sha256Computer;
use wgpu_compute_engine::GpuContext;

fn bench_single_block_batch(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let mut group = c.benchmark_group("sha256_single_block_batch");

    for size in [1, 10, 100, 1000, 10000] {
        let messages: Vec<Vec<u8>> = (0..size).map(|i| vec![i as u8; 32]).collect();

        group.bench_with_input(BenchmarkId::new("gpu", size), &messages, |b, msgs| {
            b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs)))
        });

        group.bench_with_input(BenchmarkId::new("cpu", size), &messages, |b, msgs| {
            b.iter(|| {
                for msg in msgs {
                    let mut hasher = Sha256::new();
                    hasher.update(black_box(msg));
                    let _ = hasher.finalize();
                }
            })
        });
    }

    group.finish();
}

fn bench_multi_block(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let mut group = c.benchmark_group("sha256_multi_block");

    for msg_size in [64, 128, 256, 1024] {
        let messages: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8; msg_size]).collect();

        group.bench_with_input(BenchmarkId::new("gpu", msg_size), &messages, |b, msgs| {
            b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs)))
        });

        group.bench_with_input(BenchmarkId::new("cpu", msg_size), &messages, |b, msgs| {
            b.iter(|| {
                for msg in msgs {
                    let mut hasher = Sha256::new();
                    hasher.update(black_box(msg));
                    let _ = hasher.finalize();
                }
            })
        });
    }

    group.finish();
}

fn bench_pipeline_cache(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    c.bench_function("sha256_pipeline_first_create", |b| {
        b.iter(|| {
            let _ = Sha256Computer::new(&mut ctx).expect("创建失败");
        })
    });
}

fn bench_per_message_overhead(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let mut group = c.benchmark_group("sha256_per_message_overhead");

    for msg_size in [16, 32, 55] {
        let messages: Vec<Vec<u8>> = vec![vec![0xAAu8; msg_size]; 1000];

        group.bench_with_input(BenchmarkId::new("gpu_1000x", msg_size), &messages, |b, msgs| {
            b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs)))
        });
    }

    group.finish();
}

fn bench_workgroup_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("sha256_workgroup_size");

    let messages_100: Vec<Vec<u8>> = (0..100).map(|i| vec![i as u8; 32]).collect();
    let messages_1000: Vec<Vec<u8>> = (0..1000).map(|i| vec![i as u8; 32]).collect();
    let messages_10000: Vec<Vec<u8>> = (0..10000).map(|i| vec![i as u8; 32]).collect();

    for wg_size in [64u32, 128, 256, 512] {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let sha256 =
            Sha256Computer::with_workgroup_size(&mut ctx, [wg_size, 1, 1]).expect("创建失败");

        group.bench_with_input(
            BenchmarkId::new(format!("wg{}", wg_size), 100),
            &messages_100,
            |b, msgs| b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs))),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("wg{}", wg_size), 1000),
            &messages_1000,
            |b, msgs| b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs))),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("wg{}", wg_size), 10000),
            &messages_10000,
            |b, msgs| b.iter(|| sha256.compute(black_box(&ctx), black_box(msgs))),
        );
    }

    group.finish();
}

fn bench_gpu_overhead_breakdown(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let mut group = c.benchmark_group("sha256_gpu_overhead");

    // 测试空提交开销（无实际计算）
    group.bench_function("empty_dispatch", |b| {
        use wgpu_compute_engine::{BufferUsage, ComputePipeline, GpuBuffer};

        let wgsl = r#"
            @compute @workgroup_size(1)
            fn main() {}
        "#;
        let pipeline = ComputePipeline::create(ctx.device(), wgsl, [1, 1, 1]).unwrap();
        let input = GpuBuffer::from_data(ctx.device(), &[0u32; 16], BufferUsage::Storage);
        let output = GpuBuffer::empty(ctx.device(), 64, BufferUsage::Storage);
        let params = GpuBuffer::from_data(ctx.device(), &[0u32; 4], BufferUsage::Uniform);

        b.iter(|| {
            pipeline.dispatch(
                ctx.device(),
                ctx.queue(),
                &input,
                &output,
                &params,
                [1, 1, 1],
            );
            ctx.device().poll(wgpu::Maintain::Wait);
        })
    });

    // 测试单条消息端到端（含数据准备、调度、下载）
    let single_msg = vec![vec![0xAAu8; 32]];
    group.bench_function("single_message_e2e", |b| {
        b.iter(|| sha256.compute(black_box(&ctx), black_box(&single_msg)))
    });

    // 测试 1000 条消息批量（摊薄开销后）
    let batch_1000: Vec<Vec<u8>> = (0..1000).map(|i| vec![i as u8; 32]).collect();
    group.bench_function("batch_1000_e2e", |b| {
        b.iter(|| sha256.compute(black_box(&ctx), black_box(&batch_1000)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_block_batch,
    bench_multi_block,
    bench_pipeline_cache,
    bench_per_message_overhead,
    bench_workgroup_size,
    bench_gpu_overhead_breakdown
);
criterion_main!(benches);
