//! poll(Wait) 调用计数基准测试。
//!
//! 量化 GPU 同步开销：每次 `device.poll(wgpu::Maintain::Wait)` 都会阻塞 CPU 等待 GPU，
//! 是性能优化的关键指标。
//!
//! 测试场景：
//! - SHA-256: 1000 条消息（同步 vs 批量提交）
//! - 感知哈希 (pHash): 100 张图像
//! - GPU 卷积: 512×512 图像

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpgpu_tool::poll_counter;
use gpgpu_tool::tasks::sha256::Sha256Computer;
use gpgpu_tool::{BorderMode, GpuConvolution, GpuContext};
use gpgpu_tool::tasks::phasher::{PerceptualHasher, HashAlgorithm};

// ── 辅助函数 ──────────────────────────────────────────────────

/// 生成伪随机图像数据（固定种子，可复现）。
fn make_test_image(size: usize) -> Vec<u8> {
    let mut state: u32 = 0xDEAD_BEEF;
    (0..size)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((state >> 16) & 0xFF) as u8
        })
        .collect()
}

/// 生成固定种子的伪随机消息。
fn make_test_messages(count: usize, msg_len: usize) -> Vec<Vec<u8>> {
    (0..count).map(|i| vec![(i % 256) as u8; msg_len]).collect()
}

// ── SHA-256 poll 计数基准 ─────────────────────────────────────

/// SHA-256 单次 compute（同步接口）的 poll 计数。
///
/// 路径: compute → compute_single_block_batch → download_with_pool → 1 poll
fn bench_sha256_sync_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = make_test_messages(1000, 32); // 1000 条 32 字节消息（单 block）

    c.bench_function("sha256_1000_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = sha256.compute(black_box(&ctx), black_box(&messages));
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// SHA-256 批量提交器的 poll 计数。
///
/// 路径: 100 × submit(无 poll) → 1 × wait_all(2 poll)
fn bench_sha256_batch_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = make_test_messages(1000, 32);

    c.bench_function("sha256_batch_1000_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            // 分 100 批提交，每批 10 条
            for chunk in messages.chunks(10) {
                submitter.submit(black_box(chunk)).unwrap();
            }
            let _ = submitter.wait_all().unwrap();
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// SHA-256 逐条同步调用的 poll 计数（退化场景）。
///
/// 路径: 1000 × (compute → 1 poll) = 1000 poll
fn bench_sha256_individual_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let single_msg = vec![vec![0xAAu8; 32]; 1];

    c.bench_function("sha256_individual_1000_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..1000 {
                let _ = sha256.compute(black_box(&ctx), black_box(&single_msg));
            }
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

// ── 感知哈希 poll 计数基准 ─────────────────────────────────────

/// 感知哈希（Mean）100 张图像的 poll 计数。
///
/// 路径: compute → compute_gpu → compute_phash → dispatch_and_parse → download_with_pool → 1 poll
fn bench_phash_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).expect("PerceptualHasher 创建失败");

    // 100 张 256×256 灰度图
    let images: Vec<Vec<u8>> = (0..100).map(|_| make_test_image(256 * 256)).collect();
    let dimensions: Vec<(u32, u32)> = vec![(256, 256); 100];

    c.bench_function("phash_mean_100_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = hasher.compute(black_box(&ctx), black_box(&images), black_box(&dimensions));
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// 感知哈希（Median）100 张图像的 poll 计数。
fn bench_phash_median_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Median).expect("PerceptualHasher 创建失败");

    let images: Vec<Vec<u8>> = (0..100).map(|_| make_test_image(256 * 256)).collect();
    let dimensions: Vec<(u32, u32)> = vec![(256, 256); 100];

    c.bench_function("phash_median_100_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = hasher.compute(black_box(&ctx), black_box(&images), black_box(&dimensions));
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// 感知哈希逐张处理的 poll 计数（退化场景）。
///
/// 路径: 100 × (compute → 1 poll) = 100 poll
fn bench_phash_individual_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).expect("PerceptualHasher 创建失败");

    let single_image = make_test_image(256 * 256);
    let dimensions = [(256u32, 256u32)];

    c.bench_function("phash_individual_100_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..100 {
                let _ = hasher.compute(
                    black_box(&ctx),
                    black_box(&[single_image.clone()]),
                    black_box(&dimensions),
                );
            }
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

// ── 卷积 poll 计数基准 ────────────────────────────────────────

/// 2D 卷积 512×512 的 poll 计数。
///
/// 路径: convolve_2d → download_with_pool → 1 poll
fn bench_convolution_2d_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let conv = GpuConvolution::new(&mut ctx).expect("GpuConvolution 创建失败");

    let pixels = make_test_image(512 * 512);
    // 3×3 锐化核
    let kernel: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    c.bench_function("convolution_2d_512_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = conv.convolve_2d(
                black_box(&ctx),
                black_box(&pixels),
                512,
                512,
                black_box(&kernel),
                3,
                BorderMode::Clamp,
            );
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// 可分离卷积 512×512 的 poll 计数。
///
/// 路径: convolve_separable → download_with_pool → 1 poll
fn bench_convolution_separable_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let conv = GpuConvolution::new(&mut ctx).expect("GpuConvolution 创建失败");

    let pixels = make_test_image(512 * 512);
    // 5 元素 1D 高斯核
    let kernel_1d: Vec<f32> = vec![0.06136, 0.24477, 0.38774, 0.24477, 0.06136];

    c.bench_function("convolution_sep_512_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = conv.convolve_separable(
                black_box(&ctx),
                black_box(&pixels),
                512,
                512,
                black_box(&kernel_1d),
                5,
                BorderMode::Clamp,
            );
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

/// 卷积逐次调用的 poll 计数（退化场景）。
///
/// 路径: 100 × (convolve → 1 poll) = 100 poll
fn bench_convolution_individual_poll_count(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let conv = GpuConvolution::new(&mut ctx).expect("GpuConvolution 创建失败");

    let pixels = make_test_image(256 * 256);
    let kernel: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    c.bench_function("convolution_individual_100_poll_count", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..100 {
                let _ = conv.convolve_2d(
                    black_box(&ctx),
                    black_box(&pixels),
                    256,
                    256,
                    black_box(&kernel),
                    3,
                    BorderMode::Clamp,
                );
            }
            let count = poll_counter::get();
            black_box(count);
        })
    });
}

// ── 综合 poll 计数报告 ────────────────────────────────────────

/// 打印所有场景的 poll 计数摘要（非 Criterion 基准，仅计数）。
fn bench_poll_count_summary(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let mut group = c.benchmark_group("poll_count_summary");

    // SHA-256: 批量 vs 逐条
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");
    let messages_1000 = make_test_messages(1000, 32);

    group.bench_function("sha256_batch_1000", |b| {
        b.iter(|| {
            poll_counter::reset();
            let mut submitter = sha256.batch_submitter(&ctx).unwrap();
            for chunk in messages_1000.chunks(10) {
                submitter.submit(black_box(chunk)).unwrap();
            }
            let _ = submitter.wait_all().unwrap();
            poll_counter::get()
        })
    });

    group.bench_function("sha256_sync_1000", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = sha256.compute(black_box(&ctx), black_box(&messages_1000));
            poll_counter::get()
        })
    });

    let single_msg = vec![vec![0xAAu8; 32]; 1];
    group.bench_function("sha256_individual_1000x", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..1000 {
                let _ = sha256.compute(black_box(&ctx), black_box(&single_msg));
            }
            poll_counter::get()
        })
    });

    // PHash: 批量 vs 逐张
    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).expect("PerceptualHasher 创建失败");
    let images_100: Vec<Vec<u8>> = (0..100).map(|_| make_test_image(256 * 256)).collect();
    let dims_100: Vec<(u32, u32)> = vec![(256, 256); 100];

    group.bench_function("phash_batch_100", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = hasher.compute(black_box(&ctx), black_box(&images_100), black_box(&dims_100));
            poll_counter::get()
        })
    });

    let single_img = make_test_image(256 * 256);
    let single_dim = [(256u32, 256u32)];
    group.bench_function("phash_individual_100x", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..100 {
                let _ = hasher.compute(
                    black_box(&ctx),
                    black_box(&[single_img.clone()]),
                    black_box(&single_dim),
                );
            }
            poll_counter::get()
        })
    });

    // Convolution: 单次 vs 逐次
    let conv = GpuConvolution::new(&mut ctx).expect("GpuConvolution 创建失败");
    let pixels_512 = make_test_image(512 * 512);
    let kernel: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

    group.bench_function("conv2d_single_512", |b| {
        b.iter(|| {
            poll_counter::reset();
            let _ = conv.convolve_2d(
                black_box(&ctx),
                black_box(&pixels_512),
                512, 512,
                black_box(&kernel),
                3,
                BorderMode::Clamp,
            );
            poll_counter::get()
        })
    });

    let pixels_256 = make_test_image(256 * 256);
    group.bench_function("conv2d_individual_100x", |b| {
        b.iter(|| {
            poll_counter::reset();
            for _ in 0..100 {
                let _ = conv.convolve_2d(
                    black_box(&ctx),
                    black_box(&pixels_256),
                    256, 256,
                    black_box(&kernel),
                    3,
                    BorderMode::Clamp,
                );
            }
            poll_counter::get()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sha256_sync_poll_count,
    bench_sha256_batch_poll_count,
    bench_sha256_individual_poll_count,
    bench_phash_poll_count,
    bench_phash_median_poll_count,
    bench_phash_individual_poll_count,
    bench_convolution_2d_poll_count,
    bench_convolution_separable_poll_count,
    bench_convolution_individual_poll_count,
    bench_poll_count_summary,
);
criterion_main!(benches);
