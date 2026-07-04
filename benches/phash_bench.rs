use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gpgpu_tool::{
    tasks::convolution::{BorderMode, GpuConvolution},
    tasks::gaussian_blur::GpuGaussianBlur,
    tasks::hash_common::PerceptualHashComputer,
    tasks::mean_hash::MeanHashComputer,
    tasks::median_hash::MedianHashComputer,
    tasks::block_hash::BlockHashComputer,
    tasks::gradient_hash::GradientHashComputer,
    tasks::vert_gradient_hash::VertGradientHashComputer,
    tasks::double_gradient_hash::DoubleGradientHashComputer,
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    GpuContext, HashSize,
};

#[cfg(feature = "cpu-fallback")]
use gpgpu_tool::tasks::phasher_cpu::PHasherCpu;

mod common;
use common::test_data;

const IMG_W: u32 = 256;
const IMG_H: u32 = 256;
const IMG_PIXELS: usize = (IMG_W * IMG_H) as usize;

fn generate_test_image() -> Vec<u8> {
    test_data::random_image(IMG_PIXELS)
}

fn generate_test_batch(count: usize) -> Vec<Vec<u8>> {
    let image = generate_test_image();
    (0..count).map(|_| image.clone()).collect()
}

fn generate_test_dims(count: usize) -> Vec<(u32, u32)> {
    vec![(IMG_W, IMG_H); count]
}

fn generate_gaussian_kernel_1d(kernel_size: u32, sigma: f32) -> Vec<f32> {
    let radius = (kernel_size / 2) as i32;
    let mut kernel = Vec::with_capacity(kernel_size as usize);
    let mut sum = 0.0f32;
    for x in -radius..=radius {
        let x_f = x as f32;
        let val = (-((x_f * x_f) / (2.0 * sigma * sigma))).exp();
        kernel.push(val);
        sum += val;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

// ==================== a) 单图像感知哈希延迟基准 ====================

fn bench_single_image_latency(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let image = generate_test_image();
    let dims = vec![(IMG_W, IMG_H)];
    let images = vec![image.clone()];

    let algorithms = [
        HashAlgorithm::Mean,
        HashAlgorithm::Median,
        HashAlgorithm::Block,
        HashAlgorithm::Gradient,
        HashAlgorithm::VertGradient,
        HashAlgorithm::DoubleGradient,
    ];

    let mut group = c.benchmark_group("phash_single_image_latency");

    for algo in algorithms {
        let algo_name = format!("{:?}", algo);
        let hasher = PerceptualHasher::with_resize_mode(&mut ctx, algo, true)
            .expect("创建 GPU hasher 失败");

        group.bench_with_input(
            BenchmarkId::new("gpu", &algo_name),
            &images,
            |b, imgs| {
                b.iter(|| {
                    let _ = hasher.compute(black_box(&ctx), black_box(imgs), black_box(&dims));
                })
            },
        );

        #[cfg(feature = "cpu-fallback")]
        {
            let cpu_hasher = PHasherCpu::new(algo);
            let (tw, th) = algo.target_size_for(HashSize::default());

            group.bench_with_input(
                BenchmarkId::new("cpu", &algo_name),
                &images,
                |b, imgs| {
                    b.iter(|| {
                        let resized = hasher.resize_cpu(black_box(imgs), black_box(&dims)).unwrap();
                        let _ = cpu_hasher.compute(black_box(&resized), tw, th);
                    })
                },
            );
        }
    }

    group.finish();
}

// ==================== b) 批量吞吐量基准 ====================

fn bench_batch_throughput(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, true)
        .expect("创建 GPU hasher 失败");

    #[cfg(feature = "cpu-fallback")]
    let cpu_hasher = PHasherCpu::new(HashAlgorithm::Mean);

    let mut group = c.benchmark_group("phash_batch_throughput");

    for batch_size in [1usize, 10, 100, 1000] {
        let batch = generate_test_batch(batch_size);
        let dims = generate_test_dims(batch_size);

        group.bench_with_input(
            BenchmarkId::new("gpu", batch_size),
            &batch,
            |b, imgs| {
                b.iter(|| {
                    let _ = hasher.compute(black_box(&ctx), black_box(imgs), black_box(&dims));
                })
            },
        );

        #[cfg(feature = "cpu-fallback")]
        {
            let (tw, th) = HashAlgorithm::Mean.target_size_for(HashSize::default());
            group.bench_with_input(
                BenchmarkId::new("cpu", batch_size),
                &batch,
                |b, imgs| {
                    b.iter(|| {
                        let resized = hasher.resize_cpu(black_box(imgs), black_box(&dims)).unwrap();
                        let _ = cpu_hasher.compute(black_box(&resized), tw, th);
                    })
                },
            );
        }
    }

    group.finish();
}

// ==================== c) 零拷贝 vs 非零拷贝对比基准 ====================

fn bench_zero_copy_vs_roundtrip(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let zero_copy_hasher = PerceptualHasher::with_blur(
        &mut ctx, HashAlgorithm::Mean, 1.0, 5,
    ).expect("创建零拷贝 hasher 失败");

    let roundtrip_blur = GpuGaussianBlur::new(&mut ctx).expect("创建 blur 失败");
    let roundtrip_hasher = PerceptualHasher::with_resize_mode(
        &mut ctx, HashAlgorithm::Mean, true,
    ).expect("创建非零拷贝 hasher 失败");

    let image = generate_test_image();
    let dims = vec![(IMG_W, IMG_H)];
    let images = vec![image.clone()];

    let mut group = c.benchmark_group("phash_zero_copy_vs_roundtrip");

    group.bench_function("zero_copy_blur_resize_hash", |b| {
        b.iter(|| {
            let _ = zero_copy_hasher.compute(black_box(&ctx), black_box(&images), black_box(&dims));
        })
    });

    group.bench_function("roundtrip_blur_download_upload_resize_hash", |b| {
        b.iter(|| {
            let blurred = roundtrip_blur.blur(
                black_box(&ctx), black_box(&images[0]), IMG_W, IMG_H, 5, 1.0,
            ).unwrap();
            let blurred_images = vec![blurred];
            let _ = roundtrip_hasher.compute(
                black_box(&ctx), black_box(&blurred_images), black_box(&dims),
            );
        })
    });

    group.finish();
}

// ==================== d) 带/不带高斯模糊预处理对比基准 ====================

fn bench_blur_vs_no_blur(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let hasher_with_blur = PerceptualHasher::with_blur(
        &mut ctx, HashAlgorithm::Mean, 1.0, 5,
    ).expect("创建带模糊 hasher 失败");

    let hasher_no_blur = PerceptualHasher::with_resize_mode(
        &mut ctx, HashAlgorithm::Mean, true,
    ).expect("创建无模糊 hasher 失败");

    let batch_size = 10usize;
    let batch = generate_test_batch(batch_size);
    let dims = generate_test_dims(batch_size);

    let mut group = c.benchmark_group("phash_blur_vs_no_blur");

    group.bench_with_input("with_blur_sigma1.0_k5", &batch, |b, imgs| {
        b.iter(|| {
            let _ = hasher_with_blur.compute(black_box(&ctx), black_box(imgs), black_box(&dims));
        })
    });

    group.bench_with_input("without_blur", &batch, |b, imgs| {
        b.iter(|| {
            let _ = hasher_no_blur.compute(black_box(&ctx), black_box(imgs), black_box(&dims));
        })
    });

    group.finish();
}

// ==================== e) Workgroup Size 对比基准 ====================
// workgroup_size 在 WGSL 着色器中编译时确定，运行时无法动态切换。
// 以下基准通过创建不同 workgroup_size 的管线来对比性能。
// 注意：每次切换 workgroup_size 需要重新编译管线，存在首次启动开销。

fn bench_workgroup_size(c: &mut Criterion) {
    let image = test_data::random_image(64);
    let batch: Vec<Vec<u8>> = (0..100).map(|_| image.clone()).collect();

    let mut group = c.benchmark_group("phash_workgroup_size");

    for wg_size in [8u32, 16, 32, 64] {
        let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
        let hasher = MeanHashComputer::with_workgroup_size(&mut ctx, [wg_size, wg_size, 1])
            .expect("创建失败");

        group.bench_with_input(
            BenchmarkId::new(format!("wg{}x{}", wg_size, wg_size), 100),
            &batch,
            |b, imgs| {
                b.iter(|| hasher.compute(black_box(&ctx), black_box(imgs)))
            },
        );
    }

    group.finish();
}

// ==================== f) LDS 优化对比基准 ====================

fn bench_lds_optimization(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let conv_lds = GpuConvolution::new(&mut ctx)
        .expect("创建卷积器失败")
        .with_lds(true);

    let conv_no_lds = GpuConvolution::new(&mut ctx)
        .expect("创建卷积器失败")
        .with_lds(false);

    let image = generate_test_image();
    let kernel = generate_gaussian_kernel_1d(7, 2.0);

    let mut group = c.benchmark_group("phash_lds_optimization");

    group.bench_function("gaussian_blur_lds_enabled", |b| {
        b.iter(|| {
            let _ = conv_lds.convolve_separable(
                black_box(&ctx),
                black_box(&image),
                IMG_W, IMG_H,
                black_box(&kernel), 7,
                BorderMode::Clamp,
            );
        })
    });

    group.bench_function("gaussian_blur_lds_disabled", |b| {
        b.iter(|| {
            let _ = conv_no_lds.convolve_separable(
                black_box(&ctx),
                black_box(&image),
                IMG_W, IMG_H,
                black_box(&kernel), 7,
                BorderMode::Clamp,
            );
        })
    });

    group.finish();
}

// ==================== g) Push Constant vs Uniform Buffer ====================
// 当前已全面迁移到 Push Constant，无法直接对比。
// Push Constant 相比 Uniform Buffer 的优势：
// - 无需额外绑定组和缓冲区分配
// - 参数通过 command buffer 直接传递，减少 GPU 内存访问
// - 绑定组从 3-binding 降为 2-binding，降低资源管理开销

// ==================== h) 可分离卷积精度修复前后对比 ====================
// 精度问题已修复（中间结果从 u8 截断改为 f32 全精度传递），
// 无法回退对比。修复确保了水平 pass 输出以 f32 精度传递给垂直 pass，
// 消除了 u8 量化引入的误差累积。

// ==================== 6 种算法 GPU 直接计算基准（已缩放图像） ====================

fn bench_all_algorithms_gpu_direct(c: &mut Criterion) {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let mean = MeanHashComputer::new(&mut ctx).expect("创建失败");
    let median = MedianHashComputer::new(&mut ctx).expect("创建失败");
    let block = BlockHashComputer::new(&mut ctx).expect("创建失败");
    let gradient = GradientHashComputer::new(&mut ctx).expect("创建失败");
    let vert_gradient = VertGradientHashComputer::new(&mut ctx).expect("创建失败");
    let double_gradient = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let image_8x8 = test_data::random_image(64);
    let image_9x8 = test_data::random_image(72);
    let image_8x9 = test_data::random_image(72);
    let image_9x9 = test_data::random_image(81);

    let batch_8x8: Vec<Vec<u8>> = (0..100).map(|_| image_8x8.clone()).collect();
    let batch_9x8: Vec<Vec<u8>> = (0..100).map(|_| image_9x8.clone()).collect();
    let batch_8x9: Vec<Vec<u8>> = (0..100).map(|_| image_8x9.clone()).collect();
    let batch_9x9: Vec<Vec<u8>> = (0..100).map(|_| image_9x9.clone()).collect();

    let mut group = c.benchmark_group("phash_all_algorithms_gpu_direct");

    group.bench_with_input("Mean", &batch_8x8, |b, imgs| {
        b.iter(|| mean.compute(black_box(&ctx), black_box(imgs)))
    });
    group.bench_with_input("Median", &batch_8x8, |b, imgs| {
        b.iter(|| median.compute(black_box(&ctx), black_box(imgs)))
    });
    group.bench_with_input("Block", &batch_8x8, |b, imgs| {
        b.iter(|| block.compute(black_box(&ctx), black_box(imgs)))
    });
    group.bench_with_input("Gradient", &batch_9x8, |b, imgs| {
        b.iter(|| gradient.compute(black_box(&ctx), black_box(imgs)))
    });
    group.bench_with_input("VertGradient", &batch_8x9, |b, imgs| {
        b.iter(|| vert_gradient.compute(black_box(&ctx), black_box(imgs)))
    });
    group.bench_with_input("DoubleGradient", &batch_9x9, |b, imgs| {
        b.iter(|| double_gradient.compute(black_box(&ctx), black_box(imgs)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_image_latency,
    bench_batch_throughput,
    bench_zero_copy_vs_roundtrip,
    bench_blur_vs_no_blur,
    bench_workgroup_size,
    bench_lds_optimization,
    bench_all_algorithms_gpu_direct,
);
criterion_main!(benches);
