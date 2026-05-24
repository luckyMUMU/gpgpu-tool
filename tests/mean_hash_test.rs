use wgpu_compute_engine::{
    tasks::hash_common::{HashSize, PerceptualHashComputer},
    tasks::mean_hash::MeanHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::report::TestReport;
use common::test_data;

use std::time::Instant;

fn get_gpu_info(ctx: &GpuContext) -> String {
    ctx.adapter_info()
}

#[test]
fn test_mean_hash_single_8x8() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(64);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::mean_hash(&image, 8, 8);

    assert_eq!(gpu_hash, cpu_hash, "Mean Hash 8x8 单图像测试失败");
}

#[test]
fn test_mean_hash_single_16x16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MeanHashComputer::with_config(&mut ctx, [256, 1, 1], HashSize::new(16)).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::mean_hash(&image, 16, 16);

    assert_eq!(gpu_hash, cpu_hash, "Mean Hash 16x16 单图像测试失败");
}

#[test]
fn test_mean_hash_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|_| test_data::random_image(64))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::mean_hash(image, 8, 8);
        assert_eq!(gpu_hashes[i], cpu_hash, "Mean Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
fn test_mean_hash_empty() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}

#[test]
fn test_mean_hash_full_report() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let gpu_info = get_gpu_info(&ctx);
    let mut report = TestReport::new(gpu_info);

    let sizes = vec![(8, 8), (16, 16), (32, 32)];
    for (w, h) in &sizes {
        let size = format!("{}x{}", w, h);
        let hash_size = HashSize::new(*w);
        let hasher = MeanHashComputer::with_config(&mut ctx, [256, 1, 1], hash_size).expect("创建失败");
        let image = test_data::gradient_image((w * h) as usize);
        let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
        let cpu_hash = hash_reference::mean_hash(&image, *w, *h);
        report.add_test_result("Mean", &size, "single", gpu_hash == cpu_hash);
    }

    let hasher_8 = MeanHashComputer::new(&mut ctx).expect("创建失败");

    let batch_images: Vec<Vec<u8>> = (0..10)
        .map(|_| test_data::random_image(64))
        .collect();
    let gpu_hashes = hasher_8.compute(&ctx, &batch_images).expect("计算失败");
    let all_match = batch_images.iter().enumerate().all(|(i, img)| {
        gpu_hashes[i] == hash_reference::mean_hash(img, 8, 8)
    });
    report.add_test_result("Mean", "8x8", "batch", all_match);

    let empty_result = hasher_8.compute(&ctx, &[]).expect("计算失败");
    report.add_test_result("Mean", "8x8", "empty", empty_result.is_empty());

    for (w, h) in &[(8u32, 8u32), (16, 16), (32, 32)] {
        let size = format!("{}x{}", w, h);
        let hash_size = HashSize::new(*w);
        let hasher = MeanHashComputer::with_config(&mut ctx, [256, 1, 1], hash_size).expect("创建失败");
        let image = test_data::random_image((w * h) as usize);
        let batch: Vec<Vec<u8>> = (0..100).map(|_| image.clone()).collect();

        let gpu_start = Instant::now();
        hasher.compute(&ctx, &batch).expect("计算失败");
        let gpu_time = gpu_start.elapsed().as_secs_f64() * 1000.0;

        let cpu_start = Instant::now();
        for img in &batch {
            let _ = hash_reference::mean_hash(img, *w, *h);
        }
        let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0;

        report.add_bench_result("Mean", &size, 100, gpu_time, cpu_time);
    }

    report.generate("target/test-reports/mean_hash_report.md");
}
