use gpgpu_tool::{
    tasks::hash_common::{HashSize, PerceptualHashComputer},
    tasks::median_hash::MedianHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::test_data;

#[test]
fn test_median_hash_single_8x8() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(64);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::median_hash(&image, 8, 8);

    assert_eq!(gpu_hash, cpu_hash, "Median Hash 8x8 单图像测试失败");
}

#[test]
fn test_median_hash_single_16x16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MedianHashComputer::with_config(&mut ctx, [256, 1, 1], HashSize::new(16)).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::median_hash(&image, 16, 16);

    assert_eq!(gpu_hash, cpu_hash, "Median Hash 16x16 单图像测试失败");
}

#[test]
fn test_median_hash_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|_| test_data::random_image(64))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::median_hash(image, 8, 8);
        assert_eq!(gpu_hashes[i], cpu_hash, "Median Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
fn test_median_hash_empty() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}
