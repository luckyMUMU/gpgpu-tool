use wgpu_compute_engine::{
    tasks::hash_common::PerceptualHashComputer,
    tasks::median_hash::MedianHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::test_data;

#[test]
#[ignore]
fn test_median_hash_single_8x8() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(64);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::median_hash(&image, 8, 8);

    assert_eq!(gpu_hash, cpu_hash, "Median Hash 8x8 单图像测试失败");
}

#[test]
#[ignore]
fn test_median_hash_single_16x16() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::median_hash(&image, 16, 16);

    assert_eq!(gpu_hash, cpu_hash, "Median Hash 16x16 单图像测试失败");
}

#[test]
#[ignore]
fn test_median_hash_batch() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|i| test_data::random_image(64 + i))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::median_hash(image, 8, 8);
        assert_eq!(gpu_hashes[i], cpu_hash, "Median Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
#[ignore]
fn test_median_hash_empty() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}
