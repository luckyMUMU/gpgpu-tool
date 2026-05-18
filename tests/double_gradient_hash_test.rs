use wgpu_compute_engine::{
    tasks::hash_common::PerceptualHashComputer,
    tasks::double_gradient_hash::DoubleGradientHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::test_data;

#[test]
#[ignore]
fn test_double_gradient_hash_single_9x9() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(81);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::double_gradient_hash(&image, 9, 9);

    assert_eq!(gpu_hash, cpu_hash, "Double Gradient Hash 9x9 单图像测试失败");
}

#[test]
#[ignore]
fn test_double_gradient_hash_single_17x17() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(289);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::double_gradient_hash(&image, 17, 17);

    assert_eq!(gpu_hash, cpu_hash, "Double Gradient Hash 17x17 单图像测试失败");
}

#[test]
#[ignore]
fn test_double_gradient_hash_batch() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|i| test_data::random_image(81 + i))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::double_gradient_hash(image, 9, 9);
        assert_eq!(gpu_hashes[i], cpu_hash, "Double Gradient Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
#[ignore]
fn test_double_gradient_hash_empty() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}
