use wgpu_compute_engine::{
    tasks::hash_common::PerceptualHashComputer,
    tasks::block_hash::BlockHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::test_data;

#[test]
fn test_block_hash_single_16x16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::block_hash(&image, 16, 16);

    assert_eq!(gpu_hash, cpu_hash, "Block Hash 16x16 单图像测试失败");
}

#[test]
fn test_block_hash_single_32x32() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(1024);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::block_hash(&image, 32, 32);

    assert_eq!(gpu_hash, cpu_hash, "Block Hash 32x32 单图像测试失败");
}

#[test]
fn test_block_hash_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|_| test_data::random_image(256))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::block_hash(image, 16, 16);
        assert_eq!(gpu_hashes[i], cpu_hash, "Block Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
fn test_block_hash_empty() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}
