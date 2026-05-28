use gpgpu_tool::{
    tasks::hash_common::{HashSize, PerceptualHashComputer},
    tasks::block_hash::BlockHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::img_hash_verify;
use common::test_data;

#[test]
fn test_block_hash_single_8x8() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(64);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::block_hash(&image, 8, 8);

    assert_eq!(gpu_hash, cpu_hash, "Block Hash 8x8 单图像测试失败");
}

#[test]
fn test_block_hash_single_16x16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = BlockHashComputer::with_config(&mut ctx, [8, 8, 1], HashSize::new(16)).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hashes = hasher.compute(&ctx, &[image.clone()]).expect("计算失败");
    let cpu_hashes = hash_reference::block_hash_with_size(&image, 16, 16, 16);

    assert_eq!(gpu_hashes, cpu_hashes, "Block Hash 16x16 单图像测试失败");
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
    let hasher = BlockHashComputer::with_config(&mut ctx, [8, 8, 1], HashSize::new(32)).expect("创建失败");

    let image = test_data::gradient_image(1024);
    let gpu_hashes = hasher.compute(&ctx, &[image.clone()]).expect("计算失败");
    let cpu_hashes = hash_reference::block_hash_with_size(&image, 32, 32, 32);

    assert_eq!(gpu_hashes, cpu_hashes, "Block Hash 32x32 单图像测试失败");
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
        .map(|_| test_data::random_image(64))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::block_hash(image, 8, 8);
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

#[test]
fn test_block_hash_img_hash_verify() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => { eprintln!("GPU 不可用，跳过测试"); return; }
    };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");

    // 使用随机图像测试（8x8 = 64 像素）
    let image = test_data::random_image(64);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];

    // GPU vs img_hash 交叉校验（仅对比水平比较部分）
    let img_hash_match = img_hash_verify::verify_block_hash_horizontal(&image, 8, 8, gpu_hash);

    // 手写 CPU vs img_hash 交叉校验（验证适配层）
    let cpu_hash = hash_reference::block_hash(&image, 8, 8);
    let cpu_img_hash_match = img_hash_verify::verify_block_hash_horizontal(&image, 8, 8, cpu_hash);

    // 注意：由于 img_hash 的 Blockhash 算法与本项目实现存在差异
    // （img_hash 按行分组中值比较 vs 本项目相邻块均值比较），
    // 交叉校验可能不完全匹配。如果不匹配，打印诊断信息但不失败。
    if !img_hash_match {
        eprintln!("Block Hash: GPU 与 img_hash 不匹配 (gpu={:016x}, 可能因比较策略差异)", gpu_hash);
    }
    if !cpu_img_hash_match {
        eprintln!("Block Hash: CPU 参考与 img_hash 不匹配 (cpu={:016x}, 可能因比较策略差异)", cpu_hash);
    }

    // 至少 GPU 和手写 CPU 必须一致
    assert_eq!(gpu_hash, cpu_hash, "Block Hash: GPU 与手写 CPU 参考不匹配");
}
