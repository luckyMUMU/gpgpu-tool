use gpgpu_tool::{
    tasks::hash_common::{HashSize, PerceptualHashComputer},
    tasks::double_gradient_hash::DoubleGradientHashComputer,
    GpuContext,
};

mod common;
use common::hash_reference;
use common::img_hash_verify;
use common::test_data;

#[test]
fn test_double_gradient_hash_single_9x9() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let image = test_data::gradient_image(81);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];
    let cpu_hash = hash_reference::double_gradient_hash(&image, 9, 9);

    assert_eq!(gpu_hash, cpu_hash, "Double Gradient Hash 9x9 单图像测试失败");
}

#[test]
fn test_double_gradient_hash_single_16x16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = DoubleGradientHashComputer::with_config(&mut ctx, [8, 8, 1], HashSize::new(16)).expect("创建失败");

    let image = test_data::gradient_image(256);
    let gpu_hashes = hasher.compute(&ctx, &[image.clone()]).expect("计算失败");
    let cpu_hashes = hash_reference::double_gradient_hash_with_size(&image, 16, 16, 16);

    assert_eq!(gpu_hashes, cpu_hashes, "Double Gradient Hash 16x16 单图像测试失败");
}

#[test]
fn test_double_gradient_hash_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let images: Vec<Vec<u8>> = (0..10)
        .map(|_| test_data::random_image(81))
        .collect();

    let gpu_hashes = hasher.compute(&ctx, &images).expect("计算失败");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::double_gradient_hash(image, 9, 9);
        assert_eq!(gpu_hashes[i], cpu_hash, "Double Gradient Hash 批量测试第 {} 幅失败", i);
    }
}

#[test]
fn test_double_gradient_hash_empty() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    let result = hasher.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}

#[test]
fn test_double_gradient_hash_img_hash_verify() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => { eprintln!("GPU 不可用，跳过测试"); return; }
    };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");

    // 使用随机图像测试（9x9 = 81 像素）
    let image = test_data::random_image(81);
    let gpu_hash = hasher.compute(&ctx, &[image.clone()]).expect("计算失败")[0];

    // GPU vs img_hash 交叉校验
    let img_hash_match = img_hash_verify::verify_double_gradient_hash(&image, 9, 9, gpu_hash);

    // 手写 CPU vs img_hash 交叉校验（验证适配层）
    let cpu_hash = hash_reference::double_gradient_hash(&image, 9, 9);
    let cpu_img_hash_match = img_hash_verify::verify_double_gradient_hash(&image, 9, 9, cpu_hash);

    // 注意：由于 img_hash 缩放插值与本项目直接取像素可能存在差异，
    // 交叉校验可能不完全匹配。如果不匹配，打印诊断信息但不失败。
    if !img_hash_match {
        eprintln!("Double Gradient Hash: GPU 与 img_hash 不匹配 (gpu={:016x})", gpu_hash);
    }
    if !cpu_img_hash_match {
        eprintln!("Double Gradient Hash: CPU 参考与 img_hash 不匹配 (cpu={:016x})", cpu_hash);
    }

    // 至少 GPU 和手写 CPU 必须一致
    assert_eq!(gpu_hash, cpu_hash, "Double Gradient Hash: GPU 与手写 CPU 参考不匹配");
}
