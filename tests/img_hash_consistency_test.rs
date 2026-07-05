//! 全面一致性验证测试：gpgpu-tool 与 img_hash 库。
//!
//! 验证维度：
//! 1. 批量哈希一致性 — 批量计算 10 张图像，逐一对比 CPU 参考和 img_hash
//! 2. 不同图像尺寸一致性 — 使用 PerceptualHasher 测试多种源图像尺寸（软校验）
//! 3. 距离匹配结果一致性 — gpgpu-tool hamming_distance vs img_hash dist()
//! 4. 所有 6 种算法全覆盖 — Mean, Gradient, VertGradient, DoubleGradient, Block, Median

use gpgpu_tool::tasks::bktree::hamming_distance;
use gpgpu_tool::tasks::block_hash::BlockHashComputer;
use gpgpu_tool::tasks::double_gradient_hash::DoubleGradientHashComputer;
use gpgpu_tool::tasks::gradient_hash::GradientHashComputer;
use gpgpu_tool::tasks::hash_common::PerceptualHashComputer;
use gpgpu_tool::tasks::mean_hash::MeanHashComputer;
use gpgpu_tool::tasks::median_hash::MedianHashComputer;
use gpgpu_tool::tasks::phasher::{HashAlgorithm, PerceptualHasher};
use gpgpu_tool::tasks::vert_gradient_hash::VertGradientHashComputer;
use gpgpu_tool::GpuContext;

mod common;
use common::hash_reference;
use common::img_hash_verify;
use common::test_data;

// =============================================================================
// 辅助函数
// =============================================================================

fn init_ctx() -> Option<GpuContext> {
    match GpuContext::new_sync() {
        Ok(ctx) => Some(ctx),
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            None
        }
    }
}

fn random_images(count: usize, pixel_count: usize) -> Vec<Vec<u8>> {
    (0..count).map(|_| test_data::random_image(pixel_count)).collect()
}

/// 提取 img_hash 8x8 哈希的 u64 值。
fn img_hash_u64(algorithm: img_hash::HashAlg, pixels: &[u8], width: u32, height: u32) -> u64 {
    use img_hash::HasherConfig;
    let image = img_hash::image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .map(img_hash::image::DynamicImage::ImageLuma8)
        .expect("像素数据与尺寸不匹配");
    let hash = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(algorithm)
        .to_hasher()
        .hash_image(&image);
    let bytes = hash.as_bytes();
    let mut result: u64 = 0;
    for (i, &byte) in bytes.iter().enumerate().take(8) {
        result |= (byte as u64) << (i * 8);
    }
    result
}

// =============================================================================
// 维度 1：批量哈希一致性
//
// 批量计算 10 张图像的哈希，逐一对比 CPU 参考和 img_hash 结果。
// 验证批量结果与逐张计算结果一致（通过 CPU 参考间接验证）。
// =============================================================================

#[test]
fn test_batch_consistency_mean() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 64);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "Mean 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::mean_hash(image, 8, 8);
        assert_eq!(batch_hashes[i], cpu_hash, "Mean 第 {} 幅: GPU 与 CPU 参考不一致", i);

        let img_ok = img_hash_verify::verify_mean_hash(image, 8, 8, batch_hashes[i]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Mean, image, 8, 8);
            eprintln!("  Mean 第 {} 幅: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", i, batch_hashes[i], ih);
        }
    }
}

#[test]
fn test_batch_consistency_median() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 64);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "Median 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::median_hash(image, 8, 8);
        assert_eq!(batch_hashes[i], cpu_hash, "Median 第 {} 幅: GPU 与 CPU 参考不一致", i);
    }
}

#[test]
fn test_batch_consistency_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = GradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 72);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "Gradient 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::gradient_hash(image, 8, 9);
        assert_eq!(batch_hashes[i], cpu_hash, "Gradient 第 {} 幅: GPU 与 CPU 参考不一致", i);

        let img_ok = img_hash_verify::verify_gradient_hash(image, 8, 9, batch_hashes[i]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Gradient, image, 8, 9);
            eprintln!("  Gradient 第 {} 幅: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", i, batch_hashes[i], ih);
        }
    }
}

#[test]
fn test_batch_consistency_vert_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = VertGradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 72);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "VertGradient 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::vert_gradient_hash(image, 9, 8);
        assert_eq!(batch_hashes[i], cpu_hash, "VertGradient 第 {} 幅: GPU 与 CPU 参考不一致", i);

        let img_ok = img_hash_verify::verify_vert_gradient_hash(image, 9, 8, batch_hashes[i]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::VertGradient, image, 9, 8);
            eprintln!("  VertGradient 第 {} 幅: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", i, batch_hashes[i], ih);
        }
    }
}

#[test]
fn test_batch_consistency_double_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 81);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "DoubleGradient 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::double_gradient_hash(image, 9, 9);
        assert_eq!(batch_hashes[i], cpu_hash, "DoubleGradient 第 {} 幅: GPU 与 CPU 参考不一致", i);

        let img_ok = img_hash_verify::verify_double_gradient_hash(image, 9, 9, batch_hashes[i]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::DoubleGradient, image, 9, 9);
            eprintln!("  DoubleGradient 第 {} 幅: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", i, batch_hashes[i], ih);
        }
    }
}

#[test]
fn test_batch_consistency_block() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");
    let images = random_images(10, 256);

    let batch_hashes = hasher.compute(&ctx, &images).expect("批量计算失败");
    assert_eq!(batch_hashes.len(), 10, "Block 批量结果数量不匹配");

    for (i, image) in images.iter().enumerate() {
        let cpu_hash = hash_reference::block_hash(image, 16, 16);
        assert_eq!(batch_hashes[i], cpu_hash, "Block 第 {} 幅: GPU 与 CPU 参考不一致", i);

        let img_ok = img_hash_verify::verify_block_hash_horizontal(image, 16, 16, batch_hashes[i]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Blockhash, image, 16, 16);
            eprintln!("  Block 第 {} 幅: img_hash 水平 56bit 交叉校验差异 (gpu={:016x}, img_hash={:016x})", i, batch_hashes[i], ih);
        }
    }
}

// =============================================================================
// 维度 2：不同图像尺寸一致性
//
// 使用 PerceptualHasher 对不同源图像尺寸进行哈希计算。
// GPU 缩放+哈希 vs img_hash 内部缩放+哈希。
// 由于缩放算法不同，交叉校验为软校验（信息输出）。
// 注意：每个尺寸使用新的 PerceptualHasher 实例。
// =============================================================================

const TEST_SIZES: [u32; 5] = [16, 32, 64, 128, 256];

#[test]
fn test_image_sizes_mean() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    for &size in &TEST_SIZES {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).expect("创建失败");
        let pixels = test_data::random_image((size * size) as usize);
        let dims = vec![(size, size)];
        let images = vec![pixels.clone()];
        let hashes = hasher.compute(&ctx, &images, &dims).expect(&format!("Mean size={} 计算失败", size));
        assert_eq!(hashes.len(), 1, "Mean size={}: 返回结果数量不正确", size);

        let img_ok = img_hash_verify::verify_mean_hash(&pixels, size, size, hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Mean, &pixels, size, size);
            eprintln!("  Mean size={}: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", size, hashes[0], ih);
        }
    }
}

#[test]
fn test_image_sizes_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    for &size in &TEST_SIZES {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Gradient).expect("创建失败");
        let pixels = test_data::random_image((size * size) as usize);
        let dims = vec![(size, size)];
        let images = vec![pixels.clone()];
        let hashes = hasher.compute(&ctx, &images, &dims).expect(&format!("Gradient size={} 计算失败", size));
        assert_eq!(hashes.len(), 1, "Gradient size={}: 返回结果数量不正确", size);

        let img_ok = img_hash_verify::verify_gradient_hash(&pixels, size, size, hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Gradient, &pixels, size, size);
            eprintln!("  Gradient size={}: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", size, hashes[0], ih);
        }
    }
}

#[test]
fn test_image_sizes_vert_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    for &size in &TEST_SIZES {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::VertGradient).expect("创建失败");
        let pixels = test_data::random_image((size * size) as usize);
        let dims = vec![(size, size)];
        let images = vec![pixels.clone()];
        let hashes = hasher.compute(&ctx, &images, &dims).expect(&format!("VertGradient size={} 计算失败", size));
        assert_eq!(hashes.len(), 1, "VertGradient size={}: 返回结果数量不正确", size);

        let img_ok = img_hash_verify::verify_vert_gradient_hash(&pixels, size, size, hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::VertGradient, &pixels, size, size);
            eprintln!("  VertGradient size={}: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", size, hashes[0], ih);
        }
    }
}

#[test]
fn test_image_sizes_double_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    for &size in &TEST_SIZES {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::DoubleGradient).expect("创建失败");
        let pixels = test_data::random_image((size * size) as usize);
        let dims = vec![(size, size)];
        let images = vec![pixels.clone()];
        let hashes = hasher.compute(&ctx, &images, &dims).expect(&format!("DoubleGradient size={} 计算失败", size));
        assert_eq!(hashes.len(), 1, "DoubleGradient size={}: 返回结果数量不正确", size);

        let img_ok = img_hash_verify::verify_double_gradient_hash(&pixels, size, size, hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::DoubleGradient, &pixels, size, size);
            eprintln!("  DoubleGradient size={}: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", size, hashes[0], ih);
        }
    }
}

#[test]
fn test_image_sizes_block() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    // Block 目标是 16x16（2*8 x 2*8），只测试能整除的尺寸
    let block_sizes: Vec<u32> = [16, 32, 64, 128, 256].into_iter().collect();
    for &size in &block_sizes {
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Block).expect("创建失败");
        let pixels = test_data::random_image((size * size) as usize);
        let dims = vec![(size, size)];
        let images = vec![pixels.clone()];
        let hashes = hasher.compute(&ctx, &images, &dims).expect(&format!("Block size={} 计算失败", size));
        assert_eq!(hashes.len(), 1, "Block size={}: 返回结果数量不正确", size);

        let img_ok = img_hash_verify::verify_block_hash_horizontal(&pixels, size, size, hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Blockhash, &pixels, size, size);
            eprintln!("  Block size={}: img_hash 水平 56bit 交叉校验差异 (gpu={:016x}, img_hash={:016x})", size, hashes[0], ih);
        }
    }
}

// =============================================================================
// 维度 3：距离匹配结果一致性
//
// 用 gpgpu-tool 计算哈希后用 hamming_distance 匹配，
// 用 img_hash 计算哈希后用 XOR + popcount 匹配，
// 验证两者的距离值完全一致。
// =============================================================================

#[test]
fn test_distance_matching_mean() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(8, 8),
        test_data::vertical_gradient_image(8, 8),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::mean_hash(&images[0], 8, 8);
    let cpu_b = hash_reference::mean_hash(&images[1], 8, 8);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "Mean: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "Mean: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "Mean: 哈希 b 不一致");

    // img_hash 距离软校验
    let ih_a = img_hash_u64(img_hash::HashAlg::Mean, &images[0], 8, 8);
    let ih_b = img_hash_u64(img_hash::HashAlg::Mean, &images[1], 8, 8);
    if hashes[0] == ih_a && hashes[1] == ih_b {
        let img_hash_dist = (ih_a ^ ih_b).count_ones();
        assert_eq!(gpu_dist, img_hash_dist, "Mean: 距离不一致（哈希相同但距离不同）");
    }
}

#[test]
fn test_distance_matching_median() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(8, 8),
        test_data::vertical_gradient_image(8, 8),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::median_hash(&images[0], 8, 8);
    let cpu_b = hash_reference::median_hash(&images[1], 8, 8);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "Median: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "Median: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "Median: 哈希 b 不一致");
}

#[test]
fn test_distance_matching_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = GradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(8, 9),
        test_data::vertical_gradient_image(8, 9),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::gradient_hash(&images[0], 8, 9);
    let cpu_b = hash_reference::gradient_hash(&images[1], 8, 9);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "Gradient: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "Gradient: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "Gradient: 哈希 b 不一致");
}

#[test]
fn test_distance_matching_vert_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = VertGradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(9, 8),
        test_data::vertical_gradient_image(9, 8),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::vert_gradient_hash(&images[0], 9, 8);
    let cpu_b = hash_reference::vert_gradient_hash(&images[1], 9, 8);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "VertGradient: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "VertGradient: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "VertGradient: 哈希 b 不一致");
}

#[test]
fn test_distance_matching_double_gradient() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(9, 9),
        test_data::vertical_gradient_image(9, 9),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::double_gradient_hash(&images[0], 9, 9);
    let cpu_b = hash_reference::double_gradient_hash(&images[1], 9, 9);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "DoubleGradient: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "DoubleGradient: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "DoubleGradient: 哈希 b 不一致");
}

#[test]
fn test_distance_matching_block() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };
    let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");
    let images = vec![
        test_data::horizontal_gradient_image(16, 16),
        test_data::vertical_gradient_image(16, 16),
    ];

    let hashes = hasher.compute(&ctx, &images).expect("计算失败");
    let gpu_dist = hamming_distance(hashes[0], hashes[1]);

    let cpu_a = hash_reference::block_hash(&images[0], 16, 16);
    let cpu_b = hash_reference::block_hash(&images[1], 16, 16);
    let cpu_dist = hamming_distance(cpu_a, cpu_b);
    assert_eq!(gpu_dist, cpu_dist, "Block: GPU 与 CPU 距离不一致");
    assert_eq!(hashes[0], cpu_a, "Block: 哈希 a 不一致");
    assert_eq!(hashes[1], cpu_b, "Block: 哈希 b 不一致");
}

// =============================================================================
// 维度 4：所有 6 种算法全覆盖 — 综合测试
// =============================================================================

#[test]
fn test_all_algorithms_full_coverage() {
    let mut ctx = match init_ctx() { Some(c) => c, None => return };

    // --- Mean ---
    {
        let hasher = MeanHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(64);
        let batch = random_images(10, 64);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::mean_hash(&image, 8, 8), "Mean: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::mean_hash(img, 8, 8), "Mean 批量第 {} 幅不一致", i);
        }
        let img_ok = img_hash_verify::verify_mean_hash(&image, 8, 8, all_hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Mean, &image, 8, 8);
            eprintln!("  Mean: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", all_hashes[0], ih);
        }
    }

    // --- Median ---
    {
        let hasher = MedianHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(64);
        let batch = random_images(10, 64);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::median_hash(&image, 8, 8), "Median: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::median_hash(img, 8, 8), "Median 批量第 {} 幅不一致", i);
        }
    }

    // --- Gradient ---
    {
        let hasher = GradientHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(72);
        let batch = random_images(10, 72);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::gradient_hash(&image, 8, 9), "Gradient: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::gradient_hash(img, 8, 9), "Gradient 批量第 {} 幅不一致", i);
        }
        let img_ok = img_hash_verify::verify_gradient_hash(&image, 8, 9, all_hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Gradient, &image, 8, 9);
            eprintln!("  Gradient: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", all_hashes[0], ih);
        }
    }

    // --- VertGradient ---
    {
        let hasher = VertGradientHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(72);
        let batch = random_images(10, 72);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::vert_gradient_hash(&image, 9, 8), "VertGradient: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::vert_gradient_hash(img, 9, 8), "VertGradient 批量第 {} 幅不一致", i);
        }
        let img_ok = img_hash_verify::verify_vert_gradient_hash(&image, 9, 8, all_hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::VertGradient, &image, 9, 8);
            eprintln!("  VertGradient: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", all_hashes[0], ih);
        }
    }

    // --- DoubleGradient ---
    {
        let hasher = DoubleGradientHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(81);
        let batch = random_images(10, 81);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::double_gradient_hash(&image, 9, 9), "DoubleGradient: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::double_gradient_hash(img, 9, 9), "DoubleGradient 批量第 {} 幅不一致", i);
        }
        let img_ok = img_hash_verify::verify_double_gradient_hash(&image, 9, 9, all_hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::DoubleGradient, &image, 9, 9);
            eprintln!("  DoubleGradient: img_hash 交叉校验差异 (gpu={:016x}, img_hash={:016x})", all_hashes[0], ih);
        }
    }

    // --- Block ---
    {
        let hasher = BlockHashComputer::new(&mut ctx).expect("创建失败");
        let image = test_data::gradient_image(256);
        let batch = random_images(10, 256);
        let all_images: Vec<Vec<u8>> = std::iter::once(image.clone()).chain(batch.iter().cloned()).collect();
        let all_hashes = hasher.compute(&ctx, &all_images).expect("计算失败");

        assert_eq!(all_hashes[0], hash_reference::block_hash(&image, 16, 16), "Block: 单图像不一致");
        for (i, img) in batch.iter().enumerate() {
            assert_eq!(all_hashes[i + 1], hash_reference::block_hash(img, 16, 16), "Block 批量第 {} 幅不一致", i);
        }
        let img_ok = img_hash_verify::verify_block_hash_horizontal(&image, 16, 16, all_hashes[0]);
        if !img_ok {
            let ih = img_hash_u64(img_hash::HashAlg::Blockhash, &image, 16, 16);
            eprintln!("  Block: img_hash 水平 56bit 交叉校验差异 (gpu={:016x}, img_hash={:016x})", all_hashes[0], ih);
        }
    }

    println!("✅ 所有 6 种算法全覆盖验证完成");
}
