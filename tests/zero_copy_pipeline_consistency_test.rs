//! 零拷贝流水线一致性测试
//!
//! 验证以下优化路径与原始路径结果一致：
//! 1. SIMD RGBA→灰度转换 vs 标量转换
//! 2. GPU 端 u8→u32 像素打包 vs CPU 端打包
//! 3. rayon 并行解码 vs 串行解码
//! 4. 流水线模式 vs 非流水线模式

#![cfg(feature = "czkawka-compat")]

use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
use gpgpu_tool::tasks::phasher::{HashAlgorithm, PerceptualHasher};

/// 生成测试用 RGBA 图像数据
fn make_test_rgba(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut rgba = Vec::with_capacity(pixel_count * 4);
    for i in 0..pixel_count {
        let r = ((i as u8).wrapping_mul(7)).wrapping_add(seed);
        let g = ((i as u8).wrapping_mul(13)).wrapping_add(seed.wrapping_mul(2));
        let b = ((i as u8).wrapping_mul(31)).wrapping_add(seed.wrapping_mul(3));
        rgba.extend_from_slice(&[r, g, b, 255]);
    }
    rgba
}

/// 生成一组测试图像（全部相同尺寸，确保 process_gpu_buffers 走批量路径）
fn make_test_images(count: usize) -> Vec<(Vec<u8>, u32, u32)> {
    (0..count)
        .map(|i| {
            let w = 64u32;
            let h = 64u32;
            (make_test_rgba(w, h, i as u8), w, h)
        })
        .collect()
}

/// 验证 SIMD 灰度转换与标量转换结果一致
#[test]
fn test_simd_grayscale_consistency() {
    let images = make_test_images(10);

    for (rgba, w, h) in &images {
        // 标量转换（始终可用）
        let gray_scalar = PerceptualHasher::rgba_to_grayscale_scalar(rgba, *w, *h);

        // 通过公共 API 转换（SIMD 或标量取决于 feature）
        let gray_api = PerceptualHasher::rgba_to_grayscale_cpu(rgba, *w, *h);

        assert_eq!(
            gray_scalar, gray_api,
            "SIMD/标量灰度转换结果不一致 (w={}, h={})",
            w, h
        );
    }
}

/// 验证 GPU 端打包与 CPU 端打包产生相同哈希
#[test]
fn test_gpu_pack_hash_consistency() {
    let images = make_test_images(8);

    let accelerator = match CzkawkaGpuAccelerator::new_with_gpu_resize(
        8, // hash_size = 8 (64-bit)
        HashAlgorithm::Gradient,
    ) {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("跳过: 加速器创建失败: {}", e);
            return;
        }
    };

    // 使用流水线模式计算哈希（GPU 端打包）
    let image_count = images.len();
    let images_ref = images.clone();
    let pipeline_hashes = accelerator
        .compute_hashes_zero_copy_pipelined(
            image_count,
            |i| {
                Ok((
                    images_ref[i].0.clone(),
                    images_ref[i].1,
                    images_ref[i].2,
                ))
            },
            4, // sub_batch_size = 4
        )
        .expect("流水线哈希计算失败");

    // 使用非流水线模式计算哈希（CPU 端打包）
    let images_ref2 = images.clone();
    let non_pipeline_hashes = accelerator
        .compute_hashes_zero_copy(
            image_count,
            |i| {
                Ok((
                    images_ref2[i].0.clone(),
                    images_ref2[i].1,
                    images_ref2[i].2,
                ))
            },
        )
        .expect("非流水线哈希计算失败");

    // 两种模式应产生相同数量和相同值的哈希
    assert_eq!(
        pipeline_hashes.len(),
        non_pipeline_hashes.len(),
        "哈希数量不一致"
    );

    for (i, (h1, h2)) in pipeline_hashes.iter().zip(non_pipeline_hashes.iter()).enumerate() {
        assert_eq!(
            h1.as_bytes(),
            h2.as_bytes(),
            "哈希值不一致 [{}]: GPU打包 vs CPU打包",
            i
        );
    }
}

/// 验证流水线模式与非流水线模式产生相同哈希（小规模）
#[test]
fn test_pipeline_vs_non_pipeline_consistency() {
    let images = make_test_images(12);

    let accelerator = match CzkawkaGpuAccelerator::new_with_gpu_resize(
        8,
        HashAlgorithm::Mean,
    ) {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("跳过: 加速器创建失败: {}", e);
            return;
        }
    };

    let image_count = images.len();

    // 流水线模式
    let images_ref = images.clone();
    let pipeline_hashes = accelerator
        .compute_hashes_zero_copy_pipelined(
            image_count,
            |i| Ok((images_ref[i].0.clone(), images_ref[i].1, images_ref[i].2)),
            3,
        )
        .expect("流水线哈希计算失败");

    // 非流水线模式
    let images_ref2 = images.clone();
    let non_pipeline_hashes = accelerator
        .compute_hashes_zero_copy(
            image_count,
            |i| Ok((images_ref2[i].0.clone(), images_ref2[i].1, images_ref2[i].2)),
        )
        .expect("非流水线哈希计算失败");

    assert_eq!(
        pipeline_hashes.len(),
        non_pipeline_hashes.len(),
        "哈希数量不一致"
    );

    for (i, (h1, h2)) in pipeline_hashes.iter().zip(non_pipeline_hashes.iter()).enumerate() {
        assert_eq!(
            h1.as_bytes(),
            h2.as_bytes(),
            "哈希值不一致 [{}]: 流水线 vs 非流水线",
            i
        );
    }
}

/// 验证不同子批大小产生相同哈希
#[test]
fn test_different_sub_batch_size_consistency() {
    let images = make_test_images(20);

    let accelerator = match CzkawkaGpuAccelerator::new_with_gpu_resize(
        8,
        HashAlgorithm::Gradient,
    ) {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("跳过: 加速器创建失败: {}", e);
            return;
        }
    };

    let image_count = images.len();

    // 子批大小 = 1
    let images_ref = images.clone();
    let hashes_s1 = accelerator
        .compute_hashes_zero_copy_pipelined(
            image_count,
            |i| Ok((images_ref[i].0.clone(), images_ref[i].1, images_ref[i].2)),
            1,
        )
        .expect("子批=1 哈希计算失败");

    // 子批大小 = 8
    let images_ref2 = images.clone();
    let hashes_s8 = accelerator
        .compute_hashes_zero_copy_pipelined(
            image_count,
            |i| Ok((images_ref2[i].0.clone(), images_ref2[i].1, images_ref2[i].2)),
            8,
        )
        .expect("子批=8 哈希计算失败");

    // 子批大小 = 20（全部一批）
    let images_ref3 = images.clone();
    let hashes_s20 = accelerator
        .compute_hashes_zero_copy_pipelined(
            image_count,
            |i| Ok((images_ref3[i].0.clone(), images_ref3[i].1, images_ref3[i].2)),
            20,
        )
        .expect("子批=20 哈希计算失败");

    assert_eq!(hashes_s1.len(), hashes_s8.len(), "子批=1 vs 子批=8 数量不一致");
    assert_eq!(hashes_s8.len(), hashes_s20.len(), "子批=8 vs 子批=20 数量不一致");

    for (i, ((h1, h8), h20)) in hashes_s1.iter().zip(hashes_s8.iter()).zip(hashes_s20.iter()).enumerate() {
        assert_eq!(h1.as_bytes(), h8.as_bytes(), "子批=1 vs 子批=8 哈希不一致 [{}]", i);
        assert_eq!(h8.as_bytes(), h20.as_bytes(), "子批=8 vs 子批=20 哈希不一致 [{}]", i);
    }
}

/// 验证空输入处理
#[test]
fn test_empty_input_pipeline() {
    let accelerator = match CzkawkaGpuAccelerator::new_with_gpu_resize(
        8,
        HashAlgorithm::Gradient,
    ) {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("跳过: 加速器创建失败: {}", e);
            return;
        }
    };

    let result = accelerator
        .compute_hashes_zero_copy_pipelined(0, |_| Err("不应被调用".to_string()), 4)
        .expect("空输入不应失败");

    assert!(result.is_empty(), "空输入应返回空哈希列表");
}

/// 验证 Lanczos3 流水线模式与非流水线模式一致性
#[test]
fn test_lanczos3_pipeline_consistency() {
    let count = 6;
    let images: Vec<image::DynamicImage> = (0..count)
        .map(|i| {
            let w = 48 + (i as u32 % 3) * 16;
            let h = 48 + (i as u32 % 2) * 16;
            let rgba = make_test_rgba(w, h, i as u8);
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, rgba).unwrap())
        })
        .collect();

    let accelerator = match CzkawkaGpuAccelerator::new_with_gpu_resize(
        8,
        HashAlgorithm::Gradient,
    ) {
        Ok(acc) => acc,
        Err(e) => {
            eprintln!("跳过: 加速器创建失败: {}", e);
            return;
        }
    };

    // 流水线模式
    let images_ref = images.clone();
    let pipeline_hashes = accelerator
        .compute_hashes_zero_copy_lanczos3_pipelined(
            count,
            |i| Ok(images_ref[i].clone()),
            2,
        )
        .expect("Lanczos3 流水线哈希计算失败");

    // 非流水线模式
    let images_ref2 = images.clone();
    let non_pipeline_hashes = accelerator
        .compute_hashes_zero_copy_lanczos3(
            count,
            |i| Ok(images_ref2[i].clone()),
        )
        .expect("Lanczos3 非流水线哈希计算失败");

    assert_eq!(
        pipeline_hashes.len(),
        non_pipeline_hashes.len(),
        "Lanczos3 哈希数量不一致"
    );

    for (i, (h1, h2)) in pipeline_hashes.iter().zip(non_pipeline_hashes.iter()).enumerate() {
        assert_eq!(
            h1.as_bytes(),
            h2.as_bytes(),
            "Lanczos3 哈希值不一致 [{}]: 流水线 vs 非流水线",
            i
        );
    }
}
