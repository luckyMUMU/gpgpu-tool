//! czkawka GPU 加速集成一致性测试套件。
//!
//! 测试矩阵：
//! 1. HashAlgorithm 字符串名互转（from_czkawka / to_czkawka_name）
//! 2. HashSize::from_czkawka 校验
//! 3. RGBA→灰度转换一致性（czkawka 公式）
//! 4. compute_similar_pairs 输出格式与正确性
//! 5. GPU vs CPU 哈希一致性（需要 GPU 环境）

use gpgpu_tool::{
    tasks::gpu_matcher::GpuHashMatcherBytes,
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    CzkawkaGpuAccelerator, GpuContext, GpuError, HashBytes, HashSize,
};

// ==================== HashAlgorithm 互转测试 ====================

#[test]
fn test_hash_algorithm_from_czkawka_all_variants() {
    assert_eq!(HashAlgorithm::from_czkawka("Mean"), Some(HashAlgorithm::Mean));
    assert_eq!(
        HashAlgorithm::from_czkawka("Median"),
        Some(HashAlgorithm::Median)
    );
    assert_eq!(
        HashAlgorithm::from_czkawka("Gradient"),
        Some(HashAlgorithm::Gradient)
    );
    assert_eq!(
        HashAlgorithm::from_czkawka("Block"),
        Some(HashAlgorithm::Block)
    );
    assert_eq!(
        HashAlgorithm::from_czkawka("Blockhash"),
        Some(HashAlgorithm::Block)
    );
    assert_eq!(
        HashAlgorithm::from_czkawka("VertGradient"),
        Some(HashAlgorithm::VertGradient)
    );
    assert_eq!(
        HashAlgorithm::from_czkawka("DoubleGradient"),
        Some(HashAlgorithm::DoubleGradient)
    );
}

#[test]
fn test_hash_algorithm_from_czkawka_unknown() {
    assert_eq!(HashAlgorithm::from_czkawka("Unknown"), None);
    assert_eq!(HashAlgorithm::from_czkawka(""), None);
    assert_eq!(HashAlgorithm::from_czkawka("mean"), None); // 大小写敏感
}

#[test]
fn test_hash_algorithm_to_czkawka_name() {
    assert_eq!(HashAlgorithm::Mean.to_czkawka_name(), "Mean");
    assert_eq!(HashAlgorithm::Median.to_czkawka_name(), "Median");
    assert_eq!(HashAlgorithm::Gradient.to_czkawka_name(), "Gradient");
    assert_eq!(HashAlgorithm::Block.to_czkawka_name(), "Blockhash");
    assert_eq!(
        HashAlgorithm::VertGradient.to_czkawka_name(),
        "VertGradient"
    );
    assert_eq!(
        HashAlgorithm::DoubleGradient.to_czkawka_name(),
        "DoubleGradient"
    );
}

#[test]
fn test_hash_algorithm_roundtrip() {
    let algorithms = [
        HashAlgorithm::Mean,
        HashAlgorithm::Median,
        HashAlgorithm::Gradient,
        HashAlgorithm::Block,
        HashAlgorithm::VertGradient,
        HashAlgorithm::DoubleGradient,
    ];

    for alg in &algorithms {
        let name = alg.to_czkawka_name();
        let restored = HashAlgorithm::from_czkawka(name);
        assert_eq!(restored, Some(*alg), "往返转换失败: {} -> {}", name, name);
    }
}

// ==================== HashSize::from_czkawka 测试 ====================

#[test]
fn test_hash_size_from_czkawka_valid() {
    assert_eq!(HashSize::from_czkawka(8).unwrap(), HashSize::new(8));
    assert_eq!(HashSize::from_czkawka(16).unwrap(), HashSize::new(16));
    assert_eq!(HashSize::from_czkawka(32).unwrap(), HashSize::new(32));
    assert_eq!(HashSize::from_czkawka(64).unwrap(), HashSize::new(64));
}

#[test]
fn test_hash_size_from_czkawka_invalid() {
    assert!(HashSize::from_czkawka(0).is_err());
    assert!(HashSize::from_czkawka(1).is_err());
    assert!(HashSize::from_czkawka(7).is_err());
    assert!(HashSize::from_czkawka(9).is_err());
    assert!(HashSize::from_czkawka(128).is_err());
    assert!(HashSize::from_czkawka(255).is_err());
}

#[test]
fn test_hash_size_from_czkawka_bit_widths() {
    assert_eq!(HashSize::from_czkawka(8).unwrap().bits(), 64);
    assert_eq!(HashSize::from_czkawka(16).unwrap().bits(), 256);
    assert_eq!(HashSize::from_czkawka(32).unwrap().bits(), 1024);
    assert_eq!(HashSize::from_czkawka(64).unwrap().bits(), 4096);
}

// ==================== RGBA→灰度转换一致性测试 ====================

/// 验证 RGBA→灰度转换使用 czkawka 兼容公式 `(R*77 + G*150 + B*29) >> 8`。
#[test]
fn test_rgba_to_grayscale_czkawka_formula() {
    // 纯红像素
    let red_rgba = vec![255, 0, 0, 255];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&red_rgba, 1, 1);
    assert_eq!(gray[0], ((255 * 77 + 0 * 150 + 0 * 29) >> 8) as u8);
    assert_eq!(gray[0], 76); // (255*77) >> 8 = 19635 >> 8 = 76

    // 纯绿像素
    let green_rgba = vec![0, 255, 0, 255];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&green_rgba, 1, 1);
    assert_eq!(gray[0], ((0 * 77 + 255 * 150 + 0 * 29) >> 8) as u8);
    assert_eq!(gray[0], 149); // (255*150) >> 8 = 38250 >> 8 = 149

    // 纯蓝像素
    let blue_rgba = vec![0, 0, 255, 255];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&blue_rgba, 1, 1);
    assert_eq!(gray[0], ((0 * 77 + 0 * 150 + 255 * 29) >> 8) as u8);
    assert_eq!(gray[0], 28); // (255*29) >> 8 = 7395 >> 8 = 28

    // 白色像素
    let white_rgba = vec![255, 255, 255, 255];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&white_rgba, 1, 1);
    assert_eq!(gray[0], ((255 * 77 + 255 * 150 + 255 * 29) >> 8) as u8);
    assert_eq!(gray[0], 255); // (255*256) >> 8 = 255

    // 黑色像素
    let black_rgba = vec![0, 0, 0, 255];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&black_rgba, 1, 1);
    assert_eq!(gray[0], 0);
}

/// 验证 RGBA→灰度转换 alpha 通道被忽略。
#[test]
fn test_rgba_to_grayscale_alpha_ignored() {
    let with_alpha_0 = vec![128, 64, 32, 0];
    let with_alpha_255 = vec![128, 64, 32, 255];
    let gray_0 = PerceptualHasher::rgba_to_grayscale_cpu(&with_alpha_0, 1, 1);
    let gray_255 = PerceptualHasher::rgba_to_grayscale_cpu(&with_alpha_255, 1, 1);
    assert_eq!(gray_0[0], gray_255[0], "Alpha 通道应被忽略");
}

/// 验证多像素 RGBA→灰度转换。
#[test]
fn test_rgba_to_grayscale_multi_pixel() {
    let rgba = vec![
        255, 0, 0, 255, // 红
        0, 255, 0, 255, // 绿
        0, 0, 255, 255, // 蓝
        255, 255, 255, 255, // 白
    ];
    let gray = PerceptualHasher::rgba_to_grayscale_cpu(&rgba, 2, 2);
    assert_eq!(gray.len(), 4);
    assert_eq!(gray[0], 76); // 红
    assert_eq!(gray[1], 149); // 绿
    assert_eq!(gray[2], 28); // 蓝
    assert_eq!(gray[3], 255); // 白
}

// ==================== compute_similar_pairs 测试 ====================

/// 测试 compute_similar_pairs 的边界情况。
#[test]
fn test_compute_similar_pairs_empty() {
    // 空 hashes
    let ctx = GpuContext::new_sync().unwrap_or_else(|_| {
        // 无 GPU 环境跳过
        return GpuContext::new_sync().unwrap_or_else(|_| {
            eprintln!("跳过：无 GPU 环境");
            panic!("需要 GPU 环境运行此测试");
        });
    });
    let mut ctx = ctx;
    let matcher = GpuHashMatcherBytes::new(&mut ctx).unwrap();
    let empty: Vec<HashBytes> = vec![];
    let result = matcher.compute_similar_pairs(&ctx, &empty, 10).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_compute_similar_pairs_single() {
    let ctx = GpuContext::new_sync().unwrap_or_else(|_| {
        panic!("需要 GPU 环境运行此测试");
    });
    let mut ctx = ctx;
    let matcher = GpuHashMatcherBytes::new(&mut ctx).unwrap();
    let single = vec![HashBytes::from_u64(0x12345678)];
    let result = matcher.compute_similar_pairs(&ctx, &single, 10).unwrap();
    assert!(result.is_empty(), "单个哈希不应产生任何匹配对");
}

#[test]
fn test_compute_similar_pairs_identical_hashes() {
    let ctx = GpuContext::new_sync().unwrap_or_else(|_| {
        panic!("需要 GPU 环境运行此测试");
    });
    let mut ctx = ctx;
    let matcher = GpuHashMatcherBytes::new(&mut ctx).unwrap();

    // 两个相同的哈希
    let hashes = vec![
        HashBytes::from_u64(0xAABBCCDD),
        HashBytes::from_u64(0xAABBCCDD),
    ];

    // tolerance=0 应匹配（距离=0，但 i!=j）
    let result = matcher.compute_similar_pairs(&ctx, &hashes, 0).unwrap();
    assert_eq!(result.len(), 2, "两个相同哈希应产生 2 个方向的对");
    // (0, 1, 0) 和 (1, 0, 0)
    assert!(result.contains(&(0, 1, 0)));
    assert!(result.contains(&(1, 0, 0)));
}

#[test]
fn test_compute_similar_pairs_tolerance_filter() {
    let ctx = GpuContext::new_sync().unwrap_or_else(|_| {
        panic!("需要 GPU 环境运行此测试");
    });
    let mut ctx = ctx;
    let matcher = GpuHashMatcherBytes::new(&mut ctx).unwrap();

    // 使用清晰区分的哈希值：
    // hash[0] = 0x00...00 (全零)
    // hash[1] = 0x00...01 (与 hash[0] 距离=1)
    // hash[2] = 0x00...FF (与 hash[0] 距离=8, 与 hash[1] 距离=7)
    let hashes = vec![
        HashBytes::from_u64(0x0000000000000000),
        HashBytes::from_u64(0x0000000000000001), // 距离=1 (与 hash[0])
        HashBytes::from_u64(0x00000000000000FF), // 距离=8 (与 hash[0]), 距离=7 (与 hash[1])
    ];

    // 打印距离矩阵用于调试
    let matrix = matcher.compute_distance_matrix(&ctx, &hashes, &hashes).unwrap();
    eprintln!("距离矩阵:");
    for (i, row) in matrix.iter().enumerate() {
        eprintln!("  hash[{}] vs all: {:?}", i, row);
    }

    // tolerance=0 不应匹配（所有距离 > 0）
    let result = matcher.compute_similar_pairs(&ctx, &hashes, 0).unwrap();
    assert!(result.is_empty(), "tolerance=0 不应匹配任何对");

    // tolerance=1 应匹配距离=1 的对
    let result = matcher.compute_similar_pairs(&ctx, &hashes, 1).unwrap();
    assert!(
        result.iter().any(|&(p, c, d)| d == 1 && p != c),
        "tolerance=1 应匹配距离 1 的对, 实际结果: {:?}",
        result
    );

    // tolerance=8 应匹配所有距离 ≤ 8 的对
    let result = matcher.compute_similar_pairs(&ctx, &hashes, 8).unwrap();
    assert!(
        result.iter().any(|&(p, c, d)| d == 8 && p != c),
        "tolerance=8 应匹配距离 8 的对, 实际结果: {:?}",
        result
    );

    // 验证排序（按 distance 升序）
    for i in 1..result.len() {
        assert!(
            result[i - 1].2 <= result[i].2,
            "结果应按 distance 升序排序"
        );
    }
}

// ==================== compute_similar_pairs_asymmetric 测试 ====================

#[test]
fn test_compute_similar_pairs_asymmetric() {
    let ctx = GpuContext::new_sync().unwrap_or_else(|_| {
        panic!("需要 GPU 环境运行此测试");
    });
    let mut ctx = ctx;
    let matcher = GpuHashMatcherBytes::new(&mut ctx).unwrap();

    let ref_hashes = vec![
        HashBytes::from_u64(0xFFFF),
        HashBytes::from_u64(0xAAAA),
    ];
    let normal_hashes = vec![
        HashBytes::from_u64(0xFFFF), // 与 ref[0] 相同
        HashBytes::from_u64(0x5555), // 与 ref[0] 距离=16
    ];

    let result = matcher
        .compute_similar_pairs_asymmetric(&ctx, &ref_hashes, &normal_hashes, 0)
        .unwrap();
    // tolerance=0 只匹配完全相同
    assert!(result.contains(&(0, 0, 0)), "ref[0] 和 normal[0] 应匹配");
}

// ==================== GPU 上下文集成测试 ====================

#[test]
fn test_new_for_integration_returns_context_or_unavailable() {
    match GpuContext::new_for_integration() {
        Ok(ctx) => {
            // GPU 可用，验证上下文
            assert!(!ctx.adapter_info().is_empty());
        }
        Err(GpuError::GpuUnavailable) => {
            // 无 GPU 环境，这是合法的
            eprintln!("GPU 不可用，new_for_integration 正确返回 GpuUnavailable");
        }
        Err(e) => {
            panic!("意外的错误类型: {:?}", e);
        }
    }
}

// ==================== GPU vs CPU 哈希一致性测试 ====================

/// 测试 RGBA 输入与灰度输入产生相同的哈希结果。
#[test]
fn test_compute_from_rgba_matches_grayscale() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("跳过：无 GPU 环境");
            return;
        }
    };

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Gradient).unwrap();

    // 创建一个 8x8 灰度图像
    let gray_data: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
    let dimensions = vec![(8u32, 8u32)];

    // 从灰度数据计算哈希
    let gray_hashes = hasher.compute(&ctx, &[gray_data.clone()], &dimensions).unwrap();

    // 从 RGBA 数据计算哈希（灰度→RGBA：每个像素 R=G=B=gray, A=255）
    let rgba_data: Vec<u8> = gray_data
        .iter()
        .flat_map(|&g| vec![g, g, g, 255])
        .collect();

    let rgba_hashes = hasher
        .compute_from_rgba(&ctx, &[rgba_data], &dimensions)
        .unwrap();

    assert_eq!(
        gray_hashes, rgba_hashes,
        "RGBA 输入与灰度输入应产生相同哈希"
    );
}

/// 测试 compute_to_hash_bytes 返回正确的 HashBytes。
#[test]
fn test_compute_to_hash_bytes() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("跳过：无 GPU 环境");
            return;
        }
    };

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let gray_data = vec![128u8; 64]; // 8x8 灰度
    let dimensions = vec![(8u32, 8u32)];

    let hash_bytes = hasher
        .compute_to_hash_bytes(&ctx, &[gray_data], &dimensions)
        .unwrap();

    assert_eq!(hash_bytes.len(), 1);
    // hash_size=8 → 64 bit → 8 bytes
    assert_eq!(hash_bytes[0].byte_len(), 8);
}

/// 测试 compute_from_rgba_to_hash_bytes 返回正确的 HashBytes。
#[test]
fn test_compute_from_rgba_to_hash_bytes() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("跳过：无 GPU 环境");
            return;
        }
    };

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();

    let rgba_data = vec![128u8; 64 * 4]; // 8x8 RGBA
    let dimensions = vec![(8u32, 8u32)];

    let hash_bytes = hasher
        .compute_from_rgba_to_hash_bytes(&ctx, &[rgba_data], &dimensions)
        .unwrap();

    assert_eq!(hash_bytes.len(), 1);
    assert_eq!(hash_bytes[0].byte_len(), 8); // 64-bit hash
}

// ==================== CzkawkaGpuAccelerator 测试 ====================

/// 测试 CzkawkaGpuAccelerator 创建失败时返回 GpuUnavailable（无 GPU 环境）或成功。
#[test]
fn test_czkawka_gpu_accelerator_creation() {
    match CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient) {
        Ok(accel) => {
            // GPU 可用，验证加速器
            assert!(!accel.shared_context().lock().unwrap().adapter_info().is_empty());
            eprintln!("CzkawkaGpuAccelerator 创建成功");
        }
        Err(GpuError::GpuUnavailable) => {
            eprintln!("GPU 不可用，跳过加速器测试");
        }
        Err(e) => {
            panic!("意外的错误类型: {:?}", e);
        }
    }
}

/// 测试 CzkawkaGpuAccelerator 无效 hash_size 返回 InvalidInput。
#[test]
fn test_czkawka_gpu_accelerator_invalid_hash_size() {
    let result = CzkawkaGpuAccelerator::new(7, HashAlgorithm::Mean);
    assert!(
        matches!(result, Err(GpuError::InvalidInput(_))),
        "hash_size=7 应返回 InvalidInput"
    );
}

/// 测试跨阶段共享 GPU 上下文：哈希计算 → 距离匹配。
#[test]
fn test_czkawka_gpu_accelerator_cross_stage_shared_context() {
    let accelerator = match CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient) {
        Ok(a) => a,
        Err(GpuError::GpuUnavailable) => {
            eprintln!("GPU 不可用，跳过跨阶段测试");
            return;
        }
        Err(e) => panic!("意外的错误类型: {:?}", e),
    };

    // 阶段 1：计算 RGBA 图像的哈希
    let rgba1 = vec![100u8; 16 * 16 * 4]; // 16×16 RGBA
    let rgba2 = vec![100u8; 16 * 16 * 4]; // 相同图像
    let rgba3 = vec![200u8; 16 * 16 * 4]; // 不同图像
    let dims = vec![(16u32, 16u32), (16u32, 16u32), (16u32, 16u32)];

    let hashes = accelerator
        .compute_hashes(&[rgba1, rgba2, rgba3], &dims)
        .unwrap();
    assert_eq!(hashes.len(), 3);
    assert_eq!(hashes[0].byte_len(), 8); // 64-bit hash

    // hash[0] 和 hash[1] 应相同（相同图像）
    assert_eq!(hashes[0], hashes[1], "相同图像应产生相同哈希");

    // 阶段 2：查找相似对（使用阶段 1 的哈希）
    let pairs = accelerator.find_similar_pairs(&hashes, 0).unwrap();

    // tolerance=0 应找到 hash[0] 和 hash[1] 的匹配对（距离=0）
    assert!(
        pairs.contains(&(0, 1, 0)),
        "应找到 (0, 1, 0) 匹配对, 实际: {:?}",
        pairs
    );
    assert!(
        pairs.contains(&(1, 0, 0)),
        "应找到 (1, 0, 0) 匹配对, 实际: {:?}",
        pairs
    );
}
