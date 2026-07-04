use gpgpu_tool::{
    ComputeBackend, GpuContext, HashSize, PHasherCpu,
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
};

mod common;
use common::hash_reference;
use common::test_data;

#[test]
fn test_phasher_cpu_mean() {
    let cpu = PHasherCpu::new(HashAlgorithm::Mean);
    let pixels = vec![128u8; 64];
    let result = cpu.compute(std::slice::from_ref(&pixels), 8, 8).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::mean_hash(&pixels, 8, 8);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_median() {
    let cpu = PHasherCpu::new(HashAlgorithm::Median);
    let pixels = test_data::gradient_image(64);
    let result = cpu.compute(std::slice::from_ref(&pixels), 8, 8).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::median_hash(&pixels, 8, 8);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_gradient() {
    let cpu = PHasherCpu::new(HashAlgorithm::Gradient);
    let pixels = test_data::gradient_image(8 * 9);
    let result = cpu.compute(std::slice::from_ref(&pixels), 8, 9).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::gradient_hash(&pixels, 8, 9);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_block() {
    let cpu = PHasherCpu::new(HashAlgorithm::Block);
    let pixels = test_data::gradient_image(16 * 16);
    let result = cpu.compute(std::slice::from_ref(&pixels), 16, 16).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::block_hash(&pixels, 16, 16);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_vert_gradient() {
    let cpu = PHasherCpu::new(HashAlgorithm::VertGradient);
    let pixels = test_data::gradient_image(9 * 8);
    let result = cpu.compute(std::slice::from_ref(&pixels), 9, 8).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::vert_gradient_hash(&pixels, 9, 8);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_double_gradient() {
    let cpu = PHasherCpu::new(HashAlgorithm::DoubleGradient);
    let pixels = test_data::gradient_image(9 * 9);
    let result = cpu.compute(std::slice::from_ref(&pixels), 9, 9).unwrap();
    assert_eq!(result.len(), 1);
    let expected = hash_reference::double_gradient_hash(&pixels, 9, 9);
    assert_eq!(result[0], expected);
}

#[test]
fn test_phasher_cpu_batch() {
    let cpu = PHasherCpu::new(HashAlgorithm::Mean);
    let img1 = vec![100u8; 64];
    let img2 = vec![200u8; 64];
    let img3 = test_data::gradient_image(64);
    let result = cpu.compute(&[img1.clone(), img2.clone(), img3.clone()], 8, 8).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], hash_reference::mean_hash(&img1, 8, 8));
    assert_eq!(result[1], hash_reference::mean_hash(&img2, 8, 8));
    assert_eq!(result[2], hash_reference::mean_hash(&img3, 8, 8));
}

#[test]
fn test_phasher_cpu_empty() {
    let cpu = PHasherCpu::new(HashAlgorithm::Mean);
    let result = cpu.compute(&[], 8, 8).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_phasher_cpu_uniform_image() {
    let cpu = PHasherCpu::new(HashAlgorithm::Mean);
    let pixels = vec![128u8; 64];
    let result = cpu.compute(&[pixels], 8, 8).unwrap();
    // 均匀图像所有像素等于均值，>= 均值应为全 1
    assert_eq!(result[0], u64::MAX);
}

#[test]
fn test_phasher_cpu_algorithm_accessor() {
    let cpu = PHasherCpu::new(HashAlgorithm::Gradient);
    assert_eq!(cpu.algorithm(), HashAlgorithm::Gradient);
}

#[test]
fn test_phasher_cpu_hash_size_accessor() {
    let cpu = PHasherCpu::with_hash_size(HashAlgorithm::Mean, HashSize::new(16));
    assert_eq!(cpu.hash_size(), HashSize::new(16));
}

#[test]
fn test_phasher_cpu_default() {
    let cpu = PHasherCpu::default();
    assert_eq!(cpu.algorithm(), HashAlgorithm::Mean);
    assert_eq!(cpu.hash_size(), HashSize::default());
}

#[test]
fn test_phasher_cpu_with_hash_size_16() {
    let cpu = PHasherCpu::with_hash_size(HashAlgorithm::Mean, HashSize::new(16));
    let pixels = test_data::gradient_image(16 * 16);
    let result = cpu.compute(&[pixels], 16, 16).unwrap();
    // hash_size=16 → 256 bit → 4 个 u64
    assert_eq!(result.len(), 4);
}

#[test]
fn test_phasher_cpu_all_algorithms_random() {
    let algorithms = [
        HashAlgorithm::Mean,
        HashAlgorithm::Median,
        HashAlgorithm::Gradient,
        HashAlgorithm::Block,
        HashAlgorithm::VertGradient,
        HashAlgorithm::DoubleGradient,
    ];

    for &algo in &algorithms {
        let cpu = PHasherCpu::new(algo);
        let (w, h) = algo.target_size_for(HashSize::default());
        let pixels = test_data::random_image((w * h) as usize);
        let result = cpu.compute(&[pixels], w, h);
        assert!(result.is_ok(), "{:?} 计算失败", algo);
        assert_eq!(result.unwrap().len(), 1, "{:?} 结果长度不正确", algo);
    }
}

#[test]
fn test_phasher_cpu_matches_gpu() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过对比测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Gpu {
        eprintln!("非 GPU 模式，跳过对比测试");
        return;
    }

    let algorithms = [
        HashAlgorithm::Mean,
        HashAlgorithm::Median,
        HashAlgorithm::Gradient,
        HashAlgorithm::Block,
        HashAlgorithm::VertGradient,
        HashAlgorithm::DoubleGradient,
    ];

    for &algo in &algorithms {
        let hasher = PerceptualHasher::new(&mut ctx, algo).expect("创建 PerceptualHasher 失败");
        let cpu = PHasherCpu::new(algo);
        let (w, h) = algo.target_size_for(HashSize::default());
        let pixels = test_data::random_image((w * h) as usize);
        let dimensions = vec![(w, h)];

        let gpu_result = hasher.compute(&ctx, std::slice::from_ref(&pixels), &dimensions)
            .unwrap_or_else(|e| panic!("{:?} GPU 计算失败: {}", algo, e));
        let cpu_result = cpu.compute(&[pixels], w, h)
            .unwrap_or_else(|e| panic!("{:?} CPU 计算失败: {}", algo, e));

        assert_eq!(gpu_result, cpu_result, "{:?} CPU/GPU 结果不一致", algo);
    }
}

#[test]
fn test_phasher_cpu_fallback_path() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("上下文创建失败，跳过降级路径测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Cpu {
        eprintln!("非 CPU 降级模式，跳过降级路径测试");
        return;
    }

    // CPU 降级模式下 PerceptualHasher::new() 可能失败
    let hasher = match PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean) {
        Ok(h) => h,
        Err(_) => {
            // 纯 CPU 模式下无法创建 PerceptualHasher，直接用 PHasherCpu 验证
            let cpu = PHasherCpu::new(HashAlgorithm::Mean);
            let pixels = vec![128u8; 64];
            let result = cpu.compute(&[pixels], 8, 8).unwrap();
            assert_eq!(result.len(), 1);
            return;
        }
    };

    let pixels = vec![128u8; 64];
    let dimensions = vec![(8u32, 8u32)];
    let result = hasher.compute(&ctx, &[pixels], &dimensions).unwrap();
    assert_eq!(result.len(), 1);
}

#[test]
fn test_phasher_cpu_fallback_with_resize() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("上下文创建失败，跳过降级路径测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Cpu {
        eprintln!("非 CPU 降级模式，跳过降级路径测试");
        return;
    }

    let hasher = match PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean) {
        Ok(h) => h,
        Err(_) => {
            // 纯 CPU 模式下无法创建 PerceptualHasher，直接用 PHasherCpu 验证缩放
            let cpu = PHasherCpu::new(HashAlgorithm::Mean);
            let pixels = vec![128u8; 256 * 256];
            let result = cpu.compute(&[pixels], 8, 8);
            // 图像尺寸不匹配，但 PHasherCpu 只使用前 w*h 个像素
            assert!(result.is_ok());
            return;
        }
    };

    // 传入大于目标尺寸的图像，测试 CPU 降级路径中的缩放
    let pixels = vec![128u8; 256 * 256];
    let dimensions = vec![(256u32, 256u32)];
    let result = hasher.compute(&ctx, &[pixels], &dimensions).unwrap();
    assert_eq!(result.len(), 1);
}
