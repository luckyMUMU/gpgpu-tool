#[cfg(feature = "pdq")]
use gpgpu_tool::{ComputeBackend, GpuContext, HashSize};

#[cfg(feature = "pdq")]
use gpgpu_tool::tasks::pdq_hash::{PdqHashCpu, PdqHashGpu};

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_cpu_basic() {
    // 创建 64×64 均匀灰度图像
    let pixels = vec![128u8; 64 * 64];
    let cpu = PdqHashCpu::new();
    let result = cpu.compute(&pixels).unwrap();

    // 均匀图像应成功计算，哈希为 4 个 u64
    assert_eq!(result.hash.len(), 4);
    // 均匀图像两次计算结果应一致
    let result2 = cpu.compute(&pixels).unwrap();
    assert_eq!(result.hash, result2.hash, "均匀图像的 PDQ 哈希应是确定性的");
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_cpu_gradient() {
    // 创建 64×64 渐变图像
    let pixels: Vec<u8> = (0..64)
        .flat_map(|row| {
            (0..64).map(move |col| {
                ((row * 64 + col) as f32 / (64 * 64) as f32 * 255.0) as u8
            })
        })
        .collect();
    let cpu = PdqHashCpu::new();
    let result = cpu.compute(&pixels).unwrap();

    // 渐变图像应有非零非全 1 哈希
    assert_ne!(result.hash, [0u64; 4], "渐变图像的 PDQ 哈希不应为全 0");
    assert_ne!(
        result.hash,
        [u64::MAX; 4],
        "渐变图像的 PDQ 哈希不应为全 1"
    );
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_cpu_deterministic() {
    let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();
    let cpu = PdqHashCpu::new();
    let r1 = cpu.compute(&pixels).unwrap();
    let r2 = cpu.compute(&pixels).unwrap();
    assert_eq!(r1.hash, r2.hash, "PDQ 哈希应是确定性的");
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_quality_score() {
    let pixels = vec![128u8; 64 * 64];
    let cpu = PdqHashCpu::new();
    let result = cpu.compute(&pixels).unwrap();
    assert!(
        result.quality >= 0.0 && result.quality <= 1.0,
        "质量评分应在 0-1 范围"
    );
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_cpu_input_validation() {
    let cpu = PdqHashCpu::new();
    // 输入尺寸不正确应返回错误
    let bad_pixels = vec![0u8; 100];
    let result = cpu.compute(&bad_pixels);
    assert!(result.is_err(), "非 64×64 输入应返回错误");
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_gpu_basic() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过 PDQ GPU 测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Gpu {
        eprintln!("非 GPU 模式，跳过 PDQ GPU 测试");
        return;
    }

    let gpu = match PdqHashGpu::new(&mut ctx) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("PdqHashGpu 创建失败，跳过");
            return;
        }
    };

    // 创建 64×64 均匀灰度图像
    let pixels = vec![128u8; 64 * 64];
    let hashes = gpu.compute(&ctx, &[pixels], HashSize::new(16)).unwrap();
    // 每张图输出 4 个 u64
    assert_eq!(hashes.len(), 4, "单张图像应输出 4 个 u64");
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_gpu_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过 PDQ GPU 批量测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Gpu {
        eprintln!("非 GPU 模式，跳过 PDQ GPU 批量测试");
        return;
    }

    let gpu = match PdqHashGpu::new(&mut ctx) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("PdqHashGpu 创建失败，跳过");
            return;
        }
    };

    // 批量 3 张 64×64 图像
    let images = vec![
        vec![128u8; 64 * 64],
        vec![0u8; 64 * 64],
        vec![255u8; 64 * 64],
    ];
    let hashes = gpu.compute(&ctx, &images, HashSize::new(16)).unwrap();
    // 3 张图 × 4 u64 = 12
    assert_eq!(hashes.len(), 12, "3 张图像应输出 12 个 u64");
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_cpu_gpu_consistency() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过 PDQ CPU/GPU 一致性测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Gpu {
        eprintln!("非 GPU 模式，跳过 PDQ CPU/GPU 一致性测试");
        return;
    }

    let gpu = match PdqHashGpu::new(&mut ctx) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("PdqHashGpu 创建失败，跳过");
            return;
        }
    };

    let cpu = PdqHashCpu::new();

    // 渐变图像：CPU 和 GPU 应产生相同哈希
    let pixels: Vec<u8> = (0..64)
        .flat_map(|row| {
            (0..64).map(move |col| {
                ((row * 64 + col) as f32 / (64 * 64) as f32 * 255.0) as u8
            })
        })
        .collect();

    let cpu_result = cpu.compute(&pixels).unwrap();
    let gpu_hashes = gpu.compute(&ctx, &[pixels.clone()], HashSize::new(16)).unwrap();

    let gpu_hash: [u64; 4] = [gpu_hashes[0], gpu_hashes[1], gpu_hashes[2], gpu_hashes[3]];
    assert_eq!(
        cpu_result.hash, gpu_hash,
        "CPU 和 GPU 的 PDQ 哈希应一致"
    );
}

#[cfg(feature = "pdq")]
#[test]
fn test_pdq_gpu_invalid_hash_size() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过 PDQ hash_size 验证测试");
            return;
        }
    };

    if ctx.backend() != ComputeBackend::Gpu {
        eprintln!("非 GPU 模式，跳过 PDQ hash_size 验证测试");
        return;
    }

    let gpu = match PdqHashGpu::new(&mut ctx) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("PdqHashGpu 创建失败，跳过");
            return;
        }
    };

    let pixels = vec![128u8; 64 * 64];
    let result = gpu.compute(&ctx, &[pixels.clone()], HashSize::new(8));
    assert!(result.is_err(), "hash_size=8 应返回错误，PDQ 固定为 256-bit");

    let result = gpu.compute(&ctx, &[pixels], HashSize::new(32));
    assert!(result.is_err(), "hash_size=32 应返回错误，PDQ 固定为 256-bit");
}
