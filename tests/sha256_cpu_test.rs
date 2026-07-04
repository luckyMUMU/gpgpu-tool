use gpgpu_tool::{ComputeBackend, GpuContext, Sha256Cpu, tasks::sha256::Sha256Computer};

#[test]
fn test_sha256_cpu_basic() {
    let cpu = Sha256Cpu::new();
    let messages = vec![b"hello world".to_vec()];
    let results = cpu.compute(&messages).unwrap();

    let expected: [u8; 32] = hex::decode("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(results[0], expected);
}

#[test]
fn test_sha256_cpu_empty_string() {
    let cpu = Sha256Cpu::new();
    let messages = vec![vec![]];
    let results = cpu.compute(&messages).unwrap();

    let expected: [u8; 32] = hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(results[0], expected);
}

#[test]
fn test_sha256_cpu_batch() {
    let cpu = Sha256Cpu::new();
    let messages = vec![b"hello".to_vec(), b"world".to_vec(), b"test".to_vec()];
    let results = cpu.compute(&messages).unwrap();
    assert_eq!(results.len(), 3);

    use sha2::{Digest, Sha256};
    for (i, msg) in messages.iter().enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(msg);
        let expected = hasher.finalize();
        assert_eq!(results[i].as_slice(), expected.as_slice(), "消息 {} 哈希不匹配", i);
    }
}

#[test]
fn test_sha256_cpu_long_message() {
    let cpu = Sha256Cpu::new();
    let message = vec![0xAAu8; 1024];
    let results = cpu.compute(std::slice::from_ref(&message)).unwrap();

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&message);
    let expected = hasher.finalize();
    assert_eq!(results[0].as_slice(), expected.as_slice());
}

#[test]
fn test_sha256_cpu_matches_gpu() {
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

    let cpu = Sha256Cpu::new();
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = vec![b"test message".to_vec(), b"another message".to_vec()];

    let cpu_results = cpu.compute(&messages).unwrap();
    let gpu_results = sha256.compute(&ctx, &messages).unwrap();

    assert_eq!(cpu_results.len(), gpu_results.len());
    for (i, (cpu_hash, gpu_hash)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        assert_eq!(cpu_hash, gpu_hash, "消息 {} CPU/GPU 结果不一致", i);
    }
}

#[test]
fn test_sha256_computer_cpu_fallback() {
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

    // CPU 降级模式下 Sha256Computer::new() 可能失败（纯 CPU 无设备），
    // 也可能成功（软件渲染适配器）。两种情况都验证 compute 的降级行为。
    let sha256 = match Sha256Computer::new(&mut ctx) {
        Ok(s) => s,
        Err(_) => {
            // 纯 CPU 模式下无法创建 Sha256Computer，直接用 Sha256Cpu 验证
            let cpu = Sha256Cpu::new();
            let messages = vec![b"hello".to_vec()];
            let results = cpu.compute(&messages).unwrap();
            let expected: [u8; 32] = hex::decode("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
                .unwrap()
                .try_into()
                .unwrap();
            assert_eq!(results[0], expected);
            return;
        }
    };

    let messages = vec![b"hello".to_vec()];
    let results = sha256.compute(&ctx, &messages).unwrap();

    let expected: [u8; 32] = hex::decode("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(results[0], expected);
}
