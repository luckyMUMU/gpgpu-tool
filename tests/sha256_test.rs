use wgpu_compute_engine::{tasks::sha256::Sha256Computer, GpuContext};

#[test]
fn test_gpu_context_init() {
    let ctx = GpuContext::new_sync();
    assert!(ctx.is_ok(), "GPU 上下文初始化失败: {:?}", ctx.err());
    let ctx = ctx.unwrap();
    let info = ctx.adapter_info();
    assert!(!info.is_empty(), "适配器信息不应为空");
}

#[test]
fn test_sha256_empty_string() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let result = sha256.compute(&ctx, &[vec![]]).expect("计算失败");
    let expected =
        hex::decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855").unwrap();
    assert_eq!(result[0].as_slice(), expected.as_slice());
}

#[test]
fn test_sha256_abc() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let result = sha256.compute(&ctx, &[b"abc".to_vec()]).expect("计算失败");
    let expected =
        hex::decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad").unwrap();
    assert_eq!(result[0].as_slice(), expected.as_slice());
}

#[test]
fn test_sha256_empty_batch() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let result = sha256.compute(&ctx, &[]).expect("计算失败");
    assert!(result.is_empty());
}

#[test]
fn test_sha256_batch() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = vec![b"hello".to_vec(), b"world".to_vec(), b"test".to_vec()];

    let result = sha256.compute(&ctx, &messages).expect("计算失败");
    assert_eq!(result.len(), 3);

    use sha2::{Digest, Sha256};
    for (i, msg) in messages.iter().enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(msg);
        let expected = hasher.finalize();
        assert_eq!(
            result[i].as_slice(),
            expected.as_slice(),
            "消息 {} 哈希不匹配",
            i
        );
    }
}

#[test]
fn test_sha256_56_bytes() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let message = vec![0xAAu8; 56];
    let result = sha256.compute(&ctx, &[message.clone()]).expect("计算失败");

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&message);
    let expected = hasher.finalize();
    assert_eq!(result[0].as_slice(), expected.as_slice());
}

#[test]
fn test_sha256_64_bytes() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let message = vec![0xBBu8; 64];
    let result = sha256.compute(&ctx, &[message.clone()]).expect("计算失败");

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&message);
    let expected = hasher.finalize();
    assert_eq!(result[0].as_slice(), expected.as_slice());
}

#[test]
fn test_sha256_1kb() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let message = vec![0xCCu8; 1024];
    let result = sha256.compute(&ctx, &[message.clone()]).expect("计算失败");

    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(&message);
    let expected = hasher.finalize();
    assert_eq!(result[0].as_slice(), expected.as_slice());
}
