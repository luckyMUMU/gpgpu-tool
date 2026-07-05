use gpgpu_tool::{
    GpuContext, GpuPipelineBuilder, GpuPipelineStep, HashSize,
    tasks::phasher::HashAlgorithm,
};

mod common;
use common::test_data;

#[test]
fn test_pipeline_hash_only() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);

    let hashes = GpuPipelineBuilder::new()
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &[pixels], &[width], &[height])
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 1, "应返回 1 张图像的哈希");
    assert_eq!(hashes[0].len(), 1, "64-bit 哈希应返回 1 个 u64");
}

#[test]
fn test_pipeline_resize_hash() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);

    let hashes = GpuPipelineBuilder::new()
        .resize(8, 8)
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &[pixels], &[width], &[height])
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 1, "应返回 1 张图像的哈希");
}

#[test]
fn test_pipeline_blur_resize_hash() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);

    let hashes = GpuPipelineBuilder::new()
        .blur(1.0, 5)
        .resize(8, 8)
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &[pixels], &[width], &[height])
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 1, "应返回 1 张图像的哈希");
}

#[test]
fn test_pipeline_blur_hash() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);

    let hashes = GpuPipelineBuilder::new()
        .blur(1.5, 3)
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &[pixels], &[width], &[height])
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 1, "应返回 1 张图像的哈希");
}

#[test]
fn test_pipeline_batch_images() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let images: Vec<Vec<u8>> = (0..5)
        .map(|_| test_data::random_image((width * height) as usize))
        .collect();
    let widths = vec![width; 5];
    let heights = vec![height; 5];

    let hashes = GpuPipelineBuilder::new()
        .blur(1.0, 3)
        .resize(8, 8)
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &images, &widths, &heights)
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 5, "应返回 5 张图像的哈希");
    for (i, h) in hashes.iter().enumerate() {
        assert_eq!(h.len(), 1, "第 {} 张图像应为 64-bit 哈希", i);
    }
}

#[test]
fn test_pipeline_different_algorithms() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let algorithms = [
        HashAlgorithm::Mean,
        HashAlgorithm::Median,
        HashAlgorithm::Gradient,
        HashAlgorithm::Block,
        HashAlgorithm::VertGradient,
        HashAlgorithm::DoubleGradient,
    ];

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::random_image((width * height) as usize);

    for algorithm in algorithms {
        let (tw, th) = algorithm.target_size_for(HashSize::default());
        let hashes = GpuPipelineBuilder::new()
            .resize(tw, th)
            .hash(algorithm, HashSize::default())
            .execute(&mut ctx, std::slice::from_ref(&pixels), &[width], &[height])
            .unwrap_or_else(|_| panic!("{:?} 算法管线执行失败", algorithm));

        assert!(!hashes.is_empty(), "{:?} 算法应返回哈希值", algorithm);
        assert!(!hashes[0].is_empty(), "{:?} 算法哈希不应为空", algorithm);
    }
}

#[test]
fn test_pipeline_hash_size_16() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);
    let hash_size = HashSize::new(16);

    let hashes = GpuPipelineBuilder::new()
        .resize(16, 16)
        .hash(HashAlgorithm::Mean, hash_size)
        .execute(&mut ctx, &[pixels], &[width], &[height])
        .expect("管线执行失败");

    assert_eq!(hashes.len(), 1, "应返回 1 张图像的哈希");
    assert_eq!(hashes[0].len(), 4, "256-bit 哈希应返回 4 个 u64");
}

#[test]
fn test_pipeline_validation_no_hash() {
    let builder = GpuPipelineBuilder::new()
        .blur(1.0, 5)
        .resize(8, 8);

    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let pixels = test_data::gradient_image(64 * 64);
    let result = builder.execute(&mut ctx, &[pixels], &[64], &[64]);
    assert!(result.is_err(), "缺少 Hash 步骤应返回错误");
}

#[test]
fn test_pipeline_validation_hash_not_last() {
    let builder = GpuPipelineBuilder::new()
        .hash(HashAlgorithm::Mean, HashSize::default())
        .blur(1.0, 5);

    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let pixels = test_data::gradient_image(64 * 64);
    let result = builder.execute(&mut ctx, &[pixels], &[64], &[64]);
    assert!(result.is_err(), "Hash 不是最后一步应返回错误");
}

#[test]
fn test_pipeline_validation_resize_mismatch() {
    let builder = GpuPipelineBuilder::new()
        .resize(16, 16)
        .hash(HashAlgorithm::Mean, HashSize::default());

    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let pixels = test_data::gradient_image(64 * 64);
    let result = builder.execute(&mut ctx, &[pixels], &[64], &[64]);
    assert!(result.is_err(), "Resize 尺寸与算法目标尺寸不匹配应返回错误");
}

#[test]
fn test_pipeline_validation_blur_after_resize() {
    let builder = GpuPipelineBuilder::new()
        .resize(8, 8)
        .blur(1.0, 5)
        .hash(HashAlgorithm::Mean, HashSize::default());

    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let pixels = test_data::gradient_image(64 * 64);
    let result = builder.execute(&mut ctx, &[pixels], &[64], &[64]);
    assert!(result.is_err(), "Blur 在 Resize 之后应返回错误");
}

#[test]
fn test_pipeline_steps_access() {
    let builder = GpuPipelineBuilder::new()
        .blur(1.0, 5)
        .resize(8, 8)
        .hash(HashAlgorithm::Mean, HashSize::default());

    let steps = builder.steps();
    assert_eq!(steps.len(), 3);

    assert!(matches!(steps[0], GpuPipelineStep::Blur { sigma: 1.0, kernel_size: 5 }));
    assert!(matches!(steps[1], GpuPipelineStep::Resize { width: 8, height: 8 }));
    assert!(matches!(steps[2], GpuPipelineStep::Hash { algorithm: HashAlgorithm::Mean, .. }));
}

#[test]
fn test_pipeline_empty_images() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let hashes = GpuPipelineBuilder::new()
        .hash(HashAlgorithm::Mean, HashSize::default())
        .execute(&mut ctx, &[], &[], &[])
        .expect("空图像列表应返回空结果");

    assert!(hashes.is_empty(), "空图像列表应返回空结果");
}

#[test]
fn test_pipeline_default() {
    let builder = GpuPipelineBuilder::default();
    assert!(builder.steps().is_empty(), "默认构建器应无步骤");
}
