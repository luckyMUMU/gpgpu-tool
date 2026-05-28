use gpgpu_tool::{
    tasks::gaussian_blur::GpuGaussianBlur,
    tasks::hash_common::PerceptualHashComputer,
    tasks::mean_hash::MeanHashComputer,
    tasks::phasher::{HashAlgorithm, PerceptualHasher},
    GpuContext,
};

mod common;
use common::test_data;

#[test]
fn test_blur_resize_hash_pipeline_mean() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let hasher = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.5, 5)
        .expect("创建带模糊的 PerceptualHasher 失败");

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);
    let dimensions = vec![(width, height)];

    let hashes = hasher.compute(&ctx, &[pixels], &dimensions).expect("计算失败");
    assert_eq!(hashes.len(), 1, "应返回 1 个哈希值");
}

#[test]
fn test_blur_resize_hash_pipeline_batch() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let hasher = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.0, 3)
        .expect("创建带模糊的 PerceptualHasher 失败");

    let width = 64u32;
    let height = 64u32;
    let images: Vec<Vec<u8>> = (0..5)
        .map(|_| test_data::random_image((width * height) as usize))
        .collect();
    let dimensions = vec![(width, height); 5];

    let hashes = hasher.compute(&ctx, &images, &dimensions).expect("计算失败");
    assert_eq!(hashes.len(), 5, "应返回 5 个哈希值");
}

#[test]
fn test_blur_vs_no_blur_format_consistent() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let hasher_no_blur = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, true)
        .expect("创建无模糊 PerceptualHasher 失败");
    let hasher_with_blur = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.5, 5)
        .expect("创建带模糊 PerceptualHasher 失败");

    let width = 64u32;
    let height = 64u32;
    let pixels = test_data::gradient_image((width * height) as usize);
    let dimensions = vec![(width, height)];

    let hashes_no_blur = hasher_no_blur.compute(&ctx, std::slice::from_ref(&pixels), &dimensions).expect("计算失败");
    let hashes_with_blur = hasher_with_blur.compute(&ctx, std::slice::from_ref(&pixels), &dimensions).expect("计算失败");

    assert_eq!(hashes_no_blur.len(), hashes_with_blur.len(),
        "带模糊和不带模糊的结果格式应一致");
}

#[test]
fn test_blur_different_algorithms() {
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
    let dimensions = vec![(width, height)];

    for algorithm in algorithms {
        let hasher = PerceptualHasher::with_blur(&mut ctx, algorithm, 1.0, 3)
            .unwrap_or_else(|_| panic!("创建 {:?} 算法失败", algorithm));
        let hashes = hasher.compute(&ctx, std::slice::from_ref(&pixels), &dimensions)
            .unwrap_or_else(|_| panic!("{:?} 算法计算失败", algorithm));
        assert!(!hashes.is_empty(), "{:?} 算法应返回哈希值", algorithm);
    }
}

#[test]
fn test_blur_resize_hash_matches_step_by_step() {
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

    let blur = GpuGaussianBlur::new(&mut ctx).expect("创建 GpuGaussianBlur 失败");
    let hasher = MeanHashComputer::new(&mut ctx).expect("创建 MeanHashComputer 失败");

    let blurred = blur.blur(&ctx, &pixels, width, height, 5, 1.5).expect("模糊失败");

    let target_w = 8u32;
    let target_h = 8u32;
    let resized: Vec<u8> = {
        let src_w = width as usize;
        let src_h = height as usize;
        let mut output = Vec::with_capacity((target_w * target_h) as usize);
        let x_ratio = src_w as f64 / target_w as f64;
        let y_ratio = src_h as f64 / target_h as f64;
        for dy in 0..target_h {
            let y0 = (dy as f64 * y_ratio) as usize;
            let y1 = (((dy as f64 + 1.0) * y_ratio).min(src_h as f64)) as usize;
            for dx in 0..target_w {
                let x0 = (dx as f64 * x_ratio) as usize;
                let x1 = (((dx as f64 + 1.0) * x_ratio).min(src_w as f64)) as usize;
                let mut sum: u64 = 0;
                let mut count: u64 = 0;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += blurred[y * src_w + x] as u64;
                        count += 1;
                    }
                }
                output.push((sum / count.max(1)) as u8);
            }
        }
        output
    };

    let step_by_step_hash = hasher.compute(&ctx, &[resized]).expect("计算失败")[0];

    let pipeline_hasher = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.5, 5)
        .expect("创建带模糊的 PerceptualHasher 失败");
    let dimensions = vec![(width, height)];
    let pipeline_hash = pipeline_hasher.compute(&ctx, &[pixels], &dimensions).expect("计算失败")[0];

    assert_eq!(step_by_step_hash, pipeline_hash,
        "零拷贝管线结果应与逐步执行结果一致");
}

#[test]
fn test_blur_target_size_no_resize() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let hasher = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.0, 3)
        .expect("创建带模糊的 PerceptualHasher 失败");

    let target = hasher.target_size();
    let pixels = test_data::random_image((target.0 * target.1) as usize);
    let dimensions = vec![target];

    let hashes = hasher.compute(&ctx, &[pixels], &dimensions).expect("计算失败");
    assert_eq!(hashes.len(), 1, "目标尺寸图像 + 模糊应返回 1 个哈希值");
}

#[test]
fn test_blur_kernel_sizes() {
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
    let dimensions = vec![(width, height)];

    for kernel_size in [3u32, 5, 7] {
        let hasher = PerceptualHasher::with_blur(&mut ctx, HashAlgorithm::Mean, 1.0, kernel_size)
            .unwrap_or_else(|_| panic!("kernel_size={} 创建失败", kernel_size));
        let hashes = hasher.compute(&ctx, std::slice::from_ref(&pixels), &dimensions)
            .unwrap_or_else(|_| panic!("kernel_size={} 计算失败", kernel_size));
        assert_eq!(hashes.len(), 1, "kernel_size={} 应返回 1 个哈希值", kernel_size);
    }
}
