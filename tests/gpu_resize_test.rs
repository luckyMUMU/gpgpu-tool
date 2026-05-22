use image::GenericImageView;
use wgpu_compute_engine::GpuContext;
use wgpu_compute_engine::tasks::phasher::{HashAlgorithm, PerceptualHasher};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");

fn load_all_images(dir: &str) -> Vec<(Vec<u8>, u32, u32)> {
    let mut results = Vec::new();
    let path = std::path::Path::new(dir);
    if !path.exists() { return results; }

    fn walk(dir: &std::path::Path, results: &mut Vec<(Vec<u8>, u32, u32)>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() { walk(&p, results); }
                else if let Some(ext) = p.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        if let Ok(img) = image::open(&p) {
                            let (w, h) = img.dimensions();
                            let luma = img.grayscale();
                            let gray: Vec<u8> = luma.pixels()
                                .map(|(_, _, luma)| luma.0[0])
                                .collect();
                            results.push((gray, w, h));
                        }
                    }
                }
            }
        }
    }

    walk(path, &mut results);
    results
}

#[test]
fn test_gpu_resize_correctness() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let images = load_all_images(DATA_DIR);
    if images.is_empty() {
        eprintln!("无测试图像，跳过");
        return;
    }

    let pixels: Vec<Vec<u8>> = images.iter().map(|(p, _, _)| p.clone()).collect();
    let dims: Vec<(u32, u32)> = images.iter().map(|(_, w, h)| (*w, *h)).collect();

    for algo in &[HashAlgorithm::Mean, HashAlgorithm::Block, HashAlgorithm::Gradient] {
        let hasher_cpu = PerceptualHasher::with_resize_mode(&mut ctx, *algo, false).unwrap();
        let hasher_gpu = PerceptualHasher::with_resize_mode(&mut ctx, *algo, true).unwrap();

        let hashes_cpu = hasher_cpu.compute(&ctx, &pixels, &dims).unwrap();
        let hashes_gpu = hasher_gpu.compute(&ctx, &pixels, &dims).unwrap();

        assert_eq!(hashes_cpu.len(), hashes_gpu.len(),
            "{:?}: CPU 和 GPU 哈希数量不一致", algo);

        let mismatches: Vec<_> = hashes_cpu.iter().zip(hashes_gpu.iter())
            .enumerate()
            .filter(|(_, (cpu, gpu))| cpu != gpu)
            .collect();

        if !mismatches.is_empty() {
            eprintln!("{:?}: {} / {} 哈希不匹配",
                algo, mismatches.len(), hashes_cpu.len());
            for (i, (cpu, gpu)) in mismatches.iter().take(5) {
                eprintln!("  [{}] CPU={:016x} GPU={:016x}", i, cpu, gpu);
            }
        }

        assert!(mismatches.is_empty(),
            "{:?}: GPU 缩放零拷贝流水线哈希结果与 CPU 不一致", algo);
    }
}

#[test]
fn test_gpu_resize_batch_consistency() {
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("GPU 不可用，跳过测试");
            return;
        }
    };

    let img = vec![128u8; 64 * 64];
    let pixels: Vec<Vec<u8>> = (0..10).map(|_| img.clone()).collect();
    let dims: Vec<(u32, u32)> = (0..10).map(|_| (64u32, 64u32)).collect();

    let hasher = PerceptualHasher::with_resize_mode(&mut ctx, HashAlgorithm::Mean, true).unwrap();
    let hashes = hasher.compute(&ctx, &pixels, &dims).unwrap();

    assert_eq!(hashes.len(), 10);
    let first = hashes[0];
    for (i, &h) in hashes.iter().enumerate() {
        assert_eq!(h, first, "相同图像应产生相同哈希，第 {} 张不匹配", i);
    }
}
