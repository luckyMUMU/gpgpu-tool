use std::time::Instant;

use image::GenericImageView;
use wgpu_compute_engine::GpuContext;
use wgpu_compute_engine::tasks::bktree::{BkTree, hamming_distance};
use wgpu_compute_engine::tasks::phasher::{HashAlgorithm, PerceptualHasher};

fn brute_force_search(hashes: &[u64], query: u64, threshold: u32) -> Vec<(u64, u32)> {
    hashes
        .iter()
        .filter_map(|&h| {
            let d = hamming_distance(h, query);
            if d <= threshold { Some((h, d)) } else { None }
        })
        .collect()
}

fn generate_clustered_hashes(clusters: usize, per_cluster: usize, spread: u32) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut hashes = Vec::with_capacity(clusters * per_cluster);
    for _ in 0..clusters {
        let center: u64 = rng.random();
        for _ in 0..per_cluster {
            let mut h = center;
            for _ in 0..spread {
                let bit = 1u64 << (rng.random_range(0..64));
                h ^= bit;
            }
            hashes.push(h);
        }
    }
    hashes
}

#[test]
fn test_bktree_correctness_small() {
    let mut tree = BkTree::new();
    let hashes = [0x0000000000000000, 0x0000000000000001, 0x0000000000000011, 0xFFFFFFFFFFFFFFFF];
    for &h in &hashes {
        tree.insert(h);
    }

    let results = tree.find(0x0000000000000000, 2);
    let mut found: Vec<u64> = results.iter().map(|&(h, _)| h).collect();
    found.sort();

    let mut expected: Vec<u64> = hashes
        .iter()
        .filter(|&&h| hamming_distance(h, 0x0000000000000000) <= 2)
        .copied()
        .collect();
    expected.sort();

    assert_eq!(found, expected);
}

#[test]
fn test_bktree_correctness_random() {
    use rand::Rng;
    let mut rng = rand::rng();

    let hashes: Vec<u64> = (0..1000).map(|_| rng.random()).collect();
    let tree = BkTree::from_hashes(hashes.clone());

    for _ in 0..50 {
        let query: u64 = rng.random();
        let threshold = rng.random_range(0..20);

        let mut tree_results = tree.find(query, threshold);
        tree_results.sort_by_key(|&(h, _)| h);

        let mut brute_results = brute_force_search(&hashes, query, threshold);
        brute_results.sort_by_key(|&(h, _)| h);

        assert_eq!(tree_results, brute_results, "query={:016x} threshold={}", query, threshold);
    }
}

#[test]
fn test_bktree_find_nearest() {
    let mut tree = BkTree::new();
    tree.insert(0b0000);
    tree.insert(0b0100);
    tree.insert(0b1111);

    let nearest = tree.find_nearest(0b0010);
    assert!(nearest.is_some());
    assert_eq!(nearest.unwrap().1, 1);

    let nearest = tree.find_nearest(0b0000);
    assert_eq!(nearest, Some((0b0000, 0)));
}

#[test]
fn test_bktree_with_gpu_hashes() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut ctx = match GpuContext::new_sync() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let data_dir = std::path::Path::new("tests/data");
    if !data_dir.exists() {
        return;
    }

    let mut images = Vec::new();
    let mut dims = Vec::new();
    for entry in std::fs::read_dir(data_dir).unwrap() {
        let path = entry.unwrap().path();
        if let Ok(img) = image::open(&path) {
            let luma = img.grayscale();
            let (w, h) = luma.dimensions();
            let pixels: Vec<u8> = luma.pixels().map(|(_, _, luma)| luma.0[0]).collect();
            images.push(pixels);
            dims.push((w, h));
        }
    }

    if images.is_empty() {
        return;
    }

    let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean).unwrap();
    let hashes = hasher.compute(&ctx, &images, &dims).unwrap();

    let tree = BkTree::from_hashes(hashes.clone());

    for &query in &hashes {
        let tree_results = tree.find(query, 0);
        assert!(tree_results.contains(&(query, 0)), "exact match not found");
    }

    let query = hashes[0];
    let similar = tree.find(query, 10);
    for &(h, d) in &similar {
        assert!(d <= 10, "distance {} exceeds threshold", d);
        assert_eq!(d, hamming_distance(h, query));
    }

    let nearest = tree.find_nearest(query);
    assert!(nearest.is_some());
    assert_eq!(nearest.unwrap().0, query);
}

#[test]
fn test_bktree_performance_clustered() {
    let sizes = [1_000, 10_000, 100_000];

    println!("\n=== BK-Tree 聚类数据性能测试 ===");

    for &size in &sizes {
        let clusters = size / 10;
        let per_cluster = 10;
        let hashes = generate_clustered_hashes(clusters, per_cluster, 5);

        let build_start = Instant::now();
        let tree = BkTree::from_hashes(hashes.clone());
        let build_time = build_start.elapsed();
        println!("  {} 条聚类哈希 BK-Tree 构建: {:?}", size, build_time);

        let queries: Vec<u64> = hashes[0..100.min(hashes.len())].to_vec();
        let threshold = 5u32;

        let brute_start = Instant::now();
        for &q in &queries {
            let _ = brute_force_search(&hashes, q, threshold);
        }
        let brute_time = brute_start.elapsed();

        let tree_start = Instant::now();
        for &q in &queries {
            let _ = tree.find(q, threshold);
        }
        let tree_time = tree_start.elapsed();

        let speedup = if tree_time.as_secs_f64() > 0.0 {
            brute_time.as_secs_f64() / tree_time.as_secs_f64()
        } else {
            f64::INFINITY
        };
        println!("  {} 条 × {} 次查询 (阈值={}): 暴力={:?} BK-Tree={:?} ({:.1}x)",
            size, queries.len(), threshold, brute_time, tree_time, speedup);

        let nearest_brute_start = Instant::now();
        for &q in &queries {
            let best = hashes.iter().map(|&h| (h, hamming_distance(h, q))).min_by_key(|&(_, d)| d);
            let _ = best;
        }
        let nearest_brute_time = nearest_brute_start.elapsed();

        let nearest_tree_start = Instant::now();
        for &q in &queries {
            let _ = tree.find_nearest(q);
        }
        let nearest_tree_time = nearest_tree_start.elapsed();

        let nearest_speedup = if nearest_tree_time.as_secs_f64() > 0.0 {
            nearest_brute_time.as_secs_f64() / nearest_tree_time.as_secs_f64()
        } else {
            f64::INFINITY
        };
        println!("  {} 条 × {} 次最近邻: 暴力={:?} BK-Tree={:?} ({:.1}x)",
            size, queries.len(), nearest_brute_time, nearest_tree_time, nearest_speedup);

        let query = hashes[0];
        let mut tree_results = tree.find(query, threshold);
        tree_results.sort_by_key(|&(h, _)| h);
        let mut brute_results = brute_force_search(&hashes, query, threshold);
        brute_results.sort_by_key(|&(h, _)| h);
        assert_eq!(tree_results, brute_results);
    }
}
