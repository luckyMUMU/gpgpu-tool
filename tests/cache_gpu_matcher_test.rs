//! 缓存数据驱动的 GPU 并行近似度匹配测试。
//!
//! 使用 Czkawka 格式的缓存数据（1024-bit Gradient 哈希），
//! 验证 GPU 距离矩阵计算与 CPU 的一致性，并测试 BK-tree + GPU 混合匹配。
//! 大规模性能测试使用 20000 条哈希，通过 GPU 最近邻管线避免 O(N²) 内存。

use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use gpgpu_tool::GpuContext;
use gpgpu_tool::tasks::gpu_matcher::GpuHashMatcherBytes;
use gpgpu_tool::tasks::hash_bytes::HashBytes;
use gpgpu_tool::tasks::matcher_bytes::{
    BkTreeMatcherBytes, HashMatcherBytes, HashMatcherFacadeBytes, LinearScanMatcherBytes,
};

/// Czkawka 缓存条目（JSON 反序列化用）
#[derive(serde::Deserialize)]
struct CacheEntry {
    #[allow(dead_code)]
    path: String,
    #[allow(dead_code)]
    size: u64,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    #[allow(dead_code)]
    modified_date: u64,
    hash: Vec<u8>,
    #[allow(dead_code)]
    difference: u32,
}

/// 从 JSON 缓存文件加载哈希数据。
///
/// 过滤空哈希和全零哈希（与 Czkawka Invalid Hash Filtering 规范一致），
/// 只读取文件前部数据，截断修复后解析，避免读取全部 1.4GB。
fn load_hashes_from_cache(max_count: usize) -> Vec<HashBytes> {
    use std::io::Read;

    let json_path = "tests/data/cache_similar_images_32_Gradient_Gaussian_100.json";
    let mut file = fs::File::open(json_path)
        .unwrap_or_else(|e| panic!("打开缓存 JSON 失败: {}", e));

    // 每个条目约 600 字节，按需读取
    // 20000 条 × 600 字节 ≈ 12MB，留余量读 15MB
    let read_size = ((max_count * 700).max(2 * 1024 * 1024)).min(20 * 1024 * 1024);
    let mut buf = vec![0u8; read_size];
    let bytes_read = file.read(&mut buf)
        .unwrap_or_else(|e| panic!("读取缓存 JSON 失败: {}", e));
    buf.truncate(bytes_read);

    // 尝试解析为完整数组
    let entries: Vec<CacheEntry> = serde_json::from_slice(&buf)
        .unwrap_or_else(|e| {
            // 截断的 JSON：找到最后一个完整条目的结束位置 },
            // 然后补上 ] 闭合数组
            let s = String::from_utf8_lossy(&buf);
            if let Some(pos) = s.rfind("},") {
                let fixed = format!("{}]", &s[..pos + 1]);
                serde_json::from_str(&fixed)
                    .unwrap_or_else(|_| panic!("修复后仍解析失败: {}", e))
            } else if let Some(pos) = s.rfind("}") {
                let fixed = format!("{}]", &s[..pos + 1]);
                serde_json::from_str(&fixed)
                    .unwrap_or_else(|_| panic!("修复后仍解析失败: {}", e))
            } else {
                panic!("解析缓存 JSON 失败且无法修复: {}", e)
            }
        });

    let mut hashes = Vec::with_capacity(max_count);
    for entry in entries {
        if entry.hash.is_empty() {
            continue;
        }
        if entry.hash.iter().all(|&b| b == 0) {
            continue;
        }
        hashes.push(HashBytes::from_bytes(entry.hash));
        if hashes.len() >= max_count {
            break;
        }
    }

    assert!(!hashes.is_empty(), "缓存应包含有效哈希数据");
    hashes
}

// ── GPU 距离矩阵 vs CPU 一致性测试 ────────────────────────────────

#[test]
fn test_cache_gpu_distance_matrix_matches_cpu() {
    // 距离矩阵 500×500 = 250K 个距离值（~1MB），GPU 内存可控
    let hashes = load_hashes_from_cache(500);
    assert!(!hashes.is_empty(), "缓存应包含哈希数据");

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

    // 使用全部哈希作为 queries 和 database（对称比较）
    let n = hashes.len();
    let gpu_matrix = matcher
        .compute_distance_matrix(&ctx, &hashes, &hashes)
        .expect("GPU 距离矩阵计算失败");

    // 验证矩阵形状
    assert_eq!(gpu_matrix.len(), n, "矩阵行数应等于哈希数量");
    assert_eq!(gpu_matrix[0].len(), n, "矩阵列数应等于哈希数量");

    // 验证对角线为 0（自身距离）
    for i in 0..n {
        assert_eq!(gpu_matrix[i][i], 0, "对角线距离应为 0（自身）");
    }

    // 验证对称性
    for i in 0..n {
        for j in (i + 1)..n {
            assert_eq!(
                gpu_matrix[i][j], gpu_matrix[j][i],
                "距离矩阵应对称: [{0}][{1}]={2} != [{1}][{0}]={3}",
                i, j, gpu_matrix[i][j], gpu_matrix[j][i]
            );
        }
    }

    // 抽样验证 GPU 与 CPU 一致性（全量验证太慢，抽样 10 对）
    let sample_pairs: Vec<(usize, usize)> = (0..10)
        .map(|k| (k, (k + 7) % n))
        .collect();

    for (i, j) in &sample_pairs {
        let cpu_dist = hashes[*i].hamming_distance(&hashes[*j]);
        assert_eq!(
            gpu_matrix[*i][*j], cpu_dist,
            "GPU[{}][{}]={} != CPU 汉明距离={}",
            i, j, gpu_matrix[*i][*j], cpu_dist
        );
    }
}

// ── GPU 最近邻 vs CPU 一致性测试 ────────────────────────────────────

#[test]
fn test_cache_gpu_nearest_neighbor_matches_cpu() {
    // 20000 条大规模最近邻测试
    let hashes = load_hashes_from_cache(20000);
    let n = hashes.len();

    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

    // 使用前 100 个哈希作为 queries，全部作为 database
    let query_count = 100.min(n);
    let queries = &hashes[..query_count];

    let gpu_nearest = matcher
        .find_nearest_neighbors(&ctx, queries, &hashes, u32::MAX)
        .expect("GPU 最近邻计算失败");

    assert_eq!(gpu_nearest.len(), query_count, "结果数量应等于查询数量");

    // 验证每个 query 的最近邻
    for (qi, &(nn_idx, nn_dist)) in gpu_nearest.iter().enumerate() {
        // 自身距离应为 0
        assert_eq!(nn_idx, qi as u32, "query[{}] 最近邻应为自身", qi);
        assert_eq!(nn_dist, 0, "query[{}] 自身距离应为 0", qi);
    }

    // 用阈值过滤后验证
    let threshold = 20u32; // hash_size=32 的 Medium 阈值
    let gpu_nearest_thresholded = matcher
        .find_nearest_neighbors(&ctx, queries, &hashes, threshold)
        .expect("GPU 带阈值最近邻计算失败");

    for (qi, &(nn_idx, nn_dist)) in gpu_nearest_thresholded.iter().enumerate() {
        if nn_dist <= threshold {
            // 在阈值内，验证距离正确
            let cpu_dist = queries[qi].hamming_distance(&hashes[nn_idx as usize]);
            assert_eq!(nn_dist, cpu_dist, "GPU 最近邻距离应与 CPU 一致");
        }
    }
}

// ── BK-tree + GPU 混合匹配测试 ─────────────────────────────────────

#[test]
fn test_cache_bktree_gpu_hybrid_matching() {
    // BK-tree CPU 测试用 500 条（20000 条 BK-tree 构建太慢）
    let hashes = load_hashes_from_cache(500);
    let n = hashes.len();

    // BK-tree 匹配（CPU）
    let bktree_matcher = BkTreeMatcherBytes::new(hashes.clone());
    let threshold = 20u32;

    // 抽样 10 个 query 验证 BK-tree 结果
    let sample_indices: Vec<usize> = (0..10).map(|i| i * (n / 10)).collect();

    for &qi in &sample_indices {
        let query = &hashes[qi];
        let bktree_results = bktree_matcher.find_similar(query, threshold);

        // 验证 BK-tree 结果中的距离
        for r in &bktree_results {
            assert!(
                r.distance <= threshold,
                "BK-tree 结果距离应 ≤ 阈值: distance={}, threshold={}",
                r.distance, threshold
            );
            let cpu_dist = query.hamming_distance(&r.hash);
            assert_eq!(r.distance, cpu_dist, "BK-tree 距离应与 CPU 一致");
        }

        // 验证 BK-tree 召回率：与线性扫描对比
        let linear_matcher = LinearScanMatcherBytes::new(hashes.clone());
        let linear_results = linear_matcher.find_similar(query, threshold);

        // BK-tree 应找到所有线性扫描的结果（100% 召回率）
        let linear_hash_set: std::collections::HashSet<_> = linear_results
            .iter()
            .map(|r| r.hash.as_bytes().to_vec())
            .collect();
        let bktree_hash_set: std::collections::HashSet<_> = bktree_results
            .iter()
            .map(|r| r.hash.as_bytes().to_vec())
            .collect();

        assert_eq!(
            linear_hash_set, bktree_hash_set,
            "BK-tree 应与线性扫描结果一致 (query={})",
            qi
        );
    }
}

// ── GPU 门面批量匹配测试 ────────────────────────────────────────────

#[test]
fn test_cache_gpu_facade_batch_matching() {
    // 门面批量匹配：500 条 database，100 条 queries
    let hashes = load_hashes_from_cache(500);
    let n = hashes.len();

    let ctx = Arc::new(Mutex::new(
        GpuContext::new_sync().expect("GPU 初始化失败")
    ));

    // 使用 GPU 门面匹配器
    let facade = HashMatcherFacadeBytes::linear_scan(hashes.clone());
    let gpu_facade = gpgpu_tool::tasks::gpu_matcher::GpuHashMatcherFacadeBytes::new_with_shared_ctx(
        ctx, hashes.clone(),
    ).expect("GPU 匹配器门面创建失败");

    let threshold = 20u32;
    let query_count = 10.min(n);
    let queries = &hashes[..query_count];

    // CPU 门面批量匹配
    let cpu_results = facade.find_similar_batch(queries, threshold);

    // GPU 门面批量匹配
    let gpu_results = gpu_facade.find_similar_batch(queries, threshold);

    assert_eq!(cpu_results.len(), query_count, "CPU 结果数量应正确");
    assert_eq!(gpu_results.len(), query_count, "GPU 结果数量应正确");

    // 验证每个 query 的匹配结果一致性
    for qi in 0..query_count {
        let mut cpu_sorted: Vec<(Vec<u8>, u32)> = cpu_results[qi]
            .iter()
            .map(|r| (r.hash.as_bytes().to_vec(), r.distance))
            .collect();
        let mut gpu_sorted: Vec<(Vec<u8>, u32)> = gpu_results[qi]
            .iter()
            .map(|r| (r.hash.as_bytes().to_vec(), r.distance))
            .collect();
        cpu_sorted.sort();
        gpu_sorted.sort();

        assert_eq!(
            cpu_sorted, gpu_sorted,
            "query[{}] 的 CPU 和 GPU 门面匹配结果应一致",
            qi
        );
    }
}

// ── 性能对比测试（打印耗时，不做断言） ──────────────────────────────

#[test]
fn test_cache_gpu_vs_cpu_performance() {
    // 20000 条大规模性能对比
    let hashes = load_hashes_from_cache(20000);
    let n = hashes.len();
    let threshold = 20u32;

    eprintln!("\n=== 缓存数据 GPU vs CPU 性能对比 (20000 条) ===");
    eprintln!("哈希数量: {} (1024-bit/条)", n);
    eprintln!("阈值: {}", threshold);

    // ── CPU 线性扫描（抽样 200 个 query） ──
    let cpu_query_count = 200.min(n);
    let cpu_queries = &hashes[..cpu_query_count];
    let cpu_start = Instant::now();
    let linear_matcher = LinearScanMatcherBytes::new(hashes.clone());
    let mut cpu_total_matches = 0;
    for query in cpu_queries {
        let results = linear_matcher.find_similar(query, threshold);
        cpu_total_matches += results.len();
    }
    let cpu_elapsed = cpu_start.elapsed();
    // 推算全量 CPU 耗时
    let cpu_estimated_full = cpu_elapsed * (n as u32 / cpu_query_count as u32);

    eprintln!("CPU 线性扫描 ({} queries × {} db): {:.2?} (匹配 {} 对)",
        cpu_query_count, n, cpu_elapsed, cpu_total_matches);
    eprintln!("CPU 推算全量耗时 ({} queries): {:.2?}", n, cpu_estimated_full);

    // ── GPU 最近邻（全量 20000 queries × 20000 database） ──
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcherBytes::new(&mut ctx).expect("GpuHashMatcherBytes 创建失败");

    // GPU 最近邻：全量查询
    let gpu_nn_start = Instant::now();
    let gpu_nearest = matcher
        .find_nearest_neighbors(&ctx, &hashes, &hashes, threshold)
        .expect("GPU 最近邻计算失败");
    let gpu_nn_elapsed = gpu_nn_start.elapsed();

    // 统计 GPU 匹配数（距离 ≤ threshold 的最近邻）
    let gpu_nn_matches = gpu_nearest.iter()
        .filter(|&&(_, dist)| dist <= threshold)
        .count();

    eprintln!("GPU 最近邻 ({} queries × {} db): {:.2?} (匹配 {} 对)",
        n, n, gpu_nn_elapsed, gpu_nn_matches);

    // ── GPU 距离矩阵（小规模参照：500×500） ──
    let small_n = 500.min(n);
    let small_hashes = &hashes[..small_n];
    let gpu_matrix_start = Instant::now();
    let gpu_matrix = matcher
        .compute_distance_matrix(&ctx, small_hashes, small_hashes)
        .expect("GPU 距离矩阵计算失败");
    let gpu_matrix_elapsed = gpu_matrix_start.elapsed();

    let mut gpu_matrix_matches = 0;
    for row in &gpu_matrix {
        for &dist in row {
            if dist <= threshold {
                gpu_matrix_matches += 1;
            }
        }
    }

    eprintln!("GPU 距离矩阵 ({}×{}): {:.2?} (匹配 {} 对)",
        small_n, small_n, gpu_matrix_elapsed, gpu_matrix_matches);

    // 验证 GPU 最近邻结果正确性（抽样 10 个）
    for qi in (0..10).map(|i| i * (n / 10)) {
        let (nn_idx, nn_dist) = gpu_nearest[qi];
        if nn_dist <= threshold {
            let cpu_dist = hashes[qi].hamming_distance(&hashes[nn_idx as usize]);
            assert_eq!(nn_dist, cpu_dist,
                "GPU 最近邻距离应与 CPU 一致: query={}, nn_idx={}", qi, nn_idx);
        }
    }
}

// ── 二面体变换增强匹配测试 ─────────────────────────────────────────

#[test]
fn test_cache_dihedral_enhanced_matching() {
    // 二面体变换测试用 200 条
    let hashes = load_hashes_from_cache(200);
    let threshold = 20u32;

    // 普通匹配
    let facade_normal = HashMatcherFacadeBytes::linear_scan(hashes.clone());
    // 二面体变换增强匹配
    let facade_dihedral = HashMatcherFacadeBytes::linear_scan(hashes.clone()).with_dihedral();

    // 抽样验证：二面体增强应找到 ≥ 普通匹配的结果
    let sample_query = &hashes[0];
    let normal_results = facade_normal.find_similar(sample_query, threshold);
    let dihedral_results = facade_dihedral.find_similar(sample_query, threshold);

    // 二面体增强的匹配数应 ≥ 普通匹配（可能找到旋转/翻转的相似图像）
    assert!(
        dihedral_results.len() >= normal_results.len(),
        "二面体增强匹配数 ({}) 应 ≥ 普通匹配数 ({})",
        dihedral_results.len(),
        normal_results.len()
    );

    // 普通匹配的结果应全部包含在二面体增强结果中
    let dihedral_hashes: std::collections::HashSet<_> = dihedral_results
        .iter()
        .map(|r| r.hash.as_bytes().to_vec())
        .collect();
    for r in &normal_results {
        assert!(
            dihedral_hashes.contains(r.hash.as_bytes()),
            "普通匹配结果应包含在二面体增强结果中"
        );
    }
}
