use std::sync::{Arc, Mutex};

use gpgpu_tool::GpuContext;
use gpgpu_tool::tasks::gpu_matcher::{GpuHashMatcher, GpuHashMatcherFacade};
use gpgpu_tool::tasks::matcher::{HashMatcher, LinearScanMatcher};

#[test]
fn test_distance_matrix_basic() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcher::new(&mut ctx).expect("GpuHashMatcher 创建失败");

    // 2 queries, 3 database entries
    // Each u64 hash is split into 2 u32 (lo, hi)
    let queries: Vec<u64> = vec![0x0000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF];
    let database: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_000F,
        0xFFFF_FFFF_FFFF_FFFF,
    ];

    let matrix = matcher
        .compute_distance_matrix(&ctx, &queries, &database)
        .expect("距离矩阵计算失败");

    // 2x3 matrix
    assert_eq!(matrix.len(), 2, "应有 2 行（queries）");
    assert_eq!(matrix[0].len(), 3, "每行应有 3 列（database）");

    // query=0x0000 vs db entries:
    //   0x0000: distance = 0
    //   0x000F: distance = 4 (lower 4 bits differ)
    //   0xFFFF: distance = 64
    assert_eq!(matrix[0][0], 0, "0x0000 vs 0x0000 = 0");
    assert_eq!(matrix[0][1], 4, "0x0000 vs 0x000F = 4");
    assert_eq!(matrix[0][2], 64, "0x0000 vs 0xFFFF = 64");

    // query=0xFFFF vs db entries:
    //   0x0000: distance = 64
    //   0x000F: distance = 60 (64-4)
    //   0xFFFF: distance = 0
    assert_eq!(matrix[1][0], 64, "0xFFFF vs 0x0000 = 64");
    assert_eq!(matrix[1][1], 60, "0xFFFF vs 0x000F = 60");
    assert_eq!(matrix[1][2], 0, "0xFFFF vs 0xFFFF = 0");
}

#[test]
fn test_find_nearest_basic() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcher::new(&mut ctx).expect("GpuHashMatcher 创建失败");

    // 3 queries, 3 database entries
    let queries: Vec<u64> = vec![
        0x0000_0000_0000_0000, // nearest: db[0] (dist=0)
        0x0000_0000_0000_000F, // nearest: db[1] (dist=0)
        0x0000_0000_0000_0007, // nearest: db[1] (dist=1)
    ];
    let database: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_000F,
        0xFFFF_FFFF_FFFF_FFFF,
    ];

    let nearest = matcher
        .find_nearest_neighbors(&ctx, &queries, &database, u32::MAX)
        .expect("最近邻计算失败");

    assert_eq!(nearest.len(), 3, "应有 3 个结果");

    // query[0]=0x0000 → nearest is db[0]=0x0000 (distance=0)
    assert_eq!(nearest[0].0, 0, "query[0] 最近邻索引应为 0");
    assert_eq!(nearest[0].1, 0, "query[0] 最近邻距离应为 0");

    // query[1]=0x000F → nearest is db[1]=0x000F (distance=0)
    assert_eq!(nearest[1].0, 1, "query[1] 最近邻索引应为 1");
    assert_eq!(nearest[1].1, 0, "query[1] 最近邻距离应为 0");

    // query[2]=0x0007 vs db[0]=0x0000: distance = 3 (bits 0,1,2 differ)
    // query[2]=0x0007 vs db[1]=0x000F: distance = 1 (only bit 3 differs)
    // query[2]=0x0007 vs db[2]=0xFFFF: distance = 61
    // nearest is db[1] with distance=1
    assert_eq!(nearest[2].0, 1, "query[2] 最近邻索引应为 1");
    assert_eq!(nearest[2].1, 1, "query[2] 最近邻距离应为 1");
}

#[test]
fn test_find_nearest_with_threshold() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcher::new(&mut ctx).expect("GpuHashMatcher 创建失败");

    // 2 queries, 2 database entries
    let queries: Vec<u64> = vec![
        0x0000_0000_0000_0000, // nearest: db[0] (dist=0) → within threshold
        0xFFFF_FFFF_FFFF_FFFF, // nearest: db[0] (dist=64) → exceeds threshold
    ];
    let database: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_000F,
    ];

    // threshold=10: query[0] matches (dist=0), query[1] exceeds threshold
    let nearest = matcher
        .find_nearest_neighbors(&ctx, &queries, &database, 10)
        .expect("带阈值的最近邻计算失败");

    assert_eq!(nearest.len(), 2);

    // query[0] nearest is db[0] with dist=0, within threshold
    assert_eq!(nearest[0].0, 0, "query[0] 最近邻索引应为 0");
    assert_eq!(nearest[0].1, 0, "query[0] 最近邻距离应为 0");

    // query[1] nearest distance is 64 > threshold=10, should return u32::MAX
    assert_eq!(nearest[1].0, u32::MAX, "query[1] 超出阈值，索引应为 u32::MAX");
    assert_eq!(nearest[1].1, u32::MAX, "query[1] 超出阈值，距离应为 u32::MAX");
}

#[test]
fn test_gpu_matches_cpu_hamming_distance() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuHashMatcher::new(&mut ctx).expect("GpuHashMatcher 创建失败");

    // Generate test data with known distances
    let queries: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0x1234_5678_9ABC_DEF0,
    ];
    let database: Vec<u64> = vec![
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
        0x0000_0000_0000_0001,
        0xFEDC_BA98_7654_3210,
    ];

    // Compute GPU distance matrix
    let gpu_matrix = matcher
        .compute_distance_matrix(&ctx, &queries, &database)
        .expect("GPU 距离矩阵计算失败");

    // Verify each entry matches CPU hamming_distance
    for (qi, &query) in queries.iter().enumerate() {
        for (di, &db_entry) in database.iter().enumerate() {
            let cpu_dist = gpgpu_tool::__hamming_distance(query, db_entry);
            assert_eq!(
                gpu_matrix[qi][di],
                cpu_dist,
                "GPU[{}][{}]={} != CPU({:#x}, {:#x})={}",
                qi,
                di,
                gpu_matrix[qi][di],
                query,
                db_entry,
                cpu_dist
            );
        }
    }
}

#[test]
fn test_facade_gpu_matcher() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");

    let hashes = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
    ];

    let facade = gpgpu_tool::HashMatcherFacade::gpu(&mut ctx, hashes).expect("GPU 门面创建失败");

    // Query for exact match
    let results = facade.find_similar(0x0000_0000_0000_0000, 1);
    assert!(
        results.iter().any(|r| r.hash == 0x0000_0000_0000_0000 && r.distance == 0),
        "应找到精确匹配"
    );
    assert!(
        results.iter().any(|r| r.hash == 0x0000_0000_0000_0001 && r.distance == 1),
        "应找到距离为 1 的匹配"
    );

    // Query with no match within threshold
    let results = facade.find_similar(0x1234_5678_9ABC_DEF0, 0);
    assert!(
        results.is_empty(),
        "不存在精确匹配时结果应为空"
    );

    // Query for all within large threshold
    let results = facade.find_similar(0x0000_0000_0000_0000, 64);
    assert!(
        results.len() >= 4,
        "大阈值应匹配所有条目，实际找到 {} 个",
        results.len()
    );
}

// ── Phase 2: 批量匹配测试 ────────────────────────────────────────

/// 测试 HashMatcher trait 的 find_similar_batch 默认方法
#[test]
fn test_trait_default_batch_method() {
    let hashes = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0003,
        0xFFFF_FFFF_FFFF_FFFF,
    ];

    // 使用 LinearScanMatcher 测试默认 batch 实现
    let matcher = LinearScanMatcher::new(hashes);
    let queries = vec![
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
    ];

    let batch_results = matcher.find_similar_batch(&queries, 1);

    // 应返回 2 个结果（每个 query 一个）
    assert_eq!(batch_results.len(), 2, "批量结果应有 2 个");

    // 第一个 query (0x0000) 应匹配 0x0000 和 0x0001
    assert!(
        batch_results[0].iter().any(|r| r.hash == 0x0000_0000_0000_0000 && r.distance == 0),
        "query[0] 应精确匹配 0x0000"
    );
    assert!(
        batch_results[0].iter().any(|r| r.hash == 0x0000_0000_0000_0001 && r.distance == 1),
        "query[0] 应匹配距离为 1 的 0x0001"
    );

    // 第二个 query (0xFFFF) 只应匹配 0xFFFF
    assert!(
        batch_results[1].iter().any(|r| r.hash == 0xFFFF_FFFF_FFFF_FFFF && r.distance == 0),
        "query[1] 应精确匹配 0xFFFF"
    );
}

/// 测试 GPU 批量匹配 vs CPU 逐一对比
#[test]
fn test_batch_gpu_vs_cpu() {
    let ctx = Arc::new(Mutex::new(GpuContext::new_sync().expect("GPU 初始化失败")));

    let database = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0003,
        0x0000_0000_0000_0007,
        0x0000_0000_0000_000F,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
    ];

    let queries = vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_FFFF_FFFF,
        0x1234_5678_9ABC_DEF0,
    ];

    let threshold = 3;

    // CPU 批量匹配（作为基准）
    let cpu_matcher = LinearScanMatcher::new(database.clone());
    let cpu_results = cpu_matcher.find_similar_batch(&queries, threshold);

    // GPU 批量匹配（使用共享 context）
    let gpu_facade = GpuHashMatcherFacade::new_with_shared_ctx(ctx, database)
        .expect("GPU 匹配器创建失败");
    let gpu_results = gpu_facade.find_similar_batch(&queries, threshold);

    // 结果数量应一致
    assert_eq!(
        cpu_results.len(),
        gpu_results.len(),
        "CPU 和 GPU 批量结果数量应一致"
    );

    // 每个 query 的匹配结果应一致
    for (qi, (cpu_matches, gpu_matches)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        // 排序后比较（顺序可能不同）
        let mut cpu_sorted: Vec<(u64, u32)> = cpu_matches.iter().map(|r| (r.hash, r.distance)).collect();
        let mut gpu_sorted: Vec<(u64, u32)> = gpu_matches.iter().map(|r| (r.hash, r.distance)).collect();
        cpu_sorted.sort();
        gpu_sorted.sort();

        assert_eq!(
            cpu_sorted,
            gpu_sorted,
            "query[{}] 的 CPU 和 GPU 匹配结果应一致",
            qi
        );
    }
}

/// 测试大规模批量匹配（100 queries × 1000 database）
#[test]
fn test_batch_large_scale() {
    let ctx = Arc::new(Mutex::new(GpuContext::new_sync().expect("GPU 初始化失败")));

    // 生成 1000 个 database 条目
    let database: Vec<u64> = (0..1000).map(|i| {
        // 生成分布均匀的哈希值
        let base = 0x0000_0000_0000_0000u64;
        base.wrapping_add(i * 0x0001_0000_0000_0000)
            .wrapping_add(i * 0x0000_0001_0000_0000)
            .wrapping_add(i * 0x0000_0000_0001_0000)
            .wrapping_add(i)
    }).collect();

    // 生成 100 个 queries
    let queries: Vec<u64> = (0..100).map(|i| {
        let base = 0x1234_5678_9ABC_DEF0u64;
        base.wrapping_add(i * 0x0000_0000_0000_0100)
    }).collect();

    let threshold = 20;

    // CPU 批量匹配
    let cpu_matcher = LinearScanMatcher::new(database.clone());
    let cpu_results = cpu_matcher.find_similar_batch(&queries, threshold);

    // GPU 批量匹配（使用共享 context）
    let gpu_facade = GpuHashMatcherFacade::new_with_shared_ctx(ctx, database)
        .expect("GPU 匹配器创建失败");
    let gpu_results = gpu_facade.find_similar_batch(&queries, threshold);

    // 验证结果一致性
    assert_eq!(cpu_results.len(), 100, "应有 100 个 query 的结果");
    assert_eq!(gpu_results.len(), 100, "GPU 应有 100 个 query 的结果");

    for (qi, (cpu_matches, gpu_matches)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
        let mut cpu_sorted: Vec<(u64, u32)> = cpu_matches.iter().map(|r| (r.hash, r.distance)).collect();
        let mut gpu_sorted: Vec<(u64, u32)> = gpu_matches.iter().map(|r| (r.hash, r.distance)).collect();
        cpu_sorted.sort();
        gpu_sorted.sort();

        assert_eq!(
            cpu_sorted,
            gpu_sorted,
            "大规模测试：query[{}] 的 CPU 和 GPU 结果不一致",
            qi
        );
    }
}

/// 端到端集成测试：模拟完整流水线（生成哈希→GPU批量匹配）
#[test]
fn test_end_to_end_hash_then_match() {
    let ctx = Arc::new(Mutex::new(GpuContext::new_sync().expect("GPU 初始化失败")));

    // 模拟图像哈希数据库（实际场景中这些来自图像处理）
    let image_hashes: Vec<u64> = vec![
        // 一组相似的哈希（距离 ≤ 5）
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x0000_0000_0000_0003,
        0x0000_0000_0000_0007,
        0x0000_0000_0000_000F,
        // 一组完全不同的哈希
        0xFFFF_FFFF_FFFF_FFFF,
        0xFFFF_FFFF_FFFF_FFFE,
        0xFFFF_FFFF_FFFF_FFFC,
        // 另一组
        0xAAAA_AAAA_AAAA_AAAA,
        0xAAAA_AAAA_AAAA_AAAB,
    ];

    // 模拟查询：新图像的哈希
    let query_hashes: Vec<u64> = vec![
        0x0000_0000_0000_0002,  // 应匹配第一组
        0xFFFF_FFFF_FFFF_FFFD,  // 应匹配第二组
        0xAAAA_AAAA_AAAA_AAAB,  // 应匹配第三组（精确匹配）
    ];

    let threshold = 3;

    // 创建 GPU 匹配器（使用共享 context）
    let facade = GpuHashMatcherFacade::new_with_shared_ctx(ctx, image_hashes)
        .expect("GPU 匹配器创建失败");

    // 批量查询
    let results = facade.find_similar_batch(&query_hashes, threshold);

    assert_eq!(results.len(), 3, "应有 3 个查询的结果");

    // 验证第一个查询 (0x0002) 匹配第一组
    let group1_matches: Vec<u64> = results[0].iter().map(|r| r.hash).collect();
    assert!(
        group1_matches.contains(&0x0000_0000_0000_0000),
        "query[0] 应匹配 0x0000"
    );
    assert!(
        group1_matches.contains(&0x0000_0000_0000_0001),
        "query[0] 应匹配 0x0001"
    );
    assert!(
        group1_matches.contains(&0x0000_0000_0000_0003),
        "query[0] 应匹配 0x0003"
    );

    // 验证第二个查询 (0xFFFD) 匹配第二组
    let group2_matches: Vec<u64> = results[1].iter().map(|r| r.hash).collect();
    assert!(
        group2_matches.contains(&0xFFFF_FFFF_FFFF_FFFF),
        "query[1] 应匹配 0xFFFF"
    );
    assert!(
        group2_matches.contains(&0xFFFF_FFFF_FFFF_FFFE),
        "query[1] 应匹配 0xFFFE"
    );
    assert!(
        group2_matches.contains(&0xFFFF_FFFF_FFFF_FFFC),
        "query[1] 应匹配 0xFFFC"
    );

    // 验证第三个查询 (0xAAAB) 匹配第三组
    let group3_matches: Vec<u64> = results[2].iter().map(|r| r.hash).collect();
    assert!(
        group3_matches.contains(&0xAAAA_AAAA_AAAA_AAAA),
        "query[2] 应匹配 0xAAAA"
    );
    assert!(
        group3_matches.contains(&0xAAAA_AAAA_AAAA_AAAB),
        "query[2] 应精确匹配 0xAAAB"
    );

    // 验证距离正确性
    for (qi, matches) in results.iter().enumerate() {
        for m in matches {
            let expected_dist = gpgpu_tool::__hamming_distance(query_hashes[qi], m.hash);
            assert_eq!(
                m.distance,
                expected_dist,
                "query[{}]={:#x} vs hash={:#x} 距离应为 {}",
                qi,
                query_hashes[qi],
                m.hash,
                expected_dist
            );
        }
    }
}
