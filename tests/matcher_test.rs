use std::collections::HashSet;

use gpgpu_tool::{
    BkTreeMatcher, ChainStrategy, ChainedMatcher, HashMatcher, HashMatcherFacade,
    LinearScanMatcher, MatchResult,
};

fn test_hashes() -> Vec<u64> {
    vec![
        0x0000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_FFFF_FFFF,
        0xAAAA_AAAA_AAAA_AAAA,
    ]
}

#[test]
fn test_linear_scan_100_percent_recall() {
    let matcher = LinearScanMatcher::new(test_hashes());
    let query = 0x0000_0000_0000_0000;
    let results = matcher.find_similar(query, 1);

    // 应找到自身（距离 0）和 0x0000000000000001（距离 1）
    assert!(results
        .iter()
        .any(|r| r.hash == 0x0000_0000_0000_0000 && r.distance == 0));
    assert!(results
        .iter()
        .any(|r| r.hash == 0x0000_0000_0000_0001 && r.distance == 1));
}

#[test]
fn test_linear_scan_no_match() {
    let matcher = LinearScanMatcher::new(test_hashes());
    // 阈值为 0，只有精确匹配
    let results = matcher.find_similar(0x1234_5678_9ABC_DEF0, 0);
    assert!(
        results.is_empty(),
        "不存在精确匹配时结果应为空"
    );
}

#[test]
fn test_linear_scan_exact_match() {
    let matcher = LinearScanMatcher::new(test_hashes());
    let results = matcher.find_similar(0xFFFF_FFFF_FFFF_FFFF, 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].hash, 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(results[0].distance, 0);
}

#[test]
fn test_bktree_matcher_consistency() {
    let hashes = test_hashes();
    let bktree = BkTreeMatcher::new(hashes.clone());
    let linear = LinearScanMatcher::new(hashes);

    let query = 0x0000_0000_0000_0000;
    let bktree_results = bktree.find_similar(query, 10);
    let linear_results = linear.find_similar(query, 10);

    // BK-tree 结果应是线性扫描结果的子集
    let linear_hashes: HashSet<u64> = linear_results.iter().map(|r| r.hash).collect();
    for r in &bktree_results {
        assert!(
            linear_hashes.contains(&r.hash),
            "BK-tree 结果应在线性扫描结果中"
        );
    }
}

#[test]
fn test_bktree_matcher_exact_match() {
    let matcher = BkTreeMatcher::new(test_hashes());
    let results = matcher.find_similar(0xAAAA_AAAA_AAAA_AAAA, 0);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].distance, 0);
}

#[test]
fn test_chained_union() {
    let hashes = test_hashes();
    let chained = ChainedMatcher::new(
        vec![
            Box::new(BkTreeMatcher::new(hashes.clone())),
            Box::new(LinearScanMatcher::new(hashes)),
        ],
        ChainStrategy::Union,
    );

    let results = chained.find_similar(0x0000_0000_0000_0000, 10);
    // Union 策略应包含所有匹配结果
    assert!(!results.is_empty());

    // Union 去重：每个 hash 只出现一次
    let hash_list: Vec<u64> = results.iter().map(|r| r.hash).collect();
    let unique: HashSet<u64> = hash_list.iter().copied().collect();
    assert_eq!(hash_list.len(), unique.len(), "Union 结果应无重复");
}

#[test]
fn test_chained_first_hit() {
    let hashes = test_hashes();
    let chained = ChainedMatcher::new(
        vec![
            Box::new(BkTreeMatcher::new(hashes.clone())),
            Box::new(LinearScanMatcher::new(hashes)),
        ],
        ChainStrategy::FirstHit,
    );

    let results = chained.find_similar(0x0000_0000_0000_0000, 10);
    // FirstHit 策略：BK-tree 有结果就返回
    assert!(!results.is_empty());
}

#[test]
fn test_chained_first_hit_fallback() {
    // 第一个匹配器为空，回退到第二个
    let hashes = test_hashes();
    let chained = ChainedMatcher::new(
        vec![
            Box::new(BkTreeMatcher::new(vec![])),
            Box::new(LinearScanMatcher::new(hashes)),
        ],
        ChainStrategy::FirstHit,
    );

    let results = chained.find_similar(0x0000_0000_0000_0000, 1);
    assert!(!results.is_empty(), "空 BK-tree 应回退到线性扫描");
}

#[test]
fn test_facade_linear_scan() {
    let facade = HashMatcherFacade::linear_scan(test_hashes());
    let results = facade.find_similar(0x0000_0000_0000_0000, 1);
    assert!(results.len() >= 2, "线性扫描应找到至少 2 个匹配");
}

#[test]
fn test_facade_bktree() {
    let facade = HashMatcherFacade::bktree(test_hashes());
    let results = facade.find_similar(0x0000_0000_0000_0000, 10);
    assert!(!results.is_empty());
}

#[test]
fn test_facade_chained() {
    let facade = HashMatcherFacade::chained(test_hashes());
    let results = facade.find_similar(0x0000_0000_0000_0000, 10);
    assert!(!results.is_empty());
}

#[test]
fn test_facade_chained_with_dihedral() {
    let facade = HashMatcherFacade::chained(test_hashes()).with_dihedral();
    let results = facade.find_similar(0x0000_0000_0000_0000, 10);
    assert!(!results.is_empty());
}

#[test]
fn test_facade_dihedral_expands_results() {
    // 二面体变换应扩大匹配范围
    let hashes = vec![0x0F0F_0F0F_0F0F_0F0F];
    let plain = HashMatcherFacade::linear_scan(hashes.clone());
    let dihedral = HashMatcherFacade::linear_scan(hashes).with_dihedral();

    let query = 0x0F0F_0F0F_0F0F_0F0F;
    let plain_results = plain.find_similar(query, 0);
    let dihedral_results = dihedral.find_similar(query, 0);

    // 二面体变换对 8 种变体分别匹配，结果应不少于普通匹配
    assert!(
        dihedral_results.len() >= plain_results.len(),
        "二面体变换匹配结果应不少于普通匹配"
    );
}

#[test]
fn test_match_result_equality() {
    let a = MatchResult {
        hash: 42,
        distance: 1,
        quality: None,
    };
    let b = MatchResult {
        hash: 42,
        distance: 1,
        quality: None,
    };
    let c = MatchResult {
        hash: 42,
        distance: 2,
        quality: None,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_linear_scan_batch() {
    let matcher = LinearScanMatcher::new(test_hashes());
    let queries = [0x0000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF];
    let results = matcher.find_similar_batch(&queries, 1);
    assert_eq!(results.len(), 2);
    assert!(results[0].iter().any(|r| r.distance == 0));
    assert!(results[1].iter().any(|r| r.distance == 0));
}
