//! # 哈希匹配策略
//!
//! 提供统一的 [`HashMatcher`] trait 和多种匹配策略实现：
//!
//! - [`LinearScanMatcher`]：精确线性扫描，100% 召回率
//! - [`BkTreeMatcher`]：BK-tree 近似匹配，O(log N) 搜索
//! - [`ChainedMatcher`]：责任链组合多个匹配器
//! - [`HashMatcherFacade`]：统一门面，封装策略选择和二面体变换增强

use std::collections::HashSet;

use crate::tasks::bktree::{hamming_distance, BkTree};
use crate::tasks::dihedral::DihedralHashes64;

/// 匹配结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    pub hash: u64,
    pub distance: u32,
}

/// 匹配策略抽象接口（策略模式）。
pub trait HashMatcher {
    /// 查找与 `query` 汉明距离 ≤ `threshold` 的所有哈希。
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult>;
}

/// 责任链策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStrategy {
    /// 合并所有匹配器结果（并集）。
    Union,
    /// 第一个有结果的匹配器后停止。
    FirstHit,
}

// ── LinearScanMatcher ────────────────────────────────────────────

/// 精确线性扫描匹配器，100% 召回率保证。
pub struct LinearScanMatcher {
    hashes: Vec<u64>,
}

impl LinearScanMatcher {
    pub fn new(hashes: Vec<u64>) -> Self {
        Self { hashes }
    }

    /// 批量查询：一次查询多个 query，返回每个 query 的匹配结果。
    pub fn find_similar_batch(&self, queries: &[u64], threshold: u32) -> Vec<Vec<MatchResult>> {
        queries
            .iter()
            .map(|&q| self.find_similar(q, threshold))
            .collect()
    }
}

impl HashMatcher for LinearScanMatcher {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        self.hashes
            .iter()
            .map(|&h| MatchResult {
                hash: h,
                distance: hamming_distance(query, h),
            })
            .filter(|r| r.distance <= threshold)
            .collect()
    }
}

// ── BkTreeMatcher ────────────────────────────────────────────────

/// BK-tree 适配器，将 [`BkTree`] 适配为 [`HashMatcher`] trait 实现。
pub struct BkTreeMatcher {
    tree: BkTree,
}

impl BkTreeMatcher {
    pub fn new(hashes: Vec<u64>) -> Self {
        Self {
            tree: BkTree::from_hashes(hashes),
        }
    }
}

impl HashMatcher for BkTreeMatcher {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        self.tree
            .find(query, threshold)
            .into_iter()
            .map(|(hash, distance)| MatchResult { hash, distance })
            .collect()
    }
}

// ── ChainedMatcher ───────────────────────────────────────────────

/// 责任链模式组合多个匹配器。
pub struct ChainedMatcher {
    matchers: Vec<Box<dyn HashMatcher>>,
    strategy: ChainStrategy,
}

impl ChainedMatcher {
    pub fn new(matchers: Vec<Box<dyn HashMatcher>>, strategy: ChainStrategy) -> Self {
        Self { matchers, strategy }
    }

    /// 创建 BK-tree + 线性扫描的责任链（Union 策略）。
    pub fn bk_tree_plus_linear(hashes: Vec<u64>) -> Self {
        let matchers: Vec<Box<dyn HashMatcher>> = vec![
            Box::new(BkTreeMatcher::new(hashes.clone())),
            Box::new(LinearScanMatcher::new(hashes)),
        ];
        Self::new(matchers, ChainStrategy::Union)
    }
}

impl HashMatcher for ChainedMatcher {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        match self.strategy {
            ChainStrategy::FirstHit => {
                for matcher in &self.matchers {
                    let results = matcher.find_similar(query, threshold);
                    if !results.is_empty() {
                        return results;
                    }
                }
                vec![]
            }
            ChainStrategy::Union => {
                let mut seen = HashSet::new();
                let mut all_results = Vec::new();
                for matcher in &self.matchers {
                    let results = matcher.find_similar(query, threshold);
                    for r in results {
                        if seen.insert(r.hash) {
                            all_results.push(r);
                        }
                    }
                }
                all_results
            }
        }
    }
}

// ── HashMatcherFacade ────────────────────────────────────────────

/// 统一门面（Facade 模式），封装策略选择和二面体变换增强。
pub struct HashMatcherFacade {
    matcher: Box<dyn HashMatcher>,
    dihedral_enabled: bool,
}

impl HashMatcherFacade {
    /// 创建精确线性扫描门面（100% 召回率）。
    pub fn linear_scan(hashes: Vec<u64>) -> Self {
        Self {
            matcher: Box::new(LinearScanMatcher::new(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 创建 BK-tree 门面（近似快速匹配）。
    pub fn bktree(hashes: Vec<u64>) -> Self {
        Self {
            matcher: Box::new(BkTreeMatcher::new(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 创建责任链门面（BK-tree 快速匹配 + 线性扫描补充）。
    pub fn chained(hashes: Vec<u64>) -> Self {
        Self {
            matcher: Box::new(ChainedMatcher::bk_tree_plus_linear(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 启用二面体变换匹配。
    pub fn with_dihedral(mut self) -> Self {
        self.dihedral_enabled = true;
        self
    }

    /// 查找相似哈希。
    pub fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        if self.dihedral_enabled {
            self.find_similar_dihedral(query, threshold)
        } else {
            self.matcher.find_similar(query, threshold)
        }
    }

    /// 二面体变换匹配：对查询哈希的 8 种变体分别匹配，返回去重后的结果。
    fn find_similar_dihedral(&self, query: u64, threshold: u32) -> Vec<MatchResult> {
        let variants = DihedralHashes64::from_u64(query).all();
        let mut seen = HashSet::new();
        let mut best_results = Vec::new();

        for &variant in &variants {
            let results = self.matcher.find_similar(variant, threshold);
            for r in results {
                if seen.insert(r.hash) {
                    best_results.push(r);
                }
            }
        }

        best_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hashes() -> Vec<u64> {
        vec![0x0000_0000_0000_0001, 0x0000_0000_0000_0003, 0x0000_0000_0000_0007, 0xFFFF_FFFF_FFFF_FFFF]
    }

    #[test]
    fn test_linear_scan_find_similar() {
        let matcher = LinearScanMatcher::new(sample_hashes());
        let results = matcher.find_similar(0x0000_0000_0000_0001, 1);
        assert!(results.iter().any(|r| r.hash == 0x0000_0000_0000_0001 && r.distance == 0));
        assert!(results.iter().any(|r| r.hash == 0x0000_0000_0000_0003 && r.distance == 1));
    }

    #[test]
    fn test_linear_scan_batch() {
        let matcher = LinearScanMatcher::new(sample_hashes());
        let results = matcher.find_similar_batch(&[0x0000_0000_0000_0001, 0x0000_0000_0000_0003], 1);
        assert_eq!(results.len(), 2);
        assert!(results[0].iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_bktree_matcher_find_similar() {
        let matcher = BkTreeMatcher::new(sample_hashes());
        let results = matcher.find_similar(0x0000_0000_0000_0001, 1);
        assert!(results.iter().any(|r| r.hash == 0x0000_0000_0000_0001 && r.distance == 0));
        assert!(results.iter().any(|r| r.hash == 0x0000_0000_0000_0003 && r.distance == 1));
    }

    #[test]
    fn test_chained_matcher_union() {
        let matcher = ChainedMatcher::bk_tree_plus_linear(sample_hashes());
        let results = matcher.find_similar(0x0000_0000_0000_0001, 1);
        let hashes: Vec<u64> = results.iter().map(|r| r.hash).collect();
        // Union 去重：每个 hash 只出现一次
        let unique: HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(hashes.len(), unique.len());
    }

    #[test]
    fn test_chained_matcher_first_hit() {
        let matchers: Vec<Box<dyn HashMatcher>> = vec![
            Box::new(BkTreeMatcher::new(sample_hashes())),
            Box::new(LinearScanMatcher::new(sample_hashes())),
        ];
        let matcher = ChainedMatcher::new(matchers, ChainStrategy::FirstHit);
        let results = matcher.find_similar(0x0000_0000_0000_0001, 1);
        // FirstHit：BK-tree 有结果即返回，不会继续线性扫描
        assert!(!results.is_empty());
    }

    #[test]
    fn test_chained_matcher_first_hit_empty() {
        let matchers: Vec<Box<dyn HashMatcher>> = vec![
            Box::new(BkTreeMatcher::new(vec![])),
            Box::new(LinearScanMatcher::new(sample_hashes())),
        ];
        let matcher = ChainedMatcher::new(matchers, ChainStrategy::FirstHit);
        let results = matcher.find_similar(0x0000_0000_0000_0001, 1);
        // 第一个匹配器为空，回退到第二个
        assert!(!results.is_empty());
    }

    #[test]
    fn test_facade_linear_scan() {
        let facade = HashMatcherFacade::linear_scan(sample_hashes());
        let results = facade.find_similar(0x0000_0000_0000_0001, 1);
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_facade_bktree() {
        let facade = HashMatcherFacade::bktree(sample_hashes());
        let results = facade.find_similar(0x0000_0000_0000_0001, 1);
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_facade_chained() {
        let facade = HashMatcherFacade::chained(sample_hashes());
        let results = facade.find_similar(0x0000_0000_0000_0001, 1);
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_facade_with_dihedral() {
        let facade = HashMatcherFacade::linear_scan(sample_hashes()).with_dihedral();
        let results = facade.find_similar(0x0000_0000_0000_0001, 2);
        // 二面体变换：8 种变体分别匹配，结果去重
        assert!(!results.is_empty());
    }

    #[test]
    fn test_match_result_equality() {
        let a = MatchResult { hash: 42, distance: 1 };
        let b = MatchResult { hash: 42, distance: 1 };
        let c = MatchResult { hash: 42, distance: 2 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_chain_strategy_values() {
        assert_eq!(ChainStrategy::Union, ChainStrategy::Union);
        assert_ne!(ChainStrategy::Union, ChainStrategy::FirstHit);
    }
}
