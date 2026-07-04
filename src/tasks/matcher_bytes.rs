//! # 变长哈希匹配策略
//!
//! 提供统一的 [`HashMatcherBytes`] trait 和多种匹配策略实现，
//! 支持 64-bit 到 4096-bit 的变长哈希。
//!
//! - [`LinearScanMatcherBytes`]：精确线性扫描，100% 召回率
//! - [`BkTreeMatcherBytes`]：BK-tree 近似匹配，O(log N) 搜索
//! - [`ChainedMatcherBytes`]：责任链组合多个匹配器
//! - [`HashMatcherFacadeBytes`]：统一门面，封装策略选择和二面体变换增强

use std::collections::HashSet;

use crate::tasks::bktree_bytes::BkTreeBytes;
use crate::tasks::dihedral::{
    DihedralHashes1024, DihedralHashes256, DihedralHashes4096, DihedralHashes64, DihedralTransform,
};
use crate::tasks::hash_bytes::HashBytes;

/// 变长哈希匹配结果。
#[derive(Debug, Clone)]
pub struct MatchResultBytes {
    pub hash: HashBytes,
    pub distance: u32,
    /// Hash quality score (0.0-1.0), if available.
    pub quality: Option<f32>,
}

impl PartialEq for MatchResultBytes {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.distance == other.distance
    }
}

impl Eq for MatchResultBytes {}

/// 变长哈希匹配策略抽象接口（策略模式）。
pub trait HashMatcherBytes {
    /// 查找与 `query` 汉明距离 ≤ `threshold` 的所有哈希。
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes>;

    /// 批量查询：一次查询多个 query。
    ///
    /// 默认实现逐个调用 `find_similar`，GPU 实现应覆盖此方法以获得并行加速。
    fn find_similar_batch(&self, queries: &[HashBytes], threshold: u32) -> Vec<Vec<MatchResultBytes>> {
        queries
            .iter()
            .map(|q| self.find_similar(q, threshold))
            .collect()
    }

    /// 使用归一化阈值比例查找相似哈希。
    ///
    /// `ratio` 取值范围 0.0-1.0，内部计算
    /// `threshold = (ratio * query.byte_len() as f32 * 8.0) as u32`。
    ///
    /// # 推荐比例范围
    ///
    /// - 64-bit 哈希（8 字节）：`0.0`-`0.25`
    /// - 256-bit 哈希（32 字节）：`0.0`-`0.15`
    /// - 1024-bit 哈希（128 字节）：`0.0`-`0.10`
    /// - 4096-bit 哈希（512 字节）：`0.0`-`0.05`
    /// - 推荐 `0.05`-`0.10` 用于相似图像检测
    fn find_similar_ratio(&self, query: &HashBytes, ratio: f32) -> Vec<MatchResultBytes> {
        let total_bits = query.byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar(query, threshold)
    }

    /// 使用归一化阈值比例批量查找相似哈希。
    ///
    /// 参见 [`find_similar_ratio`](HashMatcherBytes::find_similar_ratio)。
    fn find_similar_batch_ratio(
        &self,
        queries: &[HashBytes],
        ratio: f32,
    ) -> Vec<Vec<MatchResultBytes>> {
        if queries.is_empty() {
            return vec![];
        }
        let total_bits = queries[0].byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar_batch(queries, threshold)
    }
}

/// 责任链策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainStrategyBytes {
    /// 合并所有匹配器结果（并集）。
    Union,
    /// 第一个有结果的匹配器后停止。
    FirstHit,
}

// ── LinearScanMatcherBytes ────────────────────────────────────────

/// 精确线性扫描匹配器，100% 召回率保证。
pub struct LinearScanMatcherBytes {
    hashes: Vec<HashBytes>,
}

impl LinearScanMatcherBytes {
    pub fn new(hashes: Vec<HashBytes>) -> Self {
        Self { hashes }
    }
}

impl HashMatcherBytes for LinearScanMatcherBytes {
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        self.hashes
            .iter()
            .map(|h| MatchResultBytes {
                hash: h.clone(),
                distance: query.hamming_distance(h),
                quality: None,
            })
            .filter(|r| r.distance <= threshold)
            .collect()
    }
}

// ── BkTreeMatcherBytes ────────────────────────────────────────────

/// BK-tree 适配器，将 [`BkTreeBytes`] 适配为 [`HashMatcherBytes`] trait 实现。
pub struct BkTreeMatcherBytes {
    tree: BkTreeBytes,
}

impl BkTreeMatcherBytes {
    pub fn new(hashes: Vec<HashBytes>) -> Self {
        Self {
            tree: BkTreeBytes::from_hashes(hashes),
        }
    }
}

impl HashMatcherBytes for BkTreeMatcherBytes {
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        self.tree
            .find(query, threshold)
            .into_iter()
            .map(|(hash, distance)| MatchResultBytes { hash, distance, quality: None })
            .collect()
    }
}

// ── ChainedMatcherBytes ───────────────────────────────────────────

/// 责任链模式组合多个变长哈希匹配器。
pub struct ChainedMatcherBytes {
    matchers: Vec<Box<dyn HashMatcherBytes>>,
    strategy: ChainStrategyBytes,
}

impl ChainedMatcherBytes {
    pub fn new(matchers: Vec<Box<dyn HashMatcherBytes>>, strategy: ChainStrategyBytes) -> Self {
        Self { matchers, strategy }
    }

    /// 创建 BK-tree + 线性扫描的责任链（FirstHit 策略）。
    pub fn bk_tree_plus_linear(hashes: Vec<HashBytes>) -> Self {
        let matchers: Vec<Box<dyn HashMatcherBytes>> = vec![
            Box::new(BkTreeMatcherBytes::new(hashes.clone())),
            Box::new(LinearScanMatcherBytes::new(hashes)),
        ];
        Self::new(matchers, ChainStrategyBytes::FirstHit)
    }
}

impl HashMatcherBytes for ChainedMatcherBytes {
    fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        match self.strategy {
            ChainStrategyBytes::FirstHit => {
                for matcher in &self.matchers {
                    let results = matcher.find_similar(query, threshold);
                    if !results.is_empty() {
                        return results;
                    }
                }
                vec![]
            }
            ChainStrategyBytes::Union => {
                let mut seen = HashSet::new();
                let mut all_results = Vec::new();
                for matcher in &self.matchers {
                    let results = matcher.find_similar(query, threshold);
                    for r in results {
                        if seen.insert(r.hash.clone()) {
                            all_results.push(r);
                        }
                    }
                }
                all_results
            }
        }
    }
}

// ── HashMatcherFacadeBytes ────────────────────────────────────────

/// 统一门面（Facade 模式），封装策略选择和二面体变换增强。
///
/// 二面体变换目前支持 64-bit（8 字节）、256-bit（32 字节）、
/// 1024-bit（128 字节）和 4096-bit（512 字节）哈希，
/// 其他尺寸的哈希在启用二面体变换时将跳过变换。
pub struct HashMatcherFacadeBytes {
    matcher: Box<dyn HashMatcherBytes>,
    dihedral_enabled: bool,
}

impl HashMatcherFacadeBytes {
    /// 创建精确线性扫描门面（100% 召回率）。
    pub fn linear_scan(hashes: Vec<HashBytes>) -> Self {
        Self {
            matcher: Box::new(LinearScanMatcherBytes::new(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 创建 BK-tree 门面（近似快速匹配）。
    pub fn bktree(hashes: Vec<HashBytes>) -> Self {
        Self {
            matcher: Box::new(BkTreeMatcherBytes::new(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 创建责任链门面（BK-tree 快速匹配 + 线性扫描补充）。
    pub fn chained(hashes: Vec<HashBytes>) -> Self {
        Self {
            matcher: Box::new(ChainedMatcherBytes::bk_tree_plus_linear(hashes)),
            dihedral_enabled: false,
        }
    }

    /// 启用二面体变换匹配。
    ///
    /// 对查询哈希的 8 种 D4 群变体分别匹配，返回去重后的结果。
    /// 支持 64-bit（8 字节）、256-bit（32 字节）、1024-bit（128 字节）
    /// 和 4096-bit（512 字节）哈希，
    /// 其他尺寸的查询将直接使用原始哈希匹配。
    pub fn with_dihedral(mut self) -> Self {
        self.dihedral_enabled = true;
        self
    }

    /// 查找相似哈希。
    pub fn find_similar(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        if self.dihedral_enabled {
            self.find_similar_dihedral(query, threshold)
        } else {
            self.matcher.find_similar(query, threshold)
        }
    }

    /// 批量查询：一次查询多个 query。
    pub fn find_similar_batch(
        &self,
        queries: &[HashBytes],
        threshold: u32,
    ) -> Vec<Vec<MatchResultBytes>> {
        if self.dihedral_enabled {
            queries
                .iter()
                .map(|q| self.find_similar_dihedral(q, threshold))
                .collect()
        } else {
            self.matcher.find_similar_batch(queries, threshold)
        }
    }

    /// 使用归一化阈值比例查找相似哈希。
    ///
    /// `ratio` 取值范围 0.0-1.0，内部计算
    /// `threshold = (ratio * query.byte_len() as f32 * 8.0) as u32`。
    ///
    /// # 推荐比例范围
    ///
    /// - 64-bit 哈希（8 字节）：`0.0`-`0.25`
    /// - 256-bit 哈希（32 字节）：`0.0`-`0.15`
    /// - 1024-bit 哈希（128 字节）：`0.0`-`0.10`
    /// - 4096-bit 哈希（512 字节）：`0.0`-`0.05`
    /// - 推荐 `0.05`-`0.10` 用于相似图像检测
    pub fn find_similar_ratio(&self, query: &HashBytes, ratio: f32) -> Vec<MatchResultBytes> {
        let total_bits = query.byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar(query, threshold)
    }

    /// 使用归一化阈值比例批量查找相似哈希。
    ///
    /// 参见 [`find_similar_ratio`](HashMatcherFacadeBytes::find_similar_ratio)。
    pub fn find_similar_batch_ratio(
        &self,
        queries: &[HashBytes],
        ratio: f32,
    ) -> Vec<Vec<MatchResultBytes>> {
        if queries.is_empty() {
            return vec![];
        }
        let total_bits = queries[0].byte_len() as f32 * 8.0;
        let threshold = (ratio * total_bits) as u32;
        self.find_similar_batch(queries, threshold)
    }

    /// 二面体变换匹配：对查询哈希的 8 种变体分别匹配，返回去重后的结果。
    fn find_similar_dihedral(&self, query: &HashBytes, threshold: u32) -> Vec<MatchResultBytes> {
        let variants = self.dihedral_variants(query);
        let mut seen = HashSet::new();
        let mut best_results = Vec::new();

        for variant in &variants {
            let results = self.matcher.find_similar(variant, threshold);
            for r in results {
                if seen.insert(r.hash.clone()) {
                    best_results.push(r);
                }
            }
        }

        best_results
    }

    /// 根据哈希字节长度生成二面体变换变体。
    ///
    /// - 8 字节（64-bit）：使用 `DihedralHashes64`
    /// - 32 字节（256-bit）：使用 `DihedralHashes256`
    /// - 128 字节（1024-bit）：使用 `DihedralHashes1024`
    /// - 512 字节（4096-bit）：使用 `DihedralHashes4096`
    /// - 其他尺寸：仅返回原始哈希（不支持二面体变换）
    fn dihedral_variants(&self, hash: &HashBytes) -> Vec<HashBytes> {
        match hash.byte_len() {
            8 => {
                let u64s = hash.to_u64s();
                let d = DihedralHashes64::from_hash(&u64s[0]);
                d.all_variants()
                    .into_iter()
                    .map(HashBytes::from_u64)
                    .collect()
            }
            32 => {
                let u64s = hash.to_u64s();
                if u64s.len() == 4 {
                    let arr: [u64; 4] = [u64s[0], u64s[1], u64s[2], u64s[3]];
                    let d = DihedralHashes256::from_hash(&arr);
                    d.all_variants()
                        .into_iter()
                        .map(|v| HashBytes::from_u64s(&v))
                        .collect()
                } else {
                    vec![hash.clone()]
                }
            }
            128 => {
                let u64s = hash.to_u64s();
                if u64s.len() == 16 {
                    let arr: [u64; 16] = u64s.try_into().expect("长度已验证为 16");
                    let d = DihedralHashes1024::from_hash(&arr);
                    d.all_variants()
                        .into_iter()
                        .map(|v| HashBytes::from_u64s(&v))
                        .collect()
                } else {
                    vec![hash.clone()]
                }
            }
            512 => {
                let u64s = hash.to_u64s();
                if u64s.len() == 64 {
                    let arr: [u64; 64] = u64s.try_into().expect("长度已验证为 64");
                    let d = DihedralHashes4096::from_hash(&arr);
                    d.all_variants()
                        .into_iter()
                        .map(|v| HashBytes::from_u64s(&v))
                        .collect()
                } else {
                    vec![hash.clone()]
                }
            }
            _ => {
                vec![hash.clone()]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: u64) -> HashBytes {
        HashBytes::from_u64(v)
    }

    fn sample_hashes() -> Vec<HashBytes> {
        vec![h(0x0000_0000_0000_0001), h(0x0000_0000_0000_0003), h(0x0000_0000_0000_0007), h(0xFFFF_FFFF_FFFF_FFFF)]
    }

    #[test]
    fn test_linear_scan_find_similar() {
        let matcher = LinearScanMatcherBytes::new(sample_hashes());
        let results = matcher.find_similar(&h(0x0000_0000_0000_0001), 1);
        assert!(results
            .iter()
            .any(|r| r.hash == h(0x0000_0000_0000_0001) && r.distance == 0));
        assert!(results
            .iter()
            .any(|r| r.hash == h(0x0000_0000_0000_0003) && r.distance == 1));
    }

    #[test]
    fn test_linear_scan_batch() {
        let matcher = LinearScanMatcherBytes::new(sample_hashes());
        let results = matcher.find_similar_batch(
            &[h(0x0000_0000_0000_0001), h(0x0000_0000_0000_0003)],
            1,
        );
        assert_eq!(results.len(), 2);
        assert!(results[0].iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_bktree_matcher_find_similar() {
        let matcher = BkTreeMatcherBytes::new(sample_hashes());
        let results = matcher.find_similar(&h(0x0000_0000_0000_0001), 1);
        assert!(results
            .iter()
            .any(|r| r.hash == h(0x0000_0000_0000_0001) && r.distance == 0));
        assert!(results
            .iter()
            .any(|r| r.hash == h(0x0000_0000_0000_0003) && r.distance == 1));
    }

    #[test]
    fn test_chained_matcher_first_hit() {
        let matcher = ChainedMatcherBytes::bk_tree_plus_linear(sample_hashes());
        let results = matcher.find_similar(&h(0x0000_0000_0000_0001), 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_facade_linear_scan() {
        let facade = HashMatcherFacadeBytes::linear_scan(sample_hashes());
        let results = facade.find_similar(&h(0x0000_0000_0000_0001), 1);
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_facade_bktree() {
        let facade = HashMatcherFacadeBytes::bktree(sample_hashes());
        let results = facade.find_similar(&h(0x0000_0000_0000_0001), 1);
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_facade_with_dihedral() {
        let facade = HashMatcherFacadeBytes::linear_scan(sample_hashes()).with_dihedral();
        let results = facade.find_similar(&h(0x0000_0000_0000_0001), 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_facade_dihedral_256bit() {
        // 256-bit 哈希的二面体变换
        let hash_256 = HashBytes::from_u64s(&[0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]);
        let hashes = vec![hash_256.clone()];
        let facade = HashMatcherFacadeBytes::linear_scan(hashes).with_dihedral();
        let results = facade.find_similar(&hash_256, 0);
        // 精确匹配自身
        assert!(results.iter().any(|r| r.distance == 0));
    }

    #[test]
    fn test_match_result_bytes_equality() {
        let a = MatchResultBytes {
            hash: h(42),
            distance: 1,
            quality: None,
        };
        let b = MatchResultBytes {
            hash: h(42),
            distance: 1,
            quality: None,
        };
        let c = MatchResultBytes {
            hash: h(42),
            distance: 2,
            quality: None,
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_match_result_bytes_equality_ignores_quality() {
        let a = MatchResultBytes {
            hash: h(42),
            distance: 1,
            quality: None,
        };
        let b = MatchResultBytes {
            hash: h(42),
            distance: 1,
            quality: Some(0.5),
        };
        assert_eq!(a, b, "quality should be ignored in PartialEq");
    }
}
