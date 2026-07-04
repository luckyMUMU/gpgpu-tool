//! 变长哈希 BK-tree，支持 64-bit 到 4096-bit 的感知哈希近似最近邻搜索。
//!
//! 与 [`super::bktree::BkTree`]（硬编码 u64）不同，本模块使用 [`HashBytes`]
//! 作为哈希类型，支持任意长度哈希的插入和搜索。

use super::hash_bytes::HashBytes;

/// 变长哈希 BK-tree 节点。
struct BkNodeBytes {
    hash: HashBytes,
    children: Vec<(u32, Box<BkNodeBytes>)>,
}

impl BkNodeBytes {
    fn insert(&mut self, hash: HashBytes) {
        let d = self.hash.hamming_distance(&hash);
        match self
            .children
            .binary_search_by_key(&d, |(dist, _)| *dist)
        {
            Ok(idx) => self.children[idx].1.insert(hash),
            Err(idx) => self.children.insert(
                idx,
                (
                    d,
                    Box::new(BkNodeBytes {
                        hash,
                        children: Vec::new(),
                    }),
                ),
            ),
        }
    }

    fn find(&self, hash: &HashBytes, threshold: u32, results: &mut Vec<(HashBytes, u32)>) {
        let d = self.hash.hamming_distance(hash);
        if d <= threshold {
            results.push((self.hash.clone(), d));
        }

        let lo = d.saturating_sub(threshold);
        let hi = d.saturating_add(threshold);

        for &(child_dist, ref child_node) in &self.children {
            if child_dist >= lo && child_dist <= hi {
                child_node.find(hash, threshold, results);
            }
        }
    }

    fn find_nearest(&self, hash: &HashBytes, best: &mut Option<(HashBytes, u32)>) {
        let d = self.hash.hamming_distance(hash);

        if best.as_ref().is_none_or(|(_, bd)| d < *bd) {
            *best = Some((self.hash.clone(), d));
        }

        let best_d = best.as_ref().map_or(u32::MAX, |(_, bd)| *bd);
        let lo = d.saturating_sub(best_d);
        let hi = d.saturating_add(best_d);

        for &(child_dist, ref child_node) in &self.children {
            if child_dist >= lo && child_dist <= hi {
                child_node.find_nearest(hash, best);
            }
        }
    }
}

/// 变长哈希 BK-tree，用于基于汉明距离的近似最近邻搜索。
///
/// 与 [`super::bktree::BkTree`] 功能等价，但支持 64-bit 到 4096-bit
/// 的变长哈希。内部使用 [`HashBytes`] 存储哈希，汉明距离按字节
/// XOR + popcount 计算。
///
/// # 使用示例
///
/// ```no_run
/// use gpgpu_tool::tasks::hash_bytes::HashBytes;
/// use gpgpu_tool::tasks::bktree_bytes::BkTreeBytes;
///
/// let hashes = vec![
///     HashBytes::from_u64(0xA1B2C3D4),
///     HashBytes::from_u64(0x12345678),
///     HashBytes::from_u64(0x87654321),
/// ];
/// let tree = BkTreeBytes::from_hashes(hashes);
///
/// // 查找汉明距离 ≤ 5 的近似哈希
/// let query = HashBytes::from_u64(0xA1B2C3D4);
/// let similar = tree.find(&query, 5);
/// ```
pub struct BkTreeBytes {
    root: Option<BkNodeBytes>,
    len: usize,
}

impl BkTreeBytes {
    /// 创建空的 BK-tree。
    pub fn new() -> Self {
        Self {
            root: None,
            len: 0,
        }
    }

    /// 从迭代器批量构建 BK-tree，等价于依次调用 [`insert`](BkTreeBytes::insert)。
    pub fn from_hashes(hashes: impl IntoIterator<Item = HashBytes>) -> Self {
        let mut tree = Self::new();
        for hash in hashes {
            tree.insert(hash);
        }
        tree
    }

    /// 从已排序的哈希值批量构建平衡 BK-tree。
    ///
    /// 选择中间元素作为根，递归构建左右子树。
    /// 相比逐个插入，构建的树更平衡，搜索性能更稳定。
    pub fn from_hashes_sorted(mut hashes: Vec<HashBytes>) -> Self {
        if hashes.is_empty() {
            return Self::new();
        }
        hashes.sort();
        hashes.dedup();
        let len = hashes.len();
        let root = Self::build_balanced(&hashes, 0, len);
        Self {
            root: Some(root),
            len,
        }
    }

    fn build_balanced(hashes: &[HashBytes], start: usize, end: usize) -> BkNodeBytes {
        let mid = start + (end - start) / 2;
        let hash = hashes[mid].clone();
        let mut children = Vec::new();

        // 左半部分
        if mid > start {
            let left = Self::build_balanced(hashes, start, mid);
            let d = hash.hamming_distance(&left.hash);
            children.push((d, Box::new(left)));
        }
        // 右半部分
        if mid + 1 < end {
            let right = Self::build_balanced(hashes, mid + 1, end);
            let d = hash.hamming_distance(&right.hash);
            children.push((d, Box::new(right)));
        }

        BkNodeBytes { hash, children }
    }

    /// 插入一个哈希值到树中。
    pub fn insert(&mut self, hash: HashBytes) {
        self.len += 1;
        match &mut self.root {
            None => {
                self.root = Some(BkNodeBytes {
                    hash,
                    children: Vec::new(),
                });
            }
            Some(root) => {
                root.insert(hash);
            }
        }
    }

    /// 查找与 `hash` 的汉明距离 ≤ `threshold` 的所有哈希及其距离。
    ///
    /// 返回 `Vec<(HashBytes, u32)>`，按距离升序排列。
    pub fn find(&self, hash: &HashBytes, threshold: u32) -> Vec<(HashBytes, u32)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            root.find(hash, threshold, &mut results);
        }
        results.sort_by_key(|(_, d)| *d);
        results
    }

    /// 查找树中与 `hash` 最近邻的哈希及其距离。
    ///
    /// 树为空时返回 `None`。
    pub fn find_nearest(&self, hash: &HashBytes) -> Option<(HashBytes, u32)> {
        let mut best: Option<(HashBytes, u32)> = None;
        if let Some(root) = &self.root {
            root.find_nearest(hash, &mut best);
        }
        best
    }

    /// 返回树中的元素数量。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 树是否为空。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Default for BkTreeBytes {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(v: u64) -> HashBytes {
        HashBytes::from_u64(v)
    }

    #[test]
    fn test_empty_tree() {
        let tree = BkTreeBytes::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.find(&h(0), 10).is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut tree = BkTreeBytes::new();
        tree.insert(h(42));
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);

        let results = tree.find(&h(42), 0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, h(42));
        assert_eq!(results[0].1, 0);

        let results = tree.find(&h(43), 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut tree = BkTreeBytes::new();
        tree.insert(h(42));
        tree.insert(h(42));
        assert_eq!(tree.len(), 2);

        let results = tree.find(&h(42), 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_threshold() {
        let mut tree = BkTreeBytes::new();
        tree.insert(h(0b0000));
        tree.insert(h(0b0001));
        tree.insert(h(0b0011));
        tree.insert(h(0b1111));

        let results = tree.find(&h(0b0000), 1);
        let mut hashes: Vec<u64> = results.iter().map(|(h, _)| h.to_u64s()[0]).collect();
        hashes.sort();
        assert_eq!(hashes, vec![0b0000, 0b0001]);
    }

    #[test]
    fn test_from_hashes() {
        let hashes = vec![h(1), h(2), h(4), h(8), h(16)];
        let tree = BkTreeBytes::from_hashes(hashes);
        assert_eq!(tree.len(), 5);

        let results = tree.find(&h(1), 1);
        assert!(!results.is_empty());
        assert!(results.iter().any(|(hb, d)| *hb == h(1) && *d == 0));
    }

    #[test]
    fn test_from_hashes_sorted() {
        let hashes = vec![h(16), h(1), h(8), h(2), h(4)];
        let tree = BkTreeBytes::from_hashes_sorted(hashes);
        assert_eq!(tree.len(), 5);

        let results = tree.find(&h(1), 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_variable_length_hashes() {
        // 256-bit 哈希测试
        let long_hash = HashBytes::from_u64s(&[0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD]);
        let similar_long = HashBytes::from_u64s(&[0xAAAB, 0xBBBB, 0xCCCC, 0xDDDD]);

        let mut tree = BkTreeBytes::new();
        tree.insert(long_hash.clone());

        let results = tree.find(&similar_long, 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1); // 仅 1 位差异
    }

    #[test]
    fn test_find_nearest() {
        let hashes = vec![h(0b0000), h(0b0100), h(0b1000)];
        let tree = BkTreeBytes::from_hashes(hashes);

        let nearest = tree.find_nearest(&h(0b0001));
        assert!(nearest.is_some());
        let (hash, dist) = nearest.unwrap();
        assert_eq!(hash, h(0b0000));
        assert_eq!(dist, 1);
    }
}
