/// 计算两个 64 位哈希值之间的汉明距离。
///
/// 汉明距离定义为对应位不同的数量，即 `(a ^ b).count_ones()`。
/// 用于 BK-tree 中的相似度度量。
///
/// # 示例
///
/// ```
/// use gpgpu_tool::tasks::bktree::hamming_distance;
///
/// assert_eq!(hamming_distance(0, 0), 0);
/// assert_eq!(hamming_distance(0xFFFFFFFFFFFFFFFF, 0), 64);
/// assert_eq!(hamming_distance(0b1010, 0b0101), 4);
/// ```
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

struct BkNode {
    hash: u64,
    children: Vec<(u32, Box<BkNode>)>,
}

impl BkNode {
    fn insert(&mut self, hash: u64) {
        let d = hamming_distance(self.hash, hash);
        match self.children.binary_search_by_key(&d, |(dist, _)| *dist) {
            Ok(idx) => self.children[idx].1.insert(hash),
            Err(idx) => self.children.insert(idx, (d, Box::new(BkNode {
                hash,
                children: Vec::new(),
            }))),
        }
    }

    fn find(&self, hash: u64, threshold: u32, results: &mut Vec<(u64, u32)>) {
        let d = hamming_distance(self.hash, hash);
        if d <= threshold {
            results.push((self.hash, d));
        }

        let lo = d.saturating_sub(threshold);
        let hi = d.saturating_add(threshold);

        for &(child_dist, ref child_node) in &self.children {
            if child_dist >= lo && child_dist <= hi {
                child_node.find(hash, threshold, results);
            }
        }
    }

    fn find_nearest(&self, hash: u64, best: &mut Option<(u64, u32)>) {
        let d = hamming_distance(self.hash, hash);

        if best.is_none_or(|(_, bd)| d < bd) {
            *best = Some((self.hash, d));
        }

        let best_d = best.map_or(u32::MAX, |(_, bd)| bd);
        let lo = d.saturating_sub(best_d);
        let hi = d.saturating_add(best_d);

        for &(child_dist, ref child_node) in &self.children {
            if child_dist >= lo && child_dist <= hi {
                child_node.find_nearest(hash, best);
            }
        }
    }
}

/// BK-tree（Burkhard-Keller Tree）数据结构，用于基于汉明距离的近似最近邻搜索。
///
/// # 原理
///
/// BK-tree 是一种度量树，利用三角不等式剪枝搜索空间。
/// 插入时每个节点按照与父节点的汉明距离分叉；搜索时只遍历距离范围内的子树。
///
/// 对 N 条记录，搜索复杂度为 O(log N)，远优于暴力搜索的 O(N)。
///
/// # 使用示例
///
/// ```no_run
/// use gpgpu_tool::tasks::bktree::BkTree;
///
/// let hashes = vec![0xA1B2C3D4, 0x12345678, 0x87654321, 0xA1B2C3D5];
/// let tree = BkTree::from_hashes(hashes.iter().copied());
///
/// // 查找汉明距离 ≤ 5 的近似哈希
/// let similar = tree.find(0xA1B2C3D4, 5);
///
/// // 查找最近邻
/// if let Some((nearest, dist)) = tree.find_nearest(0xA1B2C3D4) {
///     println!("最近邻: {:016x}, 距离: {}", nearest, dist);
/// }
/// ```
pub struct BkTree {
    root: Option<BkNode>,
    len: usize,
}

impl BkTree {
    /// 创建空的 BK-tree。
    pub fn new() -> Self {
        Self { root: None, len: 0 }
    }

    /// 从迭代器批量构建 BK-tree，等价于依次调用 [`insert`](BkTree::insert)。
    pub fn from_hashes(hashes: impl IntoIterator<Item = u64>) -> Self {
        let mut tree = Self::new();
        for hash in hashes {
            tree.insert(hash);
        }
        tree
    }

    /// 插入一个哈希值到树中。
    pub fn insert(&mut self, hash: u64) {
        self.len += 1;
        match &mut self.root {
            None => {
                self.root = Some(BkNode {
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
    /// 返回 `Vec<(hash, distance)>`，按距离升序排列。
    pub fn find(&self, hash: u64, threshold: u32) -> Vec<(u64, u32)> {
        let mut results = Vec::new();
        if let Some(root) = &self.root {
            root.find(hash, threshold, &mut results);
        }
        results
    }

    /// 查找树中与 `hash` 最近邻的哈希及其距离。
    ///
    /// 树为空时返回 `None`。
    pub fn find_nearest(&self, hash: u64) -> Option<(u64, u32)> {
        let mut best: Option<(u64, u32)> = None;
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

impl Default for BkTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0xFFFFFFFFFFFFFFFF, 0), 64);
        assert_eq!(hamming_distance(0b1, 0), 1);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
    }

    #[test]
    fn test_empty_tree() {
        let tree = BkTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.find(0, 10).is_empty());
    }

    #[test]
    fn test_single_insert() {
        let mut tree = BkTree::new();
        tree.insert(42);
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);

        let results = tree.find(42, 0);
        assert_eq!(results, vec![(42, 0)]);

        let results = tree.find(43, 1);
        assert_eq!(results, vec![(42, 1)]);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut tree = BkTree::new();
        tree.insert(42);
        tree.insert(42);
        assert_eq!(tree.len(), 2);

        let results = tree.find(42, 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_threshold() {
        let mut tree = BkTree::new();
        tree.insert(0b0000);
        tree.insert(0b0001);
        tree.insert(0b0011);
        tree.insert(0b1111);

        let results = tree.find(0b0000, 1);
        let mut hashes: Vec<u64> = results.iter().map(|&(h, _)| h).collect();
        hashes.sort();
        assert_eq!(hashes, vec![0b0000, 0b0001]);
    }

    #[test]
    fn test_from_hashes() {
        let hashes = vec![1u64, 2, 4, 8, 16];
        let tree = BkTree::from_hashes(hashes);
        assert_eq!(tree.len(), 5);

        let results = tree.find(1, 1);
        assert!(!results.is_empty());
        assert!(results.contains(&(1, 0)));
    }

    #[test]
    fn test_large_threshold() {
        let mut tree = BkTree::new();
        for i in 0u64..100 {
            tree.insert(i);
        }

        let results = tree.find(50, 64);
        assert_eq!(results.len(), 100);
    }
}
