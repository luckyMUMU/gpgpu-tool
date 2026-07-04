/// 变长哈希类型，支持 64-bit 到 4096-bit 的感知哈希。
///
/// 内部使用 `Vec<u8>` 存储哈希字节，提供汉明距离计算、
/// 从 `Vec<u64>`（GPU 计算结果）和 `Vec<u8>`（缓存数据）转换的方法。
///
/// # 哈希尺寸对应关系
///
/// | hash_size | 网格尺寸 | 字节数 | 位宽   |
/// |-----------|---------|--------|--------|
/// | 8         | 8×8     | 8      | 64     |
/// | 16        | 16×16   | 32     | 256    |
/// | 32        | 32×32   | 128    | 1024   |
/// | 64        | 64×64   | 512    | 4096   |
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HashBytes(Vec<u8>);

impl HashBytes {
    /// 从字节向量创建哈希。
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// 从 u64 切片创建哈希（GPU 计算结果）。
    ///
    /// u64 以小端序转换为字节。
    pub fn from_u64s(u64s: &[u64]) -> Self {
        Self(u64s.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// 从单个 u64 创建 64-bit 哈希。
    pub fn from_u64(hash: u64) -> Self {
        Self(hash.to_le_bytes().to_vec())
    }

    /// 返回哈希的字节切片。
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    /// 返回哈希的字节数。
    pub fn byte_len(&self) -> usize {
        self.0.len()
    }

    /// 返回哈希的位宽。
    pub fn bit_len(&self) -> usize {
        self.0.len() * 8
    }

    /// 计算与另一个哈希的汉明距离。
    ///
    /// 逐字节 XOR 后统计置位数量（popcount）。
    /// 两个哈希长度不同时，仅比较公共前缀，多余字节按位全计为差异。
    pub fn hamming_distance(&self, other: &HashBytes) -> u32 {
        let min_len = self.0.len().min(other.0.len());
        let mut distance: u32 = 0;

        // 比较公共部分
        for i in 0..min_len {
            distance += (self.0[i] ^ other.0[i]).count_ones();
        }

        // 多余字节按位全计为差异
        if self.0.len() > min_len {
            for &byte in &self.0[min_len..] {
                distance += byte.count_ones();
            }
        }
        if other.0.len() > min_len {
            for &byte in &other.0[min_len..] {
                distance += byte.count_ones();
            }
        }

        distance
    }

    /// 转换为 u64 切片视图。
    ///
    /// 如果字节数不是 8 的倍数，返回 None。
    pub fn as_u64s(&self) -> Option<&[u64]> {
        if !self.0.len().is_multiple_of(8) {
            return None;
        }
        // 安全：字节对齐由 Vec<u8> 保证，长度是 8 的倍数
        Some(unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const u64,
                self.0.len() / 8,
            )
        })
    }

    /// 提取为 u64 向量。
    pub fn to_u64s(&self) -> Vec<u64> {
        self.0
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// 返回内部字节向量，消费此哈希。
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// 判断哈希是否全零。
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// 判断哈希是否全 0xFF。
    pub fn is_max(&self) -> bool {
        self.0.iter().all(|&b| b == 0xFF)
    }
}

impl std::fmt::Display for HashBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 仅显示前 8 字节的十六进制，过长时省略
        if self.0.len() <= 8 {
            for byte in &self.0 {
                write!(f, "{:02x}", byte)?;
            }
        } else {
            for byte in &self.0[..8] {
                write!(f, "{:02x}", byte)?;
            }
            write!(f, "..({}bytes)", self.0.len())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_same() {
        let a = HashBytes::from_bytes(vec![0xFF, 0x00, 0xAB]);
        assert_eq!(a.hamming_distance(&a), 0);
    }

    #[test]
    fn test_hamming_distance_simple() {
        let a = HashBytes::from_bytes(vec![0b00000001]);
        let b = HashBytes::from_bytes(vec![0b00000010]);
        // XOR = 0b00000011, 2 bits differ
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_hamming_distance_u64() {
        let a = HashBytes::from_u64(0x0000000000000001);
        let b = HashBytes::from_u64(0x0000000000000003);
        // XOR = 2, 1 bit differs
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_hamming_distance_different_lengths() {
        let a = HashBytes::from_bytes(vec![0xFF]);
        let b = HashBytes::from_bytes(vec![0xFF, 0x0F]);
        // 公共部分 XOR=0, 多余字节 0x0F 有 4 个置位
        assert_eq!(a.hamming_distance(&b), 4);
    }

    #[test]
    fn test_from_u64s_roundtrip() {
        let original = vec![0x1234567890ABCDEFu64, 0xFEDCBA0987654321u64];
        let hash = HashBytes::from_u64s(&original);
        assert_eq!(hash.byte_len(), 16);
        let roundtrip = hash.to_u64s();
        assert_eq!(original, roundtrip);
    }

    #[test]
    fn test_is_zero_is_max() {
        let zero = HashBytes::from_bytes(vec![0, 0, 0]);
        let max = HashBytes::from_bytes(vec![0xFF, 0xFF]);
        let mixed = HashBytes::from_bytes(vec![0x00, 0xFF]);
        assert!(zero.is_zero());
        assert!(!zero.is_max());
        assert!(max.is_max());
        assert!(!max.is_zero());
        assert!(!mixed.is_zero());
        assert!(!mixed.is_max());
    }
}
