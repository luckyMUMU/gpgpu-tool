//! # 二面体变换（Dihedral Transforms）
//!
//! 对哈希位矩阵执行 D4 群的 8 种旋转/翻转变换，
//! 无需重新计算图像哈希即可匹配旋转/翻转后的图像。
//!
//! 8 种变换：
//! - original：原始
//! - rotate90：顺时针旋转 90°
//! - rotate180：旋转 180°
//! - rotate270：顺时针旋转 270°
//! - flip_h：水平翻转（左右镜像）
//! - flip_v：垂直翻转（上下镜像）
//! - flip_diag：主对角线翻转（转置）
//! - flip_anti_diag：反对角线翻转

// ── 8×8 位矩阵操作 ──────────────────────────────────────────────

/// 顺时针旋转 90°：new[col][N-1-row] = old[row][col]
fn rotate_90_cw_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[col * 8 + (7 - row)] = bits[row * 8 + col];
        }
    }
    out
}

/// 旋转 180°：new[N-1-row][N-1-col] = old[row][col]
fn rotate_180_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[(7 - row) * 8 + (7 - col)] = bits[row * 8 + col];
        }
    }
    out
}

/// 顺时针旋转 270°（= 逆时针 90°）：new[N-1-col][row] = old[row][col]
fn rotate_270_cw_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[(7 - col) * 8 + row] = bits[row * 8 + col];
        }
    }
    out
}

/// 水平翻转（左右镜像）：new[row][N-1-col] = old[row][col]
fn flip_horizontal_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[row * 8 + (7 - col)] = bits[row * 8 + col];
        }
    }
    out
}

/// 垂直翻转（上下镜像）：new[N-1-row][col] = old[row][col]
fn flip_vertical_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[(7 - row) * 8 + col] = bits[row * 8 + col];
        }
    }
    out
}

/// 主对角线翻转（转置）：new[col][row] = old[row][col]
fn flip_diagonal_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[col * 8 + row] = bits[row * 8 + col];
        }
    }
    out
}

/// 反对角线翻转：new[N-1-col][N-1-row] = old[row][col]
fn flip_anti_diagonal_8x8(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 64];
    for row in 0..8usize {
        for col in 0..8usize {
            out[(7 - col) * 8 + (7 - row)] = bits[row * 8 + col];
        }
    }
    out
}

/// 将 8×8 位矩阵打包为 u64（LSB-first：bit i 在 byte i/8 的 bit i%8）
fn pack_8x8(bits: &[bool]) -> u64 {
    let mut hash: u64 = 0;
    for (i, &b) in bits.iter().enumerate() {
        if b {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// 将 u64 解包为 8×8 位矩阵（LSB-first）
fn unpack_u64_to_8x8(hash: u64) -> Vec<bool> {
    (0..64).map(|i| (hash >> i) & 1 == 1).collect()
}

// ── 16×16 位矩阵操作 ────────────────────────────────────────────

/// 顺时针旋转 90°（16×16）
fn rotate_90_cw_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[col * 16 + (15 - row)] = bits[row * 16 + col];
        }
    }
    out
}

/// 旋转 180°（16×16）
fn rotate_180_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[(15 - row) * 16 + (15 - col)] = bits[row * 16 + col];
        }
    }
    out
}

/// 顺时针旋转 270°（16×16）
fn rotate_270_cw_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[(15 - col) * 16 + row] = bits[row * 16 + col];
        }
    }
    out
}

/// 水平翻转（16×16）
fn flip_horizontal_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[row * 16 + (15 - col)] = bits[row * 16 + col];
        }
    }
    out
}

/// 垂直翻转（16×16）
fn flip_vertical_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[(15 - row) * 16 + col] = bits[row * 16 + col];
        }
    }
    out
}

/// 主对角线翻转（16×16）
fn flip_diagonal_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[col * 16 + row] = bits[row * 16 + col];
        }
    }
    out
}

/// 反对角线翻转（16×16）
fn flip_anti_diagonal_16x16(bits: &[bool]) -> Vec<bool> {
    let mut out = vec![false; 256];
    for row in 0..16usize {
        for col in 0..16usize {
            out[(15 - col) * 16 + (15 - row)] = bits[row * 16 + col];
        }
    }
    out
}

/// 将 16×16 位矩阵打包为 4 个 u64（LSB-first）
fn pack_16x16(bits: &[bool]) -> [u64; 4] {
    let mut result = [0u64; 4];
    for i in 0..256 {
        if bits[i] {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
    result
}

/// 将 4 个 u64 解包为 16×16 位矩阵（LSB-first）
fn unpack_u64_array_to_16x16(hash: &[u64; 4]) -> Vec<bool> {
    let mut bits = vec![false; 256];
    for chunk in 0..4 {
        for bit in 0..64 {
            bits[chunk * 64 + bit] = (hash[chunk] >> bit) & 1 == 1;
        }
    }
    bits
}

// ── 公共类型 ─────────────────────────────────────────────────────

/// 64-bit 哈希（8×8 位矩阵）的二面体变换结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DihedralHashes64 {
    pub original: u64,
    pub rotate90: u64,
    pub rotate180: u64,
    pub rotate270: u64,
    pub flip_h: u64,
    pub flip_v: u64,
    pub flip_diag: u64,
    pub flip_anti_diag: u64,
}

impl DihedralHashes64 {
    /// 从 8×8 位矩阵推导所有变体。
    /// bits 长度必须为 64，按行优先排列（bits[row * 8 + col]）。
    pub fn from_bits(bits: &[bool]) -> Self {
        assert_eq!(bits.len(), 64);

        let original = pack_8x8(bits);
        let rotate90 = pack_8x8(&rotate_90_cw_8x8(bits));
        let rotate180 = pack_8x8(&rotate_180_8x8(bits));
        let rotate270 = pack_8x8(&rotate_270_cw_8x8(bits));
        let flip_h = pack_8x8(&flip_horizontal_8x8(bits));
        let flip_v = pack_8x8(&flip_vertical_8x8(bits));
        let flip_diag = pack_8x8(&flip_diagonal_8x8(bits));
        let flip_anti_diag = pack_8x8(&flip_anti_diagonal_8x8(bits));

        Self {
            original,
            rotate90,
            rotate180,
            rotate270,
            flip_h,
            flip_v,
            flip_diag,
            flip_anti_diag,
        }
    }

    /// 从 u64 哈希值推导所有变体。
    pub fn from_u64(hash: u64) -> Self {
        let bits = unpack_u64_to_8x8(hash);
        Self::from_bits(&bits)
    }

    /// 返回所有变体。
    pub fn all(&self) -> [u64; 8] {
        [
            self.original,
            self.rotate90,
            self.rotate180,
            self.rotate270,
            self.flip_h,
            self.flip_v,
            self.flip_diag,
            self.flip_anti_diag,
        ]
    }
}

/// 256-bit 哈希（16×16 位矩阵）的二面体变换结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DihedralHashes256 {
    pub original: [u64; 4],
    pub rotate90: [u64; 4],
    pub rotate180: [u64; 4],
    pub rotate270: [u64; 4],
    pub flip_h: [u64; 4],
    pub flip_v: [u64; 4],
    pub flip_diag: [u64; 4],
    pub flip_anti_diag: [u64; 4],
}

impl DihedralHashes256 {
    /// 从 16×16 位矩阵推导所有变体。
    /// bits 长度必须为 256，按行优先排列。
    pub fn from_bits(bits: &[bool]) -> Self {
        assert_eq!(bits.len(), 256);

        let original = pack_16x16(bits);
        let rotate90 = pack_16x16(&rotate_90_cw_16x16(bits));
        let rotate180 = pack_16x16(&rotate_180_16x16(bits));
        let rotate270 = pack_16x16(&rotate_270_cw_16x16(bits));
        let flip_h = pack_16x16(&flip_horizontal_16x16(bits));
        let flip_v = pack_16x16(&flip_vertical_16x16(bits));
        let flip_diag = pack_16x16(&flip_diagonal_16x16(bits));
        let flip_anti_diag = pack_16x16(&flip_anti_diagonal_16x16(bits));

        Self {
            original,
            rotate90,
            rotate180,
            rotate270,
            flip_h,
            flip_v,
            flip_diag,
            flip_anti_diag,
        }
    }

    /// 从 4 个 u64 哈希值推导所有变体。
    pub fn from_u64_array(hash: [u64; 4]) -> Self {
        let bits = unpack_u64_array_to_16x16(&hash);
        Self::from_bits(&bits)
    }

    /// 返回所有变体。
    pub fn all(&self) -> [[u64; 4]; 8] {
        [
            self.original,
            self.rotate90,
            self.rotate180,
            self.rotate270,
            self.flip_h,
            self.flip_v,
            self.flip_diag,
            self.flip_anti_diag,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 辅助：构造一个 8×8 位矩阵，每个位置的值 = (row * 8 + col) % 3 == 0
    fn sample_8x8_bits() -> Vec<bool> {
        (0..64).map(|i| i % 3 == 0).collect()
    }

    /// 辅助：构造一个 16×16 位矩阵
    fn sample_16x16_bits() -> Vec<bool> {
        (0..256).map(|i| i % 5 == 0).collect()
    }

    #[test]
    fn test_pack_unpack_8x8_roundtrip() {
        let bits = sample_8x8_bits();
        let hash = pack_8x8(&bits);
        let restored = unpack_u64_to_8x8(hash);
        assert_eq!(bits, restored);
    }

    #[test]
    fn test_pack_unpack_16x16_roundtrip() {
        let bits = sample_16x16_bits();
        let hash = pack_16x16(&bits);
        let restored = unpack_u64_array_to_16x16(&hash);
        assert_eq!(bits, restored);
    }

    #[test]
    fn test_rotate90_four_times_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let r1 = rotate_90_cw_8x8(&bits);
        let r2 = rotate_90_cw_8x8(&r1);
        let r3 = rotate_90_cw_8x8(&r2);
        let r4 = rotate_90_cw_8x8(&r3);
        assert_eq!(r4, bits, "旋转 90° 四次应等于原始");
    }

    #[test]
    fn test_rotate180_twice_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let r1 = rotate_180_8x8(&bits);
        let r2 = rotate_180_8x8(&r1);
        assert_eq!(r2, bits, "旋转 180° 两次应等于原始");
    }

    #[test]
    fn test_rotate270_equals_rotate90_three_times_8x8() {
        let bits = sample_8x8_bits();
        let r270 = rotate_270_cw_8x8(&bits);
        let r1 = rotate_90_cw_8x8(&bits);
        let r2 = rotate_90_cw_8x8(&r1);
        let r3 = rotate_90_cw_8x8(&r2);
        assert_eq!(r270, r3, "旋转 270° 应等于旋转 90° 三次");
    }

    #[test]
    fn test_flip_horizontal_twice_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let f1 = flip_horizontal_8x8(&bits);
        let f2 = flip_horizontal_8x8(&f1);
        assert_eq!(f2, bits, "水平翻转两次应等于原始");
    }

    #[test]
    fn test_flip_vertical_twice_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let f1 = flip_vertical_8x8(&bits);
        let f2 = flip_vertical_8x8(&f1);
        assert_eq!(f2, bits, "垂直翻转两次应等于原始");
    }

    #[test]
    fn test_flip_diagonal_twice_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let f1 = flip_diagonal_8x8(&bits);
        let f2 = flip_diagonal_8x8(&f1);
        assert_eq!(f2, bits, "主对角线翻转两次应等于原始");
    }

    #[test]
    fn test_flip_anti_diagonal_twice_equals_original_8x8() {
        let bits = sample_8x8_bits();
        let f1 = flip_anti_diagonal_8x8(&bits);
        let f2 = flip_anti_diagonal_8x8(&f1);
        assert_eq!(f2, bits, "反对角线翻转两次应等于原始");
    }

    #[test]
    fn test_dihedral_hashes_64_from_u64_roundtrip() {
        let hash: u64 = 0x0123_4567_89AB_CDEF;
        let d = DihedralHashes64::from_u64(hash);
        assert_eq!(d.original, hash);
    }

    #[test]
    fn test_dihedral_hashes_64_all_has_eight_variants() {
        let d = DihedralHashes64::from_u64(0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(d.all().len(), 8);
    }

    #[test]
    fn test_dihedral_hashes_64_rotate90_four_times() {
        let hash: u64 = 0x0123_4567_89AB_CDEF;
        let d = DihedralHashes64::from_u64(hash);

        let d2 = DihedralHashes64::from_u64(d.rotate90);
        let d3 = DihedralHashes64::from_u64(d2.rotate90);
        let d4 = DihedralHashes64::from_u64(d3.rotate90);

        // d4.rotate90 = rotate90^4(hash) = hash
        assert_eq!(d4.rotate90, d.original, "旋转 90° 四次应回到原始");
    }

    #[test]
    fn test_dihedral_hashes_256_from_u64_array_roundtrip() {
        let hash: [u64; 4] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x9999_AAAA_BBBB_CCCC, 0xDDDD_EEEE_FFFF_0000];
        let d = DihedralHashes256::from_u64_array(hash);
        assert_eq!(d.original, hash);
    }

    #[test]
    fn test_dihedral_hashes_256_all_has_eight_variants() {
        let hash: [u64; 4] = [1, 2, 3, 4];
        let d = DihedralHashes256::from_u64_array(hash);
        let all = d.all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_rotate90_four_times_equals_original_16x16() {
        let bits = sample_16x16_bits();
        let r1 = rotate_90_cw_16x16(&bits);
        let r2 = rotate_90_cw_16x16(&r1);
        let r3 = rotate_90_cw_16x16(&r2);
        let r4 = rotate_90_cw_16x16(&r3);
        assert_eq!(r4, bits, "16×16 旋转 90° 四次应等于原始");
    }

    #[test]
    fn test_dihedral_hashes_256_rotate90_four_times() {
        let hash: [u64; 4] = [0xA1B2_C3D4_E5F6_0718, 0x1928_3746_5564_7382, 0x9101_1121_3141_5161, 0x7181_9101_A1B1_C1D1];
        let d = DihedralHashes256::from_u64_array(hash);

        let d2 = DihedralHashes256::from_u64_array(d.rotate90);
        let d3 = DihedralHashes256::from_u64_array(d2.rotate90);
        let d4 = DihedralHashes256::from_u64_array(d3.rotate90);

        // d4.rotate90 = rotate90^4(hash) = hash
        assert_eq!(d4.rotate90, d.original, "16×16 旋转 90° 四次应回到原始");
    }

    #[test]
    fn test_identity_matrix_8x8() {
        // 单位矩阵：对角线为 true，其余为 false
        let mut bits = vec![false; 64];
        for i in 0..8 {
            bits[i * 8 + i] = true;
        }
        let d = DihedralHashes64::from_bits(&bits);
        // 单位矩阵的转置等于自身
        assert_eq!(d.original, d.flip_diag, "单位矩阵转置应等于自身");
        // 单位矩阵旋转 180° 也等于自身
        assert_eq!(d.original, d.rotate180, "单位矩阵旋转 180° 应等于自身");
    }

    #[test]
    fn test_all_zeros_8x8() {
        let bits = vec![false; 64];
        let d = DihedralHashes64::from_bits(&bits);
        let all = d.all();
        for v in &all {
            assert_eq!(*v, 0, "全零矩阵所有变换应为零");
        }
    }

    #[test]
    fn test_all_ones_8x8() {
        let bits = vec![true; 64];
        let d = DihedralHashes64::from_bits(&bits);
        let all = d.all();
        for v in &all {
            assert_eq!(*v, u64::MAX, "全一矩阵所有变换应为 u64::MAX");
        }
    }

    #[test]
    fn test_d4_group_closure_8x8() {
        // D4 群封闭性：任意两种变换的组合等价于群中另一种变换
        let bits = sample_8x8_bits();
        let d = DihedralHashes64::from_bits(&bits);

        // rotate90 + flip_h = flip_anti_diag
        let from_r90 = DihedralHashes64::from_u64(d.rotate90);
        assert_eq!(from_r90.flip_h, d.flip_anti_diag,
            "rotate90 后水平翻转应等于反对角线翻转");

        // rotate90 + flip_v = flip_diag
        assert_eq!(from_r90.flip_v, d.flip_diag,
            "rotate90 后垂直翻转应等于主对角线翻转");
    }
}
