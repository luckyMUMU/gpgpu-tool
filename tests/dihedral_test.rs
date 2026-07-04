use std::collections::HashSet;

use gpgpu_tool::DihedralHashes64;
use gpgpu_tool::DihedralHashes256;

#[test]
fn test_dihedral_64_all_variants_unique() {
    // 构造一个有足够变化的位矩阵
    let hash: u64 = 0x1234_5678_9ABC_DEF0;
    let dihedral = DihedralHashes64::from_u64(hash);
    let all = dihedral.all();

    // 非对称哈希的二面体变体应至少有 2 种不同值
    let unique_count: HashSet<u64> = all.iter().copied().collect();
    assert!(
        unique_count.len() >= 2,
        "非对称哈希的二面体变体应至少有 2 种不同值"
    );
}

#[test]
fn test_dihedral_64_rotate180_double() {
    let hash: u64 = 0xA5A5_A5A5_A5A5_A5A5;
    let d1 = DihedralHashes64::from_u64(hash);
    // rotate180 的 rotate180 应该是 original
    let d2 = DihedralHashes64::from_u64(d1.rotate180);
    assert_eq!(d2.rotate180, hash, "旋转 180° 两次应恢复原始");
}

#[test]
fn test_dihedral_64_rotate90_four_times() {
    let hash: u64 = 0x0F0F_F0F0_55AA_55AA;
    let d1 = DihedralHashes64::from_u64(hash);
    let d2 = DihedralHashes64::from_u64(d1.rotate90);
    let d3 = DihedralHashes64::from_u64(d2.rotate90);
    let d4 = DihedralHashes64::from_u64(d3.rotate90);
    // 旋转 90° 四次应回到原始
    assert_eq!(d4.rotate90, hash, "旋转 90° 四次应回到原始");
}

#[test]
fn test_dihedral_64_flip_h_twice() {
    let hash: u64 = 0xDEAD_BEEF_CAFE_BABE;
    let d1 = DihedralHashes64::from_u64(hash);
    let d2 = DihedralHashes64::from_u64(d1.flip_h);
    assert_eq!(d2.flip_h, hash, "水平翻转两次应恢复原始");
}

#[test]
fn test_dihedral_64_flip_v_twice() {
    let hash: u64 = 0x1234_5678_ABCD_EF01;
    let d1 = DihedralHashes64::from_u64(hash);
    let d2 = DihedralHashes64::from_u64(d1.flip_v);
    assert_eq!(d2.flip_v, hash, "垂直翻转两次应恢复原始");
}

#[test]
fn test_dihedral_64_all_zeros() {
    let d = DihedralHashes64::from_u64(0);
    for v in &d.all() {
        assert_eq!(*v, 0, "全零哈希的所有二面体变体应为零");
    }
}

#[test]
fn test_dihedral_64_all_ones() {
    let d = DihedralHashes64::from_u64(u64::MAX);
    for v in &d.all() {
        assert_eq!(*v, u64::MAX, "全一哈希的所有二面体变体应为 u64::MAX");
    }
}

#[test]
fn test_dihedral_256_from_u64_array_roundtrip() {
    let hash: [u64; 4] = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x9999_AAAA_BBBB_CCCC, 0xDDDD_EEEE_FFFF_0000];
    let d = DihedralHashes256::from_u64_array(hash);
    assert_eq!(d.original, hash, "original 应与输入一致");
}

#[test]
fn test_dihedral_256_rotate180_double() {
    let hash: [u64; 4] = [0xA1B2_C3D4, 0xE5F6_0718, 0x1928_3746, 0x5564_7382];
    let d1 = DihedralHashes256::from_u64_array(hash);
    let d2 = DihedralHashes256::from_u64_array(d1.rotate180);
    assert_eq!(d2.rotate180, hash, "16×16 旋转 180° 两次应恢复原始");
}

#[test]
fn test_dihedral_256_rotate90_four_times() {
    let hash: [u64; 4] = [0x0F0F_F0F0, 0x55AA_55AA, 0x1234_5678, 0xABCD_EF01];
    let d1 = DihedralHashes256::from_u64_array(hash);
    let d2 = DihedralHashes256::from_u64_array(d1.rotate90);
    let d3 = DihedralHashes256::from_u64_array(d2.rotate90);
    let d4 = DihedralHashes256::from_u64_array(d3.rotate90);
    assert_eq!(d4.rotate90, hash, "16×16 旋转 90° 四次应回到原始");
}

#[test]
fn test_dihedral_256_all_variants_count() {
    let hash: [u64; 4] = [1, 2, 3, 4];
    let d = DihedralHashes256::from_u64_array(hash);
    assert_eq!(d.all().len(), 8, "应有 8 种二面体变体");
}


