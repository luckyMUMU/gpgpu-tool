//! GPGPU-tool 功能演示。
//!
//! 展示 GPU 加速算法库的核心能力：
//! 1. SHA-256 并行哈希
//! 2. 感知哈希（Mean Hash，支持 64-bit / 256-bit / 1024-bit）
//! 3. GPU 距离矩阵 + 哈希匹配
//! 4. 变长哈希匹配器（HashBytes + BK-tree + 门面模式）
//! 5. 二面体变换（D4 群旋转不变匹配）

use gpgpu_tool::{
    tasks::phasher::HashAlgorithm, tasks::sha256::Sha256Computer, GpuContext, GpuImageMatcher,
    HashBytes, HashMatcherBytes, HashMatcherFacadeBytes, HashSize,
};

fn main() {
    env_logger::init();

    println!("=== GPGPU-tool 功能演示 ===\n");

    // ── 1. GPU 上下文初始化 ──────────────────────────────────────
    println!("[1] 初始化 GPU 上下文...");
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    println!("    适配器: {}", ctx.adapter_info());

    // ── 2. SHA-256 并行哈希 ──────────────────────────────────────
    println!("\n[2] SHA-256 并行哈希");
    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = vec![
        b"hello world".to_vec(),
        b"wgpu-compute-engine".to_vec(),
        vec![0xABu8; 100],
    ];

    let results = sha256.compute(&ctx, &messages).expect("SHA-256 计算失败");

    for (i, (msg, hash)) in messages.iter().zip(results.iter()).enumerate() {
        let hex_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        let msg_preview = if msg.len() > 32 {
            format!("{:?}... ({} bytes)", &msg[..32], msg.len())
        } else {
            format!("{:?}", msg)
        };
        println!("    [{}] {} => {}", i, msg_preview, hex_hash);
    }

    // ── 3. 感知哈希（64-bit / 256-bit / 1024-bit）───────────────
    println!("\n[3] 感知哈希 — 多尺寸对比");

    // 生成测试图像：8×8 灰度
    let img_a = vec![100u8; 64]; // 均匀灰度 100
    let img_b = vec![102u8; 64]; // 均匀灰度 102（非常相似）
    let img_c = vec![200u8; 64]; // 均匀灰度 200（完全不同）
    let dims = vec![(8u32, 8u32)];

    for &size in &[HashSize::new(8), HashSize::new(16), HashSize::new(32)] {
        let bit_count = size.bits();
        let matcher =
            GpuImageMatcher::with_hash_size(&mut ctx, HashAlgorithm::Mean, size)
                .expect("GpuImageMatcher 创建失败");

        let hashes = matcher
            .compute_hashes(&ctx, &[img_a.clone(), img_b.clone(), img_c.clone()], &[dims[0]; 3])
            .expect("哈希计算失败");

        println!(
            "    hash_size={} ({}-bit): A={} B={} C={}",
            size.size(),
            bit_count,
            hex_preview(&hashes[0]),
            hex_preview(&hashes[1]),
            hex_preview(&hashes[2]),
        );

        // 打印详细字节用于调试
        if size.size() >= 16 {
            // 直接用 MeanHashComputer 测试（不经过 PerceptualHasher 的 resize）
            use gpgpu_tool::tasks::mean_hash::MeanHashComputer;
            use gpgpu_tool::tasks::hash_common::{PerceptualHashComputer, HashSize as HS};
            let direct_hasher = MeanHashComputer::with_config(&mut ctx, [8, 8, 1], HS::new(size.size()))
                .expect("MeanHashComputer 创建失败");
            // 生成目标尺寸的均匀图像
            let s = size.size() as usize;
            let resized_img = vec![100u8; s * s];
            let direct_result = direct_hasher.compute(&ctx, std::slice::from_ref(&resized_img))
                .expect("直接计算失败");
            println!("    MeanHashComputer 直接计算: {} 个 u64", direct_result.len());
            for (i, h) in direct_result.iter().enumerate().take(4) {
                println!("      [{}] {:#018x}", i, h);
            }
        }

        // 距离矩阵
        let matrix = matcher
            .compute_distance_matrix(&ctx, &[img_a.clone()], &dims, &[img_b.clone(), img_c.clone()], &[dims[0]; 2])
            .expect("距离矩阵计算失败");

        println!(
            "    距离: A-B={} A-C={}",
            matrix[0][0], matrix[0][1]
        );
    }

    // ── 4. 变长哈希匹配器 ────────────────────────────────────────
    println!("\n[4] 变长哈希匹配器 (HashBytes + BK-tree + 门面)");

    // 构造 64-bit 哈希数据库
    let db_hashes_64: Vec<HashBytes> = vec![
        HashBytes::from_u64(0x0000_0000_0000_0001),
        HashBytes::from_u64(0x0000_0000_0000_0003),
        HashBytes::from_u64(0xFFFF_FFFF_FFFF_FFFF),
        HashBytes::from_u64(0xAAAA_AAAA_AAAA_AAAA),
        HashBytes::from_u64(0x5555_5555_5555_5555),
    ];

    let query = HashBytes::from_u64(0x0000_0000_0000_0002);

    // 线性扫描
    let linear = gpgpu_tool::LinearScanMatcherBytes::new(db_hashes_64.clone());
    let linear_results = linear.find_similar(&query, 5);
    println!("    线性扫描 (threshold=5): {} 个匹配", linear_results.len());
    for r in &linear_results {
        println!("      distance={}, hash={}", r.distance, hex_preview(&r.hash));
    }

    // BK-tree
    let bktree = gpgpu_tool::BkTreeBytes::from_hashes(db_hashes_64.clone());
    let bk_results = bktree.find(&query, 5);
    println!("    BK-tree (threshold=5): {} 个匹配", bk_results.len());
    for (hash, dist) in &bk_results {
        println!("      distance={}, hash={}", dist, hex_preview(hash));
    }

    // 门面模式（含二面体变换增强）
    let facade = HashMatcherFacadeBytes::bktree(db_hashes_64.clone()).with_dihedral();
    let facade_results = facade.find_similar(&query, 5);
    println!("    门面+二面体 (threshold=5): {} 个匹配", facade_results.len());
    for r in &facade_results {
        println!("      distance={}, hash={}", r.distance, hex_preview(&r.hash));
    }

    // ── 5. 256-bit 变长哈希匹配 ──────────────────────────────────
    println!("\n[5] 256-bit 变长哈希匹配");

    let db_256: Vec<HashBytes> = vec![
        HashBytes::from_u64s(&[0x1111_1111_1111_1111, 0x2222_2222_2222_2222, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444]),
        HashBytes::from_u64s(&[0x1111_1111_1111_1110, 0x2222_2222_2222_2222, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444]),
        HashBytes::from_u64s(&[0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]),
    ];
    let query_256 = HashBytes::from_u64s(&[0x1111_1111_1111_1111, 0x2222_2222_2222_2222, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444]);

    let facade_256 = HashMatcherFacadeBytes::bktree(db_256).with_dihedral();
    let results_256 = facade_256.find_similar(&query_256, 5);
    println!("    256-bit 门面匹配 (threshold=5): {} 个结果", results_256.len());
    for r in &results_256 {
        println!("      distance={}, hash={}", r.distance, hex_preview(&r.hash));
    }

    // ── 6. 二面体变换 ────────────────────────────────────────────
    println!("\n[6] 二面体变换 (D4 群 8 种旋转/翻转)");

    let hash_64 = 0x0123_4567_89AB_CDEFu64;
    let d64 = gpgpu_tool::DihedralHashes64::from_u64(hash_64);
    println!("    64-bit 原始:  {:#018x}", d64.original);
    println!("    64-bit 旋转90: {:#018x}", d64.rotate90);
    println!("    64-bit 翻转H:  {:#018x}", d64.flip_h);

    let hash_256 = [0x1111_1111_1111_1111u64, 0x2222_2222_2222_2222, 0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let d256 = gpgpu_tool::DihedralHashes256::from_u64_array(hash_256);
    println!("    256-bit 原始:  {:016x}{:016x}...", d256.original[0], d256.original[1]);
    println!("    256-bit 旋转90: {:016x}{:016x}...", d256.rotate90[0], d256.rotate90[1]);

    let hash_1024: [u64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let d1024 = gpgpu_tool::DihedralHashes1024::from_u64_array(hash_1024);
    println!("    1024-bit 原始:  {:?}...", &d1024.original[..4]);
    println!("    1024-bit 旋转90: {:?}...", &d1024.rotate90[..4]);

    println!("\n=== 演示完成 ===");
}

/// 哈希值的十六进制预览（最多显示前 16 字节）。
fn hex_preview(hash: &HashBytes) -> String {
    let bytes = hash.as_bytes();
    let preview_len = bytes.len().min(16);
    let hex: String = bytes[..preview_len]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect();
    if bytes.len() > 16 {
        format!("{}... ({}B)", hex, bytes.len())
    } else {
        hex
    }
}
