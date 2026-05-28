use crate::{declare_phash_computer, impl_phash_computer_simple};

const BLOCK_HASH_WGSL: &str = include_str!("block_hash.wgsl");

declare_phash_computer!(
    BlockHashComputer,
    "Block Hash（分块哈希）GPU 计算器。\n\n\
     对每幅 16x16 灰度图像分为 8x8 个 2x2 块，\n\
     计算每块均值后相邻块比较生成 1bit，最终输出 64bit 哈希值。",
    BLOCK_HASH_WGSL,
    [8, 8, 1]
);

impl_phash_computer_simple!(BlockHashComputer);