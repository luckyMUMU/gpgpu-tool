use crate::{declare_phash_computer, impl_phash_computer_simple};

const MEAN_HASH_WGSL: &str = include_str!("mean_hash.wgsl");

declare_phash_computer!(
    MeanHashComputer,
    "Mean Hash（均值哈希）GPU 计算器。\n\n\
     对每幅 8x8 灰度图像计算所有像素的平均值，\n\
     每个像素与均值比较生成 1bit，最终输出 64bit 哈希值。",
    MEAN_HASH_WGSL,
    [8, 8, 1]
);

impl_phash_computer_simple!(MeanHashComputer);