use crate::{declare_phash_computer, impl_phash_computer_custom_dims};

const GRADIENT_HASH_WGSL: &str = include_str!("gradient_hash.wgsl");

declare_phash_computer!(
    GradientHashComputer,
    "Gradient Hash（水平梯度哈希）GPU 计算器。\n\n\
     对每幅 8x9 灰度图像计算每行相邻像素的水平差值，\n\
     差值大于 0 生成 1bit，最终输出 64bit 哈希值。",
    GRADIENT_HASH_WGSL,
    [256, 1, 1]
);

impl_phash_computer_custom_dims!(GradientHashComputer, 0);