use crate::{declare_phash_computer, impl_phash_computer_custom_dims};

const VERT_GRADIENT_HASH_WGSL: &str = include_str!("vert_gradient_hash.wgsl");

declare_phash_computer!(
    VertGradientHashComputer,
    "Vertical Gradient Hash（垂直梯度哈希）GPU 计算器。\n\n\
     对每幅 9x8 灰度图像计算每列相邻像素的垂直差值，\n\
     差值大于 0 生成 1bit，最终输出 64bit 哈希值。",
    VERT_GRADIENT_HASH_WGSL,
    [256, 1, 1]
);

impl_phash_computer_custom_dims!(VertGradientHashComputer, 1);