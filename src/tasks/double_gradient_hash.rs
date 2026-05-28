use crate::{declare_phash_computer, impl_phash_computer_simple};

const DOUBLE_GRADIENT_HASH_WGSL: &str = include_str!("double_gradient_hash.wgsl");

declare_phash_computer!(
    DoubleGradientHashComputer,
    "Double Gradient Hash（双梯度哈希）GPU 计算器。\n\n\
     对每幅 9x9 灰度图像同时计算水平和垂直方向的相邻像素差值，\n\
     各取 32bit 组合为 64bit 哈希值。",
    DOUBLE_GRADIENT_HASH_WGSL,
    [8, 8, 1]
);

impl_phash_computer_simple!(DoubleGradientHashComputer);