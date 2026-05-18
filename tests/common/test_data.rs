/// 合成图像数据生成器，用于测试。
///
/// 提供多种生成模式，支持任意尺寸，保证测试可复现。

/// 生成全相同像素的图像。
pub fn uniform_image(size: usize, value: u8) -> Vec<u8> {
    vec![value; size]
}

/// 生成渐变图像（0, 1, 2, ... 循环）。
pub fn gradient_image(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

/// 使用固定种子生成伪随机图像。
pub fn random_image(size: usize) -> Vec<u8> {
    // 使用简单的 LCG 伪随机数生成器，保证跨平台可复现
    let mut state: u32 = 12345;
    (0..size)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            ((state >> 16) & 0xFF) as u8
        })
        .collect()
}

/// 生成水平渐变图像（每行从左到右递增）。
pub fn horizontal_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for col in 0..width {
            pixels.push(((row * width + col) % 256) as u8);
        }
    }
    pixels
}

/// 生成垂直渐变图像（每列从上到下递增）。
pub fn vertical_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for col in 0..width {
            pixels.push(((col * height + row) % 256) as u8);
        }
    }
    pixels
}

/// 获取算法推荐的测试尺寸。
pub fn get_test_sizes(algorithm: &str) -> Vec<(u32, u32)> {
    match algorithm {
        "mean" | "median" => vec![(8, 8), (16, 16), (32, 32)],
        "gradient" => vec![(8, 9), (16, 17), (32, 33)],
        "vert_gradient" => vec![(9, 8), (17, 16), (33, 32)],
        "block" => vec![(16, 16), (32, 32)],
        "double_gradient" => vec![(9, 9), (17, 17), (33, 33)],
        _ => vec![(8, 8)],
    }
}
