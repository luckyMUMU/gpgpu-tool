/// CPU 参考实现：6 种感知哈希算法（基准测试专用）。
///
/// 与 WGSL 实现保持一致的 bit 映射策略（独立 bit_pos 计数器）。
/// 支持可变 width/height 参数。

/// Mean Hash（均值哈希）CPU 参考实现。
pub fn mean_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    let count = pixels.len().min(64);
    let sum: u32 = pixels[..count].iter().map(|&p| p as u32).sum();
    let mean = sum / count as u32;

    let mut hash: u64 = 0;
    for (i, &pixel) in pixels[..count].iter().enumerate() {
        if pixel as u32 > mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Median Hash（中值哈希）CPU 参考实现。
pub fn median_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    let count = pixels.len().min(64);
    let mut sorted: Vec<u8> = pixels[..count].to_vec();
    sorted.sort_unstable();
    let median = sorted[count / 2];

    let mut hash: u64 = 0;
    for (i, &pixel) in pixels[..count].iter().enumerate() {
        if pixel > median {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Gradient Hash（水平梯度哈希）CPU 参考实现。
pub fn gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let mut hash: u64 = 0;
    let mut bit_pos: u32 = 0;

    for row in 0..height {
        for col in 0..(width - 1) {
            if bit_pos >= 64 {
                break;
            }
            let idx = (row * width + col) as usize;
            let left = pixels[idx];
            let right = pixels[idx + 1];
            if right > left {
                hash |= 1u64 << bit_pos;
            }
            bit_pos += 1;
        }
    }
    hash
}

/// Vertical Gradient Hash（垂直梯度哈希）CPU 参考实现。
pub fn vert_gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let mut hash: u64 = 0;
    let mut bit_pos: u32 = 0;

    for col in 0..width {
        for row in 0..(height - 1) {
            if bit_pos >= 64 {
                break;
            }
            let idx = (row * width + col) as usize;
            let current = pixels[idx];
            let below = pixels[idx + width as usize];
            if below > current {
                hash |= 1u64 << bit_pos;
            }
            bit_pos += 1;
        }
    }
    hash
}

/// Block Hash（分块哈希）CPU 参考实现。
pub fn block_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let blocks_x = 8u32;
    let blocks_y = 8u32;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;

    let mut block_means = Vec::with_capacity(64);
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum: u32 = 0;
            for dy in 0..block_h {
                for dx in 0..block_w {
                    let px = bx * block_w + dx;
                    let py = by * block_h + dy;
                    let idx = (py * width + px) as usize;
                    sum += pixels[idx] as u32;
                }
            }
            let mean = sum / (block_w * block_h);
            block_means.push(mean);
        }
    }

    let mut hash: u64 = 0;
    let mut bit_pos: u32 = 0;
    for by in 0..blocks_y {
        for bx in 0..(blocks_x - 1) {
            if bit_pos >= 64 {
                break;
            }
            let idx = (by * blocks_x + bx) as usize;
            let current = block_means[idx];
            let next = block_means[idx + 1];
            if next > current {
                hash |= 1u64 << bit_pos;
            }
            bit_pos += 1;
        }
    }
    hash
}

/// Double Gradient Hash（双梯度哈希）CPU 参考实现。
pub fn double_gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let mut hash_low: u32 = 0;
    let mut hash_high: u32 = 0;
    let mut h_bit_pos: u32 = 0;
    let mut v_bit_pos: u32 = 0;

    // 水平梯度
    for row in 0..height {
        for col in 0..(width - 1) {
            if h_bit_pos >= 32 {
                break;
            }
            let idx = (row * width + col) as usize;
            let left = pixels[idx];
            let right = pixels[idx + 1];
            if right > left {
                hash_low |= 1u32 << h_bit_pos;
            }
            h_bit_pos += 1;
        }
    }

    // 垂直梯度
    for col in 0..width {
        for row in 0..(height - 1) {
            if v_bit_pos >= 32 {
                break;
            }
            let idx = (row * width + col) as usize;
            let current = pixels[idx];
            let below = pixels[idx + width as usize];
            if below > current {
                hash_high |= 1u32 << v_bit_pos;
            }
            v_bit_pos += 1;
        }
    }

    (hash_low as u64) | ((hash_high as u64) << 32)
}
