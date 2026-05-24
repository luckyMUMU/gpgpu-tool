/// CPU 参考实现：6 种感知哈希算法。
///
/// 与 WGSL 实现保持一致的算法逻辑和比较运算符。

/// Mean Hash（均值哈希）CPU 参考实现。
///
/// 使用 f32 均值（与 GPU WGSL 一致），像素 >= 均值生成 1bit。
pub fn mean_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    let count = pixels.len().min(64);
    let sum: f32 = pixels[..count].iter().map(|&p| p as f32).sum();
    let mean = sum / count as f32;

    let mut hash: u64 = 0;
    for (i, &pixel) in pixels[..count].iter().enumerate() {
        if pixel as f32 >= mean {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Median Hash（中值哈希）CPU 参考实现。
///
/// 使用直方图中值算法（与 GPU WGSL 一致），像素 > 中值生成 1bit。
pub fn median_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    let count = pixels.len().min(64);
    let mut histogram = [0u32; 256];
    for &pixel in &pixels[..count] {
        histogram[pixel as usize] += 1;
    }
    let half = count as u32 / 2;
    let mut cum = 0u32;
    let mut median = 128u32;
    for v in 0..256u32 {
        cum += histogram[v as usize];
        if cum > half {
            median = v;
            break;
        }
    }

    let mut hash: u64 = 0;
    for (i, &pixel) in pixels[..count].iter().enumerate() {
        if pixel as u32 > median {
            hash |= 1u64 << i;
        }
    }
    hash
}

/// Gradient Hash（水平梯度哈希）CPU 参考实现。
///
/// 每行相邻像素水平比较，差值 > 0 生成 1bit。
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
///
/// 每列相邻像素垂直比较，差值 > 0 生成 1bit。
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
///
/// 将图像分为 8x8 块，计算每块均值，相邻块均值比较生成 1bit。
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

    // 水平比较：8行 × 7比较 = 56 bit
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

    // 垂直比较：第0行与第1行 = 8 bit，补足 64 bit
    for bx in 0..blocks_x {
        if bit_pos >= 64 {
            break;
        }
        let idx = bx as usize;
        let current = block_means[idx];
        let below = block_means[idx + blocks_x as usize];
        if below > current {
            hash |= 1u64 << bit_pos;
        }
        bit_pos += 1;
    }

    hash
}

pub fn block_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let blocks_x = hash_size;
    let blocks_y = hash_size;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;
    let total_bits = hash_size * hash_size;
    let u32s_per_image = total_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);

    let mut block_means = Vec::with_capacity((blocks_x * blocks_y) as usize);
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

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    let mut bit_pos: u32 = 0;

    for by in 0..blocks_y {
        for bx in 0..(blocks_x - 1) {
            if bit_pos >= total_bits { break; }
            let idx = (by * blocks_x + bx) as usize;
            let current = block_means[idx];
            let next = block_means[idx + 1];
            if next > current {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    for by in 0..(blocks_y - 1) {
        for bx in 0..blocks_x {
            if bit_pos >= total_bits { break; }
            let idx = (by * blocks_x + bx) as usize;
            let current = block_means[idx];
            let below = block_means[idx + blocks_x as usize];
            if below > current {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
            bit_pos += 1;
        }
    }

    let mut result = Vec::with_capacity(u64s_per_image as usize);
    for u64_idx in 0..u64s_per_image {
        let lo = hash_u32s[u64_idx as usize * 2];
        let hi = if u64_idx as usize * 2 + 1 < u32s_per_image as usize {
            hash_u32s[u64_idx as usize * 2 + 1]
        } else {
            0u32
        };
        result.push((lo as u64) | ((hi as u64) << 32));
    }
    result
}

/// Double Gradient Hash（双梯度哈希）CPU 参考实现。
///
/// 水平梯度占低 32bit，垂直梯度占高 32bit。
pub fn double_gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    let mut hash_low: u32 = 0;
    let mut hash_high: u32 = 0;
    let mut h_bit_pos: u32 = 0;
    let mut v_bit_pos: u32 = 0;

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