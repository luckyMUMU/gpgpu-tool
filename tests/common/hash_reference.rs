/// CPU 参考实现：6 种感知哈希算法。
///
/// 与 WGSL 实现保持一致的算法逻辑和比较运算符。
///
/// Mean Hash（均值哈希）CPU 参考实现。
///
/// 使用 f32 均值（与 GPU WGSL 一致），像素 >= 均值生成 1bit。
pub fn mean_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    mean_hash_with_size(pixels, _width, _height, 8)[0]
}

/// Mean Hash 支持任意 hash_size 的 CPU 参考实现，返回 Vec<u64>。
///
/// 与 GPU WGSL 着色器逻辑一致：遍历 width*height 个像素计算均值，
/// 每个像素与均值比较生成 1bit，bit_pos = dy * width + dx。
pub fn mean_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let pixel_count = (width * height) as usize;
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = hash_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);
    let total_bits = hash_bits.min(pixel_count as u32);

    let sum: f32 = pixels[..pixel_count].iter().map(|&p| p as f32).sum();
    let mean = sum / pixel_count as f32;

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    for dy in 0..height {
        for dx in 0..width {
            let bit_pos = dy * width + dx;
            if bit_pos >= total_bits { break; }
            let pixel = pixels[bit_pos as usize] as f32;
            if pixel >= mean {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
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

/// Median Hash（中值哈希）CPU 参考实现。
///
/// 使用直方图中值算法（与 GPU WGSL 一致），像素 > 中值生成 1bit。
pub fn median_hash(pixels: &[u8], _width: u32, _height: u32) -> u64 {
    median_hash_with_size(pixels, _width, _height, 8)[0]
}

/// Median Hash 支持任意 hash_size 的 CPU 参考实现，返回 Vec<u64>。
///
/// 与 GPU WGSL 着色器逻辑一致：遍历 width*height 个像素计算直方图中值，
/// 每个像素与中值比较生成 1bit，bit_pos = dy * width + dx。
pub fn median_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let pixel_count = (width * height) as usize;
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = hash_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);
    let total_bits = hash_bits.min(pixel_count as u32);

    let mut histogram = [0u32; 256];
    for &pixel in &pixels[..pixel_count] {
        histogram[pixel as usize] += 1;
    }
    let half = pixel_count as u32 / 2;
    let mut cum = 0u32;
    let mut median = 128u32;
    for v in 0..256u32 {
        cum += histogram[v as usize];
        if cum > half {
            median = v;
            break;
        }
    }

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    for dy in 0..height {
        for dx in 0..width {
            let bit_pos = dy * width + dx;
            if bit_pos >= total_bits { break; }
            if pixels[bit_pos as usize] as u32 > median {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
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

/// Gradient Hash（水平梯度哈希）CPU 参考实现。
///
/// 每行相邻像素水平比较，差值 > 0 生成 1bit。
pub fn gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    gradient_hash_with_size(pixels, width, height, 8)[0]
}

/// Gradient Hash 支持任意 hash_size 的 CPU 参考实现，返回 Vec<u64>。
///
/// 与 GPU WGSL 着色器逻辑一致：每行相邻像素水平比较，
/// bit_pos = row * (width - 1) + col，差值 > 0 生成 1bit。
pub fn gradient_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = hash_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    for row in 0..height {
        for col in 0..(width - 1) {
            let bit_pos = row * (width - 1) + col;
            if bit_pos >= hash_bits { break; }
            let idx = (row * width + col) as usize;
            let current = pixels[idx] as u32;
            let next = pixels[idx + 1] as u32;
            if next > current {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
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

/// Vertical Gradient Hash（垂直梯度哈希）CPU 参考实现。
///
/// 每列相邻像素垂直比较，差值 > 0 生成 1bit。
pub fn vert_gradient_hash(pixels: &[u8], width: u32, height: u32) -> u64 {
    vert_gradient_hash_with_size(pixels, width, height, 8)[0]
}

/// Vertical Gradient Hash 支持任意 hash_size 的 CPU 参考实现，返回 Vec<u64>。
///
/// 与 GPU WGSL 着色器逻辑一致：每列相邻像素垂直比较，
/// bit_pos = col * (height - 1) + row，差值 > 0 生成 1bit。
pub fn vert_gradient_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = hash_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    for col in 0..width {
        for row in 0..(height - 1) {
            let bit_pos = col * (height - 1) + row;
            if bit_pos >= hash_bits { break; }
            let idx = (row * width + col) as usize;
            let current = pixels[idx] as u32;
            let below = pixels[idx + width as usize] as u32;
            if below > current {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
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
    double_gradient_hash_with_size(pixels, width, height, 8)[0]
}

/// Double Gradient Hash 支持任意 hash_size 的 CPU 参考实现，返回 Vec<u64>。
///
/// 与 GPU WGSL 着色器逻辑一致：水平梯度占前半 bit，垂直梯度占后半 bit。
/// 水平：bit_pos = row * (width - 1) + col
/// 垂直：v_bit_pos = h_limit + col * (height - 1) + row
pub fn double_gradient_hash_with_size(pixels: &[u8], width: u32, height: u32, hash_size: u32) -> Vec<u64> {
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = hash_bits.div_ceil(32);
    let u64s_per_image = u32s_per_image.div_ceil(2);
    let h_limit = hash_bits / 2;

    let mut hash_u32s = vec![0u32; u32s_per_image as usize];
    for row in 0..height {
        for col in 0..(width - 1) {
            let bit_pos = row * (width - 1) + col;
            if bit_pos >= h_limit { break; }
            let idx = (row * width + col) as usize;
            let cur = pixels[idx] as u32;
            let nxt = pixels[idx + 1] as u32;
            if nxt > cur {
                let u32_idx = (bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (bit_pos % 32);
            }
        }
    }

    for col in 0..width {
        for row in 0..(height - 1) {
            let v_bit_pos = h_limit + col * (height - 1) + row;
            if v_bit_pos >= hash_bits { break; }
            let idx = (row * width + col) as usize;
            let cur = pixels[idx] as u32;
            let blw = pixels[idx + width as usize] as u32;
            if blw > cur {
                let u32_idx = (v_bit_pos / 32) as usize;
                hash_u32s[u32_idx] |= 1u32 << (v_bit_pos % 32);
            }
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