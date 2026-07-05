/// 将 u8 灰度像素数组打包为 u32 数组（每像素 1 个 u32）。
///
/// 返回新分配的 `Vec<u32>`。如需零分配，请使用 [`pack_u8_to_u32_inplace`]。
/// 如果已有预分配的输出 buffer，优先使用 [`pack_u8_to_u32_into`] 或
/// [`pack_u8_batch_to_u32`] 以避免中间分配。
pub fn pack_u8_to_u32(pixels: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(pixels.len());
    pack_u8_to_u32_into(pixels, &mut out);
    out
}

/// 将 u8 灰度像素追加写入已有的 u32 输出 buffer。
///
/// 与 [`pack_u8_to_u32`] 相比，省去了创建临时 Vec 再 extend 的开销；
/// 调用方可提前 `with_capacity` 一次分配，避免多次 realloc。
pub fn pack_u8_to_u32_into(pixels: &[u8], output: &mut Vec<u32>) {
    output.reserve(pixels.len());
    // SAFETY: 读取 pixels 中的 u8 并写入 output 中的 u32，
    // 两段内存不重叠（output 是 reserve 后的未初始化空间）。
    // 这等价于 pixels.iter().map(|&p| p as u32).for_each(|v| output.push(v))，
    // 但通过 chunk 读取 + batch push 减少了循环开销。
    let src = pixels;
    let dst_start = output.len();
    output.resize(dst_start + src.len(), 0u32);
    let dst = &mut output[dst_start..];
    // 使用 chunks_exact 处理对齐的 u8 -> u32 转换，每次处理 8 个像素
    // （8 * u8 = 8 bytes，正好与 SIMD 对齐）
    const CHUNK: usize = 8;
    let chunks_exact = src.chunks_exact(CHUNK);
    let remainder = chunks_exact.remainder();
    let mut i = 0;
    for chunk in chunks_exact {
        // 手动展开 8 次以消除边界检查
        dst[i]     = chunk[0] as u32;
        dst[i + 1] = chunk[1] as u32;
        dst[i + 2] = chunk[2] as u32;
        dst[i + 3] = chunk[3] as u32;
        dst[i + 4] = chunk[4] as u32;
        dst[i + 5] = chunk[5] as u32;
        dst[i + 6] = chunk[6] as u32;
        dst[i + 7] = chunk[7] as u32;
        i += CHUNK;
    }
    for &p in remainder {
        dst[i] = p as u32;
        i += 1;
    }
}

/// 批量将多张 u8 图像打包为连续的 u32 数组。
///
/// 在整个 batch 上一次性预分配，避免每张图像单独分配临时 Vec。
/// 输出 buffer 中按 [image_0_pixels | image_1_pixels | ...] 布局。
pub fn pack_u8_batch_to_u32(images: &[Vec<u8>], output: &mut Vec<u32>) {
    let total: usize = images.iter().map(|img| img.len()).sum();
    output.reserve(total);
    for img in images {
        pack_u8_to_u32_into(img, output);
    }
}

/// 将 u8 灰度像素就地打包到预分配的 u32 缓冲区中（零堆分配）。
///
/// `output` 的长度必须 **≥ `pixels.len()`**，否则超出部分被静默截断。
pub fn pack_u8_to_u32_inplace(pixels: &[u8], output: &mut [u32]) {
    let n = pixels.len().min(output.len());
    // chunks_exact 消除边界检查 → 自动向量化友好
    let mut i = 0;
    for chunk in pixels[..n].chunks_exact(4) {
        output[i]     = chunk[0] as u32;
        output[i + 1] = chunk[1] as u32;
        output[i + 2] = chunk[2] as u32;
        output[i + 3] = chunk[3] as u32;
        i += 4;
    }
    for &p in &pixels[i..n] {
        output[i] = p as u32;
        i += 1;
    }
}

/// 将 u32 数组解包为 u8 灰度像素数组。
///
/// 返回新分配的 `Vec<u8>`。如需零分配，请使用 [`unpack_u32_to_u8_inplace`]。
pub fn unpack_u32_to_u8(data: &[u32], count: usize) -> Vec<u8> {
    debug_assert!(count <= data.len(), "unpack_u32_to_u8: count {} > data.len() {}", count, data.len());
    let safe_count = count.min(data.len());
    data[..safe_count].iter().map(|&v| v as u8).collect()
}

/// 将 u32 数组就地解包到预分配的 u8 缓冲区中（零堆分配）。
///
/// `output` 的长度必须 **≥ `count`**，否则写入 `output.len()` 个元素后停止。
pub fn unpack_u32_to_u8_inplace(data: &[u32], count: usize, output: &mut [u8]) {
    debug_assert!(count <= data.len(), "unpack_u32_to_u8_inplace: count {} > data.len() {}", count, data.len());
    let safe_count = count.min(data.len()).min(output.len());
    let mut i = 0;
    for chunk in data[..safe_count].chunks_exact(4) {
        output[i]     = chunk[0] as u8;
        output[i + 1] = chunk[1] as u8;
        output[i + 2] = chunk[2] as u8;
        output[i + 3] = chunk[3] as u8;
        i += 4;
    }
    for &v in &data[i..safe_count] {
        output[i] = v as u8;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_inplace_matches_allocating() {
        let pixels: Vec<u8> = (0..=255).collect();
        let expected = pack_u8_to_u32(&pixels);
        let mut buf = vec![0u32; pixels.len()];
        pack_u8_to_u32_inplace(&pixels, &mut buf);
        assert_eq!(buf, expected);
    }

    #[test]
    fn unpack_inplace_matches_allocating() {
        let data: Vec<u32> = (0..256).map(|x| x as u32 * 0x01010101).collect();
        let expected = unpack_u32_to_u8(&data, 256);
        let mut buf = vec![0u8; 256];
        unpack_u32_to_u8_inplace(&data, 256, &mut buf);
        assert_eq!(buf, expected);
    }

    #[test]
    fn pack_inplace_short_output() {
        let pixels: Vec<u8> = vec![10, 20, 30, 40, 50];
        let mut buf = vec![0u32; 3]; // 比 pixels 短
        pack_u8_to_u32_inplace(&pixels, &mut buf);
        assert_eq!(buf, vec![10, 20, 30]);
    }

    #[test]
    fn unpack_inplace_short_output() {
        let data: Vec<u32> = vec![100, 200, 300, 400, 500];
        let mut buf = vec![0u8; 3];
        unpack_u32_to_u8_inplace(&data, 5, &mut buf);
        // u32→u8 截断: 100→100, 200→200, 300→44 (300 % 256)
        assert_eq!(buf, vec![100, 200, 44]);
    }

    #[test]
    fn pack_inplace_empty() {
        let mut buf = vec![0u32; 0];
        pack_u8_to_u32_inplace(&[], &mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn unpack_inplace_empty() {
        let mut buf = vec![0u8; 0];
        unpack_u32_to_u8_inplace(&[], 0, &mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn pack_inplace_remainder() {
        // 测试 chunks_exact 不整除时的尾部处理
        let pixels: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7];
        let mut buf = vec![0u32; 7];
        pack_u8_to_u32_inplace(&pixels, &mut buf);
        assert_eq!(buf, vec![1, 2, 3, 4, 5, 6, 7]);
    }
}
