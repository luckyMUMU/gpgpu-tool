/// 将 u8 灰度像素数组打包为 u32 数组（每像素 1 个 u32）。
pub fn pack_u8_to_u32(pixels: &[u8]) -> Vec<u32> {
    pixels.iter().map(|&p| p as u32).collect()
}

/// 将 u32 数组解包为 u8 灰度像素数组。
pub fn unpack_u32_to_u8(data: &[u32], count: usize) -> Vec<u8> {
    data[..count].iter().map(|&v| v as u8).collect()
}
