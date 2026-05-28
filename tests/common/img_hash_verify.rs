use img_hash::{HashAlg, HasherConfig, ImageHash};

/// 将灰度像素数据转换为 img_hash 使用的 DynamicImage。
///
/// 使用 img_hash 内部重新导出的 image 0.23 类型，
/// 避免与本项目 dev-dependencies 中的 image 0.25 产生类型冲突。
fn pixels_to_dynamic_image(pixels: &[u8], width: u32, height: u32) -> img_hash::image::DynamicImage {
    let gray_image: img_hash::image::GrayImage =
        img_hash::image::ImageBuffer::from_raw(width, height, pixels.to_vec())
            .expect("像素数据与尺寸不匹配");
    img_hash::image::DynamicImage::ImageLuma8(gray_image)
}

/// 从 ImageHash 提取 bit 数据并转换为 u64。
///
/// img_hash 内部 BoolsToBytes 以 LSB-first 存储每个 bit，
/// 与本项目 GPU 实现的 bit 排列方式一致，可直接按小端序拼接。
fn hash_to_u64(hash: &ImageHash<Box<[u8]>>) -> u64 {
    let bytes = hash.as_bytes();
    let mut result: u64 = 0;
    for (i, &byte) in bytes.iter().enumerate().take(8) {
        result |= (byte as u64) << (i * 8);
    }
    result
}

/// 从 ImageHash 提取前 n_bits 个 bit 并转换为 u64。
///
/// 用于 Blockhash 校验时仅对比水平比较部分的 bit。
fn hash_to_u64_n_bits(hash: &ImageHash<Box<[u8]>>, n_bits: usize) -> u64 {
    let bytes = hash.as_bytes();
    let mut result: u64 = 0;
    for bit_pos in 0..n_bits.min(64) {
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        if byte_idx < bytes.len() && (bytes[byte_idx] >> bit_idx) & 1 == 1 {
            result |= 1u64 << bit_pos;
        }
    }
    result
}

/// 校验 Mean Hash。
///
/// 注意：img_hash 使用整数均值（u32 除法截断），本项目 GPU 使用 f32 均值，
/// 两者在边界情况下可能产生不同结果。
pub fn verify_mean_hash(pixels: &[u8], width: u32, height: u32, gpu_hash: u64) -> bool {
    let image = pixels_to_dynamic_image(pixels, width, height);
    let hasher = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(HashAlg::Mean)
        .to_hasher();
    let hash = hasher.hash_image(&image);
    hash_to_u64(&hash) == gpu_hash
}

/// 校验 Gradient Hash。
///
/// img_hash 内部自动将图像缩放到 (hash_width+1) × hash_height，
/// 然后按行做水平相邻像素比较（右 > 左），与本项目比较方向一致。
pub fn verify_gradient_hash(pixels: &[u8], width: u32, height: u32, gpu_hash: u64) -> bool {
    let image = pixels_to_dynamic_image(pixels, width, height);
    let hasher = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(HashAlg::Gradient)
        .to_hasher();
    let hash = hasher.hash_image(&image);
    hash_to_u64(&hash) == gpu_hash
}

/// 校验 VertGradient Hash。
///
/// img_hash 内部自动将图像缩放到 hash_width × (hash_height+1)，
/// 然后按列做垂直相邻像素比较（下 > 上），与本项目比较方向一致。
pub fn verify_vert_gradient_hash(pixels: &[u8], width: u32, height: u32, gpu_hash: u64) -> bool {
    let image = pixels_to_dynamic_image(pixels, width, height);
    let hasher = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(HashAlg::VertGradient)
        .to_hasher();
    let hash = hasher.hash_image(&image);
    hash_to_u64(&hash) == gpu_hash
}

/// 校验 DoubleGradient Hash。
///
/// img_hash 先做水平梯度比较再接垂直梯度比较，
/// 与本项目低 32 bit 水平 + 高 32 bit 垂直的排列一致。
pub fn verify_double_gradient_hash(pixels: &[u8], width: u32, height: u32, gpu_hash: u64) -> bool {
    let image = pixels_to_dynamic_image(pixels, width, height);
    let hasher = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(HashAlg::DoubleGradient)
        .to_hasher();
    let hash = hasher.hash_image(&image);
    hash_to_u64(&hash) == gpu_hash
}

/// 校验 Block Hash（仅水平比较部分）。
///
/// img_hash 的 Blockhash 算法与本项目实现存在差异：
/// - img_hash：按行分组，每组计算中值后逐块与中值比较
/// - 本项目：相邻块均值比较（水平 56 bit + 垂直 8 bit）
/// 此函数仅对比水平比较部分的前 56 bit。
pub fn verify_block_hash_horizontal(pixels: &[u8], width: u32, height: u32, gpu_hash: u64) -> bool {
    let image = pixels_to_dynamic_image(pixels, width, height);
    let hasher = HasherConfig::new()
        .hash_size(8, 8)
        .hash_alg(HashAlg::Blockhash)
        .to_hasher();
    let hash = hasher.hash_image(&image);
    let ref_hash = hash_to_u64_n_bits(&hash, 56);
    let gpu_hash_horizontal = gpu_hash & ((1u64 << 56) - 1);
    ref_hash == gpu_hash_horizontal
}
