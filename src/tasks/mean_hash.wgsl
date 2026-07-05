// Mean Hash 着色器：从输入缓冲区读取预计算的均值，与像素比较设置哈希位。
//
// 输入缓冲区布局（每张图）：[pixels_per_image 个 u32 像素, 1 个 u32 均值]
// 步长 stride = pixels_per_image + 1，均值位于 pixels[base + pixels_per_image]。
//
// 均值由 Rust 端预计算后追加到像素数据末尾，避免 GPU 端 O(N²) 循环
// （原方案每线程遍历全部像素计算均值，hash_size=32 时 DX12 循环展开导致
// register pressure 过大，着色器编译静默失败）。
@group(0) @binding(0) var<storage, read> pixels: array<u32>;

@group(0) @binding(1) var<storage, read_write> hashes: array<atomic<u32>>;

struct Params {
    image_count: u32,
    width: u32,
    height: u32,
    hash_size: u32,
};

var<push_constant> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.z;
    if (img_idx >= params.image_count) { return; }

    let dx = gid.x;
    let dy = gid.y;
    let width = params.width;
    let height = params.height;
    if (dx >= width || dy >= height) { return; }

    let hash_bits = params.hash_size * params.hash_size;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let pixels_per_image = width * height;
    // 每张图步长 = 像素数 + 1（末尾追加预计算的均值）
    let stride = pixels_per_image + 1u;
    let base = img_idx * stride;
    let total_bits = min(hash_bits, pixels_per_image);

    // 从输入缓冲区末尾读取预计算的均值（f32 位模式存储为 u32）
    let mean = bitcast<f32>(pixels[base + pixels_per_image]);

    let bit_pos = dy * width + dx;
    if (bit_pos >= total_bits) { return; }

    if (f32(pixels[base + bit_pos]) >= mean) {
        let u32_idx = bit_pos / 32u;
        let bit = bit_pos % 32u;
        let out_base = img_idx * u32s_per_image;
        atomicOr(&hashes[out_base + u32_idx], 1u << bit);
    }
}
