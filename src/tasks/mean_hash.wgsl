@group(0) @binding(0) var<storage, read> pixels: array<u32>;

@group(0) @binding(1) var<storage, read_write> hashes: array<u32>;

@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.x;
    let image_count = params.x;

    if (img_idx >= image_count) {
        return;
    }

    let width = params.y;
    let height = params.z;
    let pixels_per_image = width * height;
    let base = img_idx * pixels_per_image;

    // 使用 f32 计算均值，避免整数截断导致边界像素归类不一致
    var sum: f32 = 0.0;
    for (var i = 0u; i < pixels_per_image; i = i + 1u) {
        sum = sum + f32(pixels[base + i]);
    }
    let mean = sum / f32(pixels_per_image);

    // 生成 64bit 哈希（像素 >= 均值生成 1bit）
    var hash_low: u32 = 0u;
    var hash_high: u32 = 0u;

    for (var i = 0u; i < 32u; i = i + 1u) {
        if (f32(pixels[base + i]) >= mean) {
            hash_low = hash_low | (1u << i);
        }
    }
    for (var i = 0u; i < 32u; i = i + 1u) {
        if (f32(pixels[base + 32u + i]) >= mean) {
            hash_high = hash_high | (1u << i);
        }
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}