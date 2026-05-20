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

    // 垂直梯度：每列相邻像素比较，使用 bit_pos 顺序计数
    var hash_low: u32 = 0u;
    var hash_high: u32 = 0u;
    var bit_pos: u32 = 0u;

    for (var col = 0u; col < width; col = col + 1u) {
        for (var row = 0u; row < height - 1u; row = row + 1u) {
            if (bit_pos >= 64u) {
                break;
            }
            let idx = row * width + col;
            let current = pixels[base + idx];
            let below = pixels[base + idx + width];
            let bit = select(0u, 1u, below > current);

            if (bit_pos < 32u) {
                hash_low = hash_low | (bit << bit_pos);
            } else {
                hash_high = hash_high | (bit << (bit_pos - 32u));
            }
            bit_pos = bit_pos + 1u;
        }
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}