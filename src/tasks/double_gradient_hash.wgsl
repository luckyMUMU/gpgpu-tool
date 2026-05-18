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

    // 双梯度：同时计算水平和垂直梯度，各取 32bit 组合为 64bit
    var hash_low: u32 = 0u;  // 水平梯度 32bit
    var hash_high: u32 = 0u; // 垂直梯度 32bit

    // 水平梯度（每行前 width-1 个比较）
    for (var row = 0u; row < height; row = row + 1u) {
        for (var col = 0u; col < width - 1u; col = col + 1u) {
            let idx = row * width + col;
            let left = pixels[base + idx];
            let right = pixels[base + idx + 1u];
            let bit = select(0u, 1u, right > left);

            let bit_idx = row * (width - 1u) + col;
            if (bit_idx < 32u) {
                hash_low = hash_low | (bit << bit_idx);
            }
        }
    }

    // 垂直梯度（每列前 height-1 个比较）
    for (var col = 0u; col < width; col = col + 1u) {
        for (var row = 0u; row < height - 1u; row = row + 1u) {
            let idx = row * width + col;
            let current = pixels[base + idx];
            let below = pixels[base + idx + width];
            let bit = select(0u, 1u, below > current);

            let bit_idx = col * (height - 1u) + row;
            if (bit_idx < 32u) {
                hash_high = hash_high | (bit << bit_idx);
            }
        }
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}
