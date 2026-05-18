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

    // 分块：16x16 图像分为 8x8 个 2x2 块
    let block_size = 2u;
    let blocks_x = width / block_size;
    let blocks_y = height / block_size;

    // 计算每块均值
    var block_means: array<u32, 64>;
    for (var by = 0u; by < blocks_y; by = by + 1u) {
        for (var bx = 0u; bx < blocks_x; bx = bx + 1u) {
            var sum: u32 = 0u;
            for (var dy = 0u; dy < block_size; dy = dy + 1u) {
                for (var dx = 0u; dx < block_size; dx = dx + 1u) {
                    let px = bx * block_size + dx;
                    let py = by * block_size + dy;
                    let idx = py * width + px;
                    sum = sum + pixels[base + idx];
                }
            }
            let block_idx = by * blocks_x + bx;
            block_means[block_idx] = sum / (block_size * block_size);
        }
    }

    // 相邻块均值比较生成 64bit 哈希
    var hash_low: u32 = 0u;
    var hash_high: u32 = 0u;

    for (var by = 0u; by < blocks_y; by = by + 1u) {
        for (var bx = 0u; bx < blocks_x - 1u; bx = bx + 1u) {
            let idx = by * blocks_x + bx;
            let current = block_means[idx];
            let next = block_means[idx + 1u];
            let bit = select(0u, 1u, next > current);

            if (idx < 32u) {
                hash_low = hash_low | (bit << idx);
            } else {
                hash_high = hash_high | (bit << (idx - 32u));
            }
        }
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}
