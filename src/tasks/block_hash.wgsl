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

    // 分块：始终分为 8x8 块
    let blocks_x = 8u;
    let blocks_y = 8u;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;

    // 计算每块均值
    var block_means: array<u32, 64>;
    for (var by = 0u; by < blocks_y; by = by + 1u) {
        for (var bx = 0u; bx < blocks_x; bx = bx + 1u) {
            var sum: u32 = 0u;
            for (var dy = 0u; dy < block_h; dy = dy + 1u) {
                for (var dx = 0u; dx < block_w; dx = dx + 1u) {
                    let px = bx * block_w + dx;
                    let py = by * block_h + dy;
                    let idx = py * width + px;
                    sum = sum + pixels[base + idx];
                }
            }
            let block_idx = by * blocks_x + bx;
            block_means[block_idx] = sum / (block_w * block_h);
        }
    }

    var hash_low: u32 = 0u;
    var hash_high: u32 = 0u;
    var bit_pos: u32 = 0u;

    // 水平比较：8行 × 7比较 = 56 bit
    for (var by = 0u; by < blocks_y; by = by + 1u) {
        for (var bx = 0u; bx < blocks_x - 1u; bx = bx + 1u) {
            if (bit_pos >= 64u) {
                break;
            }
            let idx = by * blocks_x + bx;
            let current = block_means[idx];
            let next = block_means[idx + 1u];
            let bit = select(0u, 1u, next > current);

            if (bit_pos < 32u) {
                hash_low = hash_low | (bit << bit_pos);
            } else {
                hash_high = hash_high | (bit << (bit_pos - 32u));
            }
            bit_pos = bit_pos + 1u;
        }
    }

    // 垂直比较：第0行与第1行 = 8 bit，补足 64 bit
    for (var bx = 0u; bx < blocks_x; bx = bx + 1u) {
        if (bit_pos >= 64u) {
            break;
        }
        let idx = 0u * blocks_x + bx;
        let current = block_means[idx];
        let below = block_means[idx + blocks_x];
        let bit = select(0u, 1u, below > current);

        if (bit_pos < 32u) {
            hash_low = hash_low | (bit << bit_pos);
        } else {
            hash_high = hash_high | (bit << (bit_pos - 32u));
        }
        bit_pos = bit_pos + 1u;
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}
