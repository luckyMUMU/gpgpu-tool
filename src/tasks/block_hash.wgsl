@group(0) @binding(0) var<storage, read> pixels: array<u32>;

@group(0) @binding(1) var<storage, read_write> hashes: array<atomic<u32>>;

struct Params {
    image_count: u32,
    width: u32,
    height: u32,
    hash_size: u32,
};

var<push_constant> params: Params;

fn block_mean(img_idx: u32, bx: u32, by: u32, block_w: u32, block_h: u32, width: u32, pixels_per_image: u32) -> u32 {
    var sum: u32 = 0u;
    let base = img_idx * pixels_per_image;
    for (var dy = 0u; dy < block_h; dy = dy + 1u) {
        for (var dx = 0u; dx < block_w; dx = dx + 1u) {
            let px = bx * block_w + dx;
            let py = by * block_h + dy;
            sum = sum + pixels[base + py * width + px];
        }
    }
    return sum / (block_w * block_h);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.z;
    if (img_idx >= params.image_count) { return; }

    let bx = gid.x;
    let by = gid.y;
    let hash_size = params.hash_size;
    if (bx >= hash_size || by >= hash_size) { return; }

    let width = params.width;
    let height = params.height;
    let hash_bits = hash_size * hash_size;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let pixels_per_image = width * height;

    let blocks_x = hash_size;
    let blocks_y = hash_size;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;

    let out_base = img_idx * u32s_per_image;

    if (bx < blocks_x - 1u) {
        let m_left = block_mean(img_idx, bx, by, block_w, block_h, width, pixels_per_image);
        let m_right = block_mean(img_idx, bx + 1u, by, block_w, block_h, width, pixels_per_image);
        let bit_pos = by * (blocks_x - 1u) + bx;
        if (bit_pos < hash_bits) {
            let bit = select(0u, 1u, m_right > m_left);
            if (bit == 1u) {
                let u32_idx = bit_pos / 32u;
                atomicOr(&hashes[out_base + u32_idx], 1u << (bit_pos % 32u));
            }
        }
    }

    if (by < blocks_y - 1u) {
        let m_top = block_mean(img_idx, bx, by, block_w, block_h, width, pixels_per_image);
        let m_bottom = block_mean(img_idx, bx, by + 1u, block_w, block_h, width, pixels_per_image);
        let bit_pos = blocks_y * (blocks_x - 1u) + by * blocks_x + bx;
        if (bit_pos < hash_bits) {
            let bit = select(0u, 1u, m_bottom > m_top);
            if (bit == 1u) {
                let u32_idx = bit_pos / 32u;
                atomicOr(&hashes[out_base + u32_idx], 1u << (bit_pos % 32u));
            }
        }
    }
}
