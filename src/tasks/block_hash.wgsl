@group(0) @binding(0) var<storage, read> pixels: array<u32>;

@group(0) @binding(1) var<storage, read_write> hashes: array<u32>;

struct Params {
    image_count: u32,
    width: u32,
    height: u32,
    hash_size_bits: u32,
    hash_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(2) var<uniform> params: Params;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.x;
    if (img_idx >= params.image_count) { return; }

    let width = params.width;
    let height = params.height;
    let hash_bits = params.hash_size_bits;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let pixels_per_image = width * height;

    let blocks_x = params.hash_size;
    let blocks_y = params.hash_size;
    let block_w = width / blocks_x;
    let block_h = height / blocks_y;

    var hash_u32s: array<u32, 128>;
    for (var u = 0u; u < u32s_per_image; u = u + 1u) { hash_u32s[u] = 0u; }
    var bit_pos: u32 = 0u;

    for (var by = 0u; by < blocks_y; by = by + 1u) {
        if (bit_pos >= hash_bits) { break; }
        for (var bx = 0u; bx < blocks_x - 1u; bx = bx + 1u) {
            if (bit_pos >= hash_bits) { break; }
            let m_left = block_mean(img_idx, bx, by, block_w, block_h, width, pixels_per_image);
            let m_right = block_mean(img_idx, bx + 1u, by, block_w, block_h, width, pixels_per_image);
            let bit = select(0u, 1u, m_right > m_left);
            let u32_idx = bit_pos / 32u;
            hash_u32s[u32_idx] = hash_u32s[u32_idx] | (bit << (bit_pos % 32u));
            bit_pos = bit_pos + 1u;
        }
    }

    for (var by = 0u; by < blocks_y - 1u; by = by + 1u) {
        if (bit_pos >= hash_bits) { break; }
        for (var bx = 0u; bx < blocks_x; bx = bx + 1u) {
            if (bit_pos >= hash_bits) { break; }
            let m_top = block_mean(img_idx, bx, by, block_w, block_h, width, pixels_per_image);
            let m_bottom = block_mean(img_idx, bx, by + 1u, block_w, block_h, width, pixels_per_image);
            let bit = select(0u, 1u, m_bottom > m_top);
            let u32_idx = bit_pos / 32u;
            hash_u32s[u32_idx] = hash_u32s[u32_idx] | (bit << (bit_pos % 32u));
            bit_pos = bit_pos + 1u;
        }
    }

    let out_base = img_idx * u32s_per_image;
    for (var u = 0u; u < u32s_per_image; u = u + 1u) {
        hashes[out_base + u] = hash_u32s[u];
    }
}
