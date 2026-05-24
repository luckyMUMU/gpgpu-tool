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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.x;
    let image_count = params.image_count;
    if (img_idx >= image_count) { return; }

    let width = params.width;
    let height = params.height;
    let hash_bits = params.hash_size_bits;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let base = img_idx * width * height;

    var hash_u32s: array<u32, 128>;
    for (var u = 0u; u < u32s_per_image; u = u + 1u) { hash_u32s[u] = 0u; }
    var bit_pos: u32 = 0u;

    let h_limit = hash_bits / 2u;
    for (var row = 0u; row < height; row = row + 1u) {
        for (var col = 0u; col < width - 1u; col = col + 1u) {
            if (bit_pos >= h_limit) { break; }
            let idx = row * width + col;
            let cur = pixels[base + idx];
            let nxt = pixels[base + idx + 1u];
            let bit = select(0u, 1u, nxt > cur);
            let u32_idx = bit_pos / 32u;
            hash_u32s[u32_idx] = hash_u32s[u32_idx] | (bit << (bit_pos % 32u));
            bit_pos = bit_pos + 1u;
        }
    }

    let v_start = h_limit;
    for (var col = 0u; col < width; col = col + 1u) {
        for (var row = 0u; row < height - 1u; row = row + 1u) {
            if (bit_pos >= hash_bits) { break; }
            let idx = row * width + col;
            let cur = pixels[base + idx];
            let blw = pixels[base + idx + width];
            let bit = select(0u, 1u, blw > cur);
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
