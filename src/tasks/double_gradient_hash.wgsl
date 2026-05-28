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

    let col = gid.x;
    let row = gid.y;
    let width = params.width;
    let height = params.height;
    if (col >= width || row >= height) { return; }

    let hash_bits = params.hash_size * params.hash_size;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let base = img_idx * width * height;
    let out_base = img_idx * u32s_per_image;
    let h_limit = hash_bits / 2u;

    if (col < width - 1u) {
        let bit_pos = row * (width - 1u) + col;
        if (bit_pos < h_limit) {
            let idx = row * width + col;
            let cur = pixels[base + idx];
            let nxt = pixels[base + idx + 1u];
            let bit = select(0u, 1u, nxt > cur);
            if (bit == 1u) {
                let u32_idx = bit_pos / 32u;
                atomicOr(&hashes[out_base + u32_idx], 1u << (bit_pos % 32u));
            }
        }
    }

    if (row < height - 1u) {
        let v_bit_pos = h_limit + col * (height - 1u) + row;
        if (v_bit_pos < hash_bits) {
            let idx = row * width + col;
            let cur = pixels[base + idx];
            let blw = pixels[base + idx + width];
            let bit = select(0u, 1u, blw > cur);
            if (bit == 1u) {
                let u32_idx = v_bit_pos / 32u;
                atomicOr(&hashes[out_base + u32_idx], 1u << (v_bit_pos % 32u));
            }
        }
    }
}
