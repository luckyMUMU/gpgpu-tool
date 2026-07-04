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
    let pixels_per_image = width * height;
    let base = img_idx * pixels_per_image;

    if (col >= width - 1u) { return; }

    let bit_pos = row * (width - 1u) + col;
    if (bit_pos >= hash_bits) { return; }

    let idx = row * width + col;
    let current = pixels[base + idx];
    let next = pixels[base + idx + 1u];
    let bit = select(0u, 1u, next > current);
    if (bit == 1u) {
        let u32_idx = bit_pos / 32u;
        let out_base = img_idx * u32s_per_image;
        atomicOr(&hashes[out_base + u32_idx], 1u << (bit_pos % 32u));
    }
}
