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

    let dx = gid.x;
    let dy = gid.y;
    let width = params.width;
    let height = params.height;
    if (dx >= width || dy >= height) { return; }

    let hash_bits = params.hash_size * params.hash_size;
    let u32s_per_image = (hash_bits + 31u) / 32u;
    let pixels_per_image = width * height;
    let base = img_idx * pixels_per_image;
    let total_bits = min(hash_bits, pixels_per_image);

    var sum: f32 = 0.0;
    for (var i = 0u; i < pixels_per_image; i = i + 1u) {
        sum = sum + f32(pixels[base + i]);
    }
    let mean = sum / f32(pixels_per_image);

    let bit_pos = dy * width + dx;
    if (bit_pos >= total_bits) { return; }

    if (f32(pixels[base + bit_pos]) >= mean) {
        let u32_idx = bit_pos / 32u;
        let bit = bit_pos % 32u;
        let out_base = img_idx * u32s_per_image;
        atomicOr(&hashes[out_base + u32_idx], 1u << bit);
    }
}
