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
    let pixels_per_image = width * height;
    let base = img_idx * pixels_per_image;
    let total_bits = min(hash_bits, pixels_per_image);

    var sum: f32 = 0.0;
    for (var i = 0u; i < pixels_per_image; i = i + 1u) {
        sum = sum + f32(pixels[base + i]);
    }
    let mean = sum / f32(pixels_per_image);

    var hash_u32s: array<u32, 128>;
    for (var u = 0u; u < u32s_per_image; u = u + 1u) { hash_u32s[u] = 0u; }

    for (var i = 0u; i < total_bits; i = i + 1u) {
        if (f32(pixels[base + i]) >= mean) {
            let u32_idx = i / 32u;
            let bit = i % 32u;
            hash_u32s[u32_idx] = hash_u32s[u32_idx] | (1u << bit);
        }
    }

    let out_base = img_idx * u32s_per_image;
    for (var u = 0u; u < u32s_per_image; u = u + 1u) {
        hashes[out_base + u] = hash_u32s[u];
    }
}
