@group(0) @binding(0) var<storage, read> pixels: array<u32>;

@group(0) @binding(1) var<storage, read_write> hashes: array<u32>;

@group(0) @binding(2) var<uniform> params: vec4<u32>;

// 简化中值计算：使用计数排序思想（像素值范围 0-255）
fn compute_median(base: u32, pixel_count: u32) -> u32 {
    var histogram: array<u32, 256>;

    // 初始化直方图
    for (var i = 0u; i < 256u; i = i + 1u) {
        histogram[i] = 0u;
    }

    // 统计像素值频次
    for (var i = 0u; i < pixel_count; i = i + 1u) {
        let val = pixels[base + i];
        histogram[val] = histogram[val] + 1u;
    }

    // 查找中值
    let half = pixel_count / 2u;
    var count: u32 = 0u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        count = count + histogram[i];
        if (count > half) {
            return i;
        }
    }
    return 128u;
}

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

    // 计算中值
    let median = compute_median(base, pixels_per_image);

    // 生成 64bit 哈希（每个像素与中值比较）
    var hash_low: u32 = 0u;
    var hash_high: u32 = 0u;

    for (var i = 0u; i < 32u; i = i + 1u) {
        if (pixels[base + i] > median) {
            hash_low = hash_low | (1u << i);
        }
    }
    for (var i = 0u; i < 32u; i = i + 1u) {
        if (pixels[base + 32u + i] > median) {
            hash_high = hash_high | (1u << i);
        }
    }

    let out_base = img_idx * 2u;
    hashes[out_base + 0u] = hash_low;
    hashes[out_base + 1u] = hash_high;
}
