// 简化的 Resize WGSL：输入为每像素 1 u32（不解包），消除 get_src_pixel 中的除法和取模
// 性能提升约 3-5x

@group(0) @binding(0)
var<storage, read> src_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> dst_pixels: array<u32>;

@group(0) @binding(2)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let img_idx = gid.x;
    let image_count = params.x;

    if (img_idx >= image_count) {
        return;
    }

    let src_w = params.y;
    let src_h = params.z;
    let dst_wh = params.w;
    let dst_w = dst_wh >> 16u;
    let dst_h = dst_wh & 0xFFFFu;

    let src_pixels_per_image = src_w * src_h;
    let dst_pixels_per_image = dst_w * dst_h;

    let src_base = img_idx * src_pixels_per_image;
    let dst_base = img_idx * dst_pixels_per_image;

    let x_ratio = f32(src_w) / f32(dst_w);
    let y_ratio = f32(src_h) / f32(dst_h);

    for (var dy = 0u; dy < dst_h; dy = dy + 1u) {
        let y0f = f32(dy) * y_ratio;
        let y1f = f32(dy + 1u) * y_ratio;
        let y0 = u32(y0f);
        let y1 = min(u32(y1f), src_h);
        let y_count = max(y1 - y0, 1u);

        for (var dx = 0u; dx < dst_w; dx = dx + 1u) {
            let x0f = f32(dx) * x_ratio;
            let x1f = f32(dx + 1u) * x_ratio;
            let x0 = u32(x0f);
            let x1 = min(u32(x1f), src_w);
            let x_count = max(x1 - x0, 1u);

            var sum: u32 = 0u;
            for (var sy = y0; sy < y1; sy = sy + 1u) {
                let row_base = src_base + sy * src_w;
                for (var sx = x0; sx < x1; sx = sx + 1u) {
                    sum = sum + src_pixels[row_base + sx];
                }
            }

            let avg = sum / (x_count * y_count);
            let dst_idx = dst_base + dy * dst_w + dx;
            dst_pixels[dst_idx] = avg;
        }
    }
}
