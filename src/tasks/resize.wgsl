// Resize WGSL：每目标像素一个线程，2D workgroup 布局
// gid.x = dx, gid.y = dy, gid.z = img_idx

@group(0) @binding(0)
var<storage, read> src_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> dst_pixels: array<u32>;

struct ResizeParams {
    image_count: u32,
    src_w: u32,
    src_h: u32,
    packed_dst: u32,
};

var<push_constant> params: ResizeParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_wh = params.packed_dst;
    let dst_w = dst_wh >> 16u;
    let dst_h = dst_wh & 0xFFFFu;

    let dx = gid.x;
    let dy = gid.y;
    let img_idx = gid.z;

    if (dx >= dst_w || dy >= dst_h || img_idx >= params.image_count) {
        return;
    }

    let src_pixels_per_image = params.src_w * params.src_h;
    let src_base = img_idx * src_pixels_per_image;
    let dst_base = img_idx * dst_w * dst_h;

    // 双线性插值：计算目标像素对应的源区域
    let x_ratio = f32(params.src_w) / f32(dst_w);
    let y_ratio = f32(params.src_h) / f32(dst_h);

    let x0f = f32(dx) * x_ratio;
    let x1f = f32(dx + 1u) * x_ratio;
    let x0 = u32(x0f);
    let x1 = min(u32(x1f), params.src_w);
    let x_count = max(x1 - x0, 1u);

    let y0f = f32(dy) * y_ratio;
    let y1f = f32(dy + 1u) * y_ratio;
    let y0 = u32(y0f);
    let y1 = min(u32(y1f), params.src_h);
    let y_count = max(y1 - y0, 1u);

    // 区域平均：累加源区域所有像素后取平均
    var sum: u32 = 0u;
    for (var sy = y0; sy < y1; sy = sy + 1u) {
        let row_base = src_base + sy * params.src_w;
        for (var sx = x0; sx < x1; sx = sx + 1u) {
            sum = sum + src_pixels[row_base + sx];
        }
    }

    let avg = sum / (x_count * y_count);
    let dst_idx = dst_base + dy * dst_w + dx;
    dst_pixels[dst_idx] = avg;
}
