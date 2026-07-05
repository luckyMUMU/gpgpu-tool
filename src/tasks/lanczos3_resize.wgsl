// Lanczos3 Resize WGSL：2-pass 可分离卷积（带抗混叠 + f32 中间精度）
//
// Pass 1 (lanczos3_horizontal): src (in_w × in_h) → temp (out_w × in_h)
//   输入：u32 像素值 [0..255]
//   输出：f32 值（bitcast 为 u32 存储），保留振铃效应（可能 < 0 或 > 255）
//
// Pass 2 (lanczos3_vertical): temp (in_w × in_h) → dst (in_w × out_h)
//   输入：f32 值（bitcast 从 u32 读取）
//   输出：u32 像素值 [0..255]（最终 clamp）
//
// 关键：中间 buffer 不 clamp，保留 Lanczos3 振铃信息，仅在最终输出 clamp。
// 这与 image crate 的实现一致（中间结果用 f64，最终 clamp 为 u8）。
//
// 采样公式与 image crate Lanczos3 一致：
//   ratio = in_size / out_size
//   scale = max(1.0, ratio)
//   support = 3.0 * scale
//   src = (dst + 0.5) * ratio - 0.5
//   weight = lanczos3((i - src) / scale)
//   pixel = Σ(src_pixel * weight) / Σ(weight)

const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0)
var<storage, read> src_pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> dst_pixels: array<u32>;

struct Lanczos3ResizeParams {
    image_count: u32,
    in_w: u32,
    in_h: u32,
    out_w: u32,
    out_h: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

var<push_constant> params: Lanczos3ResizeParams;

/// 归一化 sinc 函数：sin(πx) / (πx)，sinc(0) = 1。
fn sinc(x: f32) -> f32 {
    if (abs(x) < 0.0001) {
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

/// Lanczos3 核函数：sinc(x) * sinc(x/3)，支撑域 |x| < 3。
fn lanczos3(x: f32) -> f32 {
    if (abs(x) >= 3.0) {
        return 0.0;
    }
    return sinc(x) * sinc(x / 3.0);
}

/// 水平 pass：src (in_w × in_h) → temp (out_w × in_h)
///
/// 输入：u32 像素值 [0..255]
/// 输出：f32 值（通过 bitcast 存储为 u32），不 clamp，保留振铃信息
///
/// gid.x = dst_x (输出宽度方向), gid.y = src_y (高度不变), gid.z = img_idx
@compute @workgroup_size(8, 8, 1)
fn lanczos3_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x;
    let dy = gid.y;
    let img_idx = gid.z;

    if (dx >= params.out_w || dy >= params.in_h || img_idx >= params.image_count) {
        return;
    }

    let src_base = img_idx * params.in_w * params.in_h;
    let dst_base = img_idx * params.out_w * params.in_h;

    // 缩放比率和抗混叠参数
    let ratio = f32(params.in_w) / f32(params.out_w);
    let scale = max(1.0, ratio);
    let support = 3.0 * scale;

    // 源坐标映射：与 image crate 一致
    let sx = (f32(dx) + 0.5) * ratio - 0.5;

    // 采样范围：覆盖所有 |i - sx| < support 的整数
    let left = i32(floor(sx - support));
    let right = i32(floor(sx + support));

    var sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var i = left; i <= right; i = i + 1) {
        let clamped_x = clamp(i, 0, i32(params.in_w) - 1);
        let pixel = f32(src_pixels[src_base + u32(clamped_x) + dy * params.in_w]);
        let weight = lanczos3((f32(i) - sx) / scale);
        sum = sum + pixel * weight;
        weight_sum = weight_sum + weight;
    }

    // 存储为 f32（bitcast 为 u32），不 clamp，保留振铃信息
    let result = sum / weight_sum;
    dst_pixels[dst_base + dy * params.out_w + dx] = bitcast<u32>(result);
}

/// 垂直 pass：temp (in_w × in_h) → dst (in_w × out_h)
///
/// 输入：f32 值（通过 bitcast 从 u32 读取）
/// 输出：u32 像素值 [0..255]（最终 clamp）
///
/// gid.x = src_x (宽度不变), gid.y = dst_y (输出高度方向), gid.z = img_idx
@compute @workgroup_size(8, 8, 1)
fn lanczos3_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x;
    let dy = gid.y;
    let img_idx = gid.z;

    if (dx >= params.in_w || dy >= params.out_h || img_idx >= params.image_count) {
        return;
    }

    let src_base = img_idx * params.in_w * params.in_h;
    let dst_base = img_idx * params.in_w * params.out_h;

    let ratio = f32(params.in_h) / f32(params.out_h);
    let scale = max(1.0, ratio);
    let support = 3.0 * scale;

    let sy = (f32(dy) + 0.5) * ratio - 0.5;

    let top = i32(floor(sy - support));
    let bottom = i32(floor(sy + support));

    var sum: f32 = 0.0;
    var weight_sum: f32 = 0.0;

    for (var i = top; i <= bottom; i = i + 1) {
        let clamped_y = clamp(i, 0, i32(params.in_h) - 1);
        // 从中间 buffer 读取 f32 值（bitcast 从 u32）
        let pixel = bitcast<f32>(src_pixels[src_base + u32(clamped_y) * params.in_w + dx]);
        let weight = lanczos3((f32(i) - sy) / scale);
        sum = sum + pixel * weight;
        weight_sum = weight_sum + weight;
    }

    // 最终输出 clamp 到 [0, 255]
    let result = sum / weight_sum;
    dst_pixels[dst_base + dy * params.in_w + dx] = u32(clamp(result, 0.0, 255.0));
}
