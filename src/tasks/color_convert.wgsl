// RGBA→灰度 GPU 转换着色器
//
// 使用 czkawka 同款公式：(R*77 + G*150 + B*29) >> 8
// 保证 GPU 转换结果与 czkawka CPU 路径一致。
//
// 输入：RGBA 数据，每个 u32 包含一个像素（R=低位, G, B, A=高位）
// 输出：灰度数据，每个 u32 包含一个灰度像素值（0-255）
//
// 绑定布局：
//   0: input  — RGBA 像素数据（storage, read）
//   1: output — 灰度像素数据（storage, read_write）
//   2: params — 转换参数（uniform）

struct ColorConvertParams {
    pixel_count: u32,  // 总像素数（所有图像合计）
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: ColorConvertParams;

/// RGBA→灰度转换，使用 czkawka 兼容公式。
///
/// czkawka 使用整数运算 `(R*77 + G*150 + B*29) >> 8` 避免浮点精度差异，
/// 本着色器采用相同公式确保结果一致。
@compute @workgroup_size(64)
fn rgba_to_grayscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.pixel_count) {
        return;
    }

    let rgba = input[idx];
    let r = rgba & 0xFFu;
    let g = (rgba >> 8u) & 0xFFu;
    let b = (rgba >> 16u) & 0xFFu;
    // alpha = (rgba >> 24u) & 0xFFu; // 忽略 alpha 通道

    // czkawka 兼容灰度公式：整数运算，无浮点精度差异
    let gray = (r * 77u + g * 150u + b * 29u) >> 8u;

    output[idx] = gray;
}
