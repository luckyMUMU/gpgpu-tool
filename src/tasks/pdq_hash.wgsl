// PDQ DCT-II 着色器
// 可分离两趟 DCT：水平 1D-DCT + 垂直 1D-DCT
// 水平模式：输入为 u32 像素值，输出为 f32 DCT 系数（bitcast 为 u32 存储）
// 垂直模式：输入为 f32 中间结果（bitcast 为 u32 存储），输出为 f32 DCT 系数

struct DctParams {
    width: u32,
    height: u32,
    dct_size: u32,
    pass_mode: u32,
    image_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: DctParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.image_count * params.width * params.height;
    if (gid.x >= total) { return; }

    let pixels_per_image = params.width * params.height;
    let img_idx = gid.x / pixels_per_image;
    let pixel_in_img = gid.x % pixels_per_image;
    let img_offset = img_idx * pixels_per_image;

    let N = params.dct_size;
    let PI = 3.14159265358979323846;

    if (params.pass_mode == 0u) {
        // 水平 1D-DCT：对每行做 DCT
        let row = pixel_in_img / params.width;
        let k = pixel_in_img % params.width;
        let row_offset = img_offset + row * params.width;

        var sum: f32 = 0.0;
        for (var n = 0u; n < N; n = n + 1u) {
            let pixel_val = f32(input[row_offset + n]);
            let angle = PI * f32(2u * n + 1u) * f32(k) / f32(2u * N);
            sum = sum + pixel_val * cos(angle);
        }
        output[gid.x] = bitcast<u32>(sum);
    } else {
        // 垂直 1D-DCT：对每列做 DCT
        let col = pixel_in_img % params.width;
        let k = pixel_in_img / params.width;

        var sum: f32 = 0.0;
        for (var n = 0u; n < N; n = n + 1u) {
            let val = bitcast<f32>(input[img_offset + n * params.width + col]);
            let angle = PI * f32(2u * n + 1u) * f32(k) / f32(2u * N);
            sum = sum + val * cos(angle);
        }
        output[gid.x] = bitcast<u32>(sum);
    }
}
