// u8→u32 像素打包着色器
//
// 在 GPU 端完成灰度 u8 像素到 u32 的打包（每像素 1 个 u32）。
// 替代 CPU 端 `pixel_pack::pack_u8_to_u32()`，减少 4× 上传数据量。
//
// 输入：原始 u8 灰度数据，reinterpret 为 u32 数组（每 4 字节 = 4 个灰度像素）
// 输出：u32 数组，每个 u32 含 1 个灰度像素值（0-255）
//
// 每线程处理 4 个像素（1 个输入 u32 → 4 个输出 u32）
// workgroup_size = 256 → 每 workgroup 处理 1024 像素
//
// 绑定布局：
//   0: raw_input     — 原始 u8 数据（storage, read）
//   1: packed_output — 打包后 u32 数据（storage, read_write）
//   2: pack_params   — 参数（uniform）

struct PackParams {
    pixel_count: u32,  // 总像素数
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> raw_input: array<u32>;
@group(0) @binding(1) var<storage, read_write> packed_output: array<u32>;
@group(0) @binding(2) var<uniform> pack_params: PackParams;

@compute @workgroup_size(256)
fn pack_u8_to_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    // 每线程处理 4 个像素
    let base_pixel = gid.x * 4u;
    if (base_pixel >= pack_params.pixel_count) {
        return;
    }

    // 读取 1 个 u32（含 4 个 u8 灰度像素）
    let word_idx = gid.x;
    let word = raw_input[word_idx];

    // 拆分 4 个 u8 像素
    let p0 = word & 0xFFu;
    let p1 = (word >> 8u) & 0xFFu;
    let p2 = (word >> 16u) & 0xFFu;
    let p3 = (word >> 24u) & 0xFFu;

    // 写入 4 个 u32
    if (base_pixel < pack_params.pixel_count) {
        packed_output[base_pixel] = p0;
    }
    if (base_pixel + 1u < pack_params.pixel_count) {
        packed_output[base_pixel + 1u] = p1;
    }
    if (base_pixel + 2u < pack_params.pixel_count) {
        packed_output[base_pixel + 2u] = p2;
    }
    if (base_pixel + 3u < pack_params.pixel_count) {
        packed_output[base_pixel + 3u] = p3;
    }
}
