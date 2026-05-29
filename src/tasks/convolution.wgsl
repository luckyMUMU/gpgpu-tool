// GPU 2D 卷积着色器
// 支持三种边界模式（Zero/Clamp/Reflect）和三种通道模式（2D/水平1D/垂直1D）
// 2D workgroup 布局：gid.x = x, gid.y = y
// separable_fused 入口点使用 LDS 共享内存优化可分离卷积

struct ConvParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    kernel_radius: u32,
    border_mode: u32,   // 0=Zero, 1=Clamp, 2=Reflect
    pass_mode: u32,     // 0=Full2D, 1=Horizontal1D, 2=Vertical1D, 3=FusedSeparable
    _pad1: u32,
    _pad2: u32,
    // 卷积核数据，最大支持 11×11=121 个 f32，填充至 124 对齐到 vec4 边界
    kernel: array<f32, 124>,
};

@group(0) @binding(0)
var<storage, read> pixels: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@group(0) @binding(2)
var<storage, read> params: ConvParams;

// LDS 共享内存：存储水平卷积中间结果（含垂直 pass 所需的 halo 行）
// LDS 大小 = (WG_Y + 2 * MAX_RADIUS) * WG_X = (8 + 2*5) * 8 = 144
// 如果修改 MAX_KERNEL_SIZE 或 workgroup_size，需同步更新此值及 Rust 端编译时断言
// 布局：(WG_Y + 2 * MAX_RADIUS) 行 × WG_X 列 = 18 × 8 = 144 个 f32
// MAX_RADIUS = 5（对应最大 kernel_size = 11）
// 行 0..r-1: 顶部 halo
// 行 r..r+7: 中心区域
// 行 r+8..r+8+r-1: 底部 halo
var<workgroup> lds_h: array<f32, 144>;

// 半样本对称反射坐标映射
fn reflect_coord(c: i32, size: i32) -> i32 {
    if (size == 1) { return 0; }
    var result = c;
    let period = 2 * size;
    result = ((result % period) + period) % period;
    if (result >= size) {
        result = 2 * size - 1 - result;
    }
    return result;
}

// 读取像素值：垂直 1D 读取水平 pass 的 f32 中间结果，其余模式读取 u32 像素值
fn read_pixel_value(idx: u32) -> f32 {
    if (params.pass_mode == 2u) {
        return bitcast<f32>(pixels[idx]);
    }
    return f32(pixels[idx]);
}

// 根据边界模式获取像素值
fn get_pixel(x: i32, y: i32, border_mode: u32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    if (border_mode == 0u) {
        if (x < 0 || x >= w || y < 0 || y >= h) {
            return 0.0;
        }
        return read_pixel_value(u32(y) * params.width + u32(x));
    }

    var fx = x;
    var fy = y;

    if (border_mode == 1u) {
        fx = clamp(x, 0, w - 1);
        fy = clamp(y, 0, h - 1);
    } else if (border_mode == 2u) {
        fx = reflect_coord(x, w);
        fy = reflect_coord(y, h);
    }

    return read_pixel_value(u32(fy) * params.width + u32(fx));
}

// 计算单个像素的水平 1D 卷积
fn compute_horizontal_conv(global_x: i32, global_y: i32, border_mode: u32) -> f32 {
    var sum: f32 = 0.0;
    let ks = i32(params.kernel_size);
    let r = i32(params.kernel_radius);
    for (var kx = 0; kx < ks; kx = kx + 1) {
        let sx = global_x + kx - r;
        let pixel_val = get_pixel(sx, global_y, border_mode);
        let weight = params.kernel[u32(kx)];
        sum = sum + pixel_val * weight;
    }
    return sum;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = i32(gid.x);
    let y = i32(gid.y);

    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    let pixel_idx = gid.y * params.width + gid.x;

    let pass_mode = params.pass_mode;
    let border_mode = params.border_mode;
    let kernel_size = i32(params.kernel_size);
    let kernel_radius = i32(params.kernel_radius);

    var sum: f32 = 0.0;

    if (pass_mode == 0u) {
        // 完整 2D 卷积
        for (var ky = 0; ky < kernel_size; ky = ky + 1) {
            for (var kx = 0; kx < kernel_size; kx = kx + 1) {
                let sx = x + kx - kernel_radius;
                let sy = y + ky - kernel_radius;
                let pixel_val = get_pixel(sx, sy, border_mode);
                let weight = params.kernel[u32(ky) * params.kernel_size + u32(kx)];
                sum = sum + pixel_val * weight;
            }
        }
    } else if (pass_mode == 1u) {
        // 水平 1D 卷积
        for (var kx = 0; kx < kernel_size; kx = kx + 1) {
            let sx = x + kx - kernel_radius;
            let pixel_val = get_pixel(sx, y, border_mode);
            let weight = params.kernel[u32(kx)];
            sum = sum + pixel_val * weight;
        }
    } else if (pass_mode == 2u) {
        // 垂直 1D 卷积
        for (var ky = 0; ky < kernel_size; ky = ky + 1) {
            let sy = y + ky - kernel_radius;
            let pixel_val = get_pixel(x, sy, border_mode);
            let weight = params.kernel[u32(ky)];
            sum = sum + pixel_val * weight;
        }
    }

    if (pass_mode == 1u) {
        // 水平 1D 卷积：bitcast 保留完整 f32 精度作为中间结果
        output[pixel_idx] = bitcast<u32>(sum);
    } else {
        // 完整 2D 或垂直 1D 卷积：钳制到 [0, 255] 并写入输出
        let result = u32(clamp(sum, 0.0, 255.0));
        output[pixel_idx] = result;
    }
}

// LDS 优化的融合可分离卷积：水平 + 垂直两趟在单次 dispatch 中完成
// 水平 pass 结果存入 LDS，垂直 pass 从 LDS 读取，避免一次全局内存往返
@compute @workgroup_size(8, 8, 1)
fn separable_fused(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let local_x = i32(lid.x);
    let local_y = i32(lid.y);
    let global_x = i32(gid.x);
    let global_y = i32(gid.y);

    let r = i32(params.kernel_radius);
    let lds_stride = 8u;
    let local_idx = u32(local_y) * 8u + u32(local_x);
    let border_mode = params.border_mode;

    // ===== 阶段 1: 计算中心区域的水平卷积，存入 LDS =====
    // 所有线程都参与计算（含越界线程），确保 LDS 数据完整
    let h_sum = compute_horizontal_conv(global_x, global_y, border_mode);
    lds_h[u32(local_y + r) * lds_stride + u32(local_x)] = h_sum;

    // ===== 阶段 2: 加载 halo 行到 LDS =====
    // 顶部 halo: 行 0..r-1（global row = wid.y*8 - r .. wid.y*8 - 1）
    // 底部 halo: 行 r+8..r+8+r-1（global row = wid.y*8 + 8 .. wid.y*8 + 8 + r - 1）
    // 64 个线程轮流加载 2*r*8 个 halo 像素
    let total_halo = u32(r) * 2u * lds_stride;
    var h_idx = local_idx;
    while (h_idx < total_halo) {
        let is_top = h_idx < u32(r) * lds_stride;
        let halo_col = h_idx % lds_stride;

        var halo_global_row: i32;
        var lds_row: u32;

        if (is_top) {
            let halo_row = h_idx / lds_stride;
            halo_global_row = i32(wid.y) * 8 - r + i32(halo_row);
            lds_row = halo_row;
        } else {
            let bottom_idx = h_idx - u32(r) * lds_stride;
            let halo_row = bottom_idx / lds_stride;
            halo_global_row = i32(wid.y) * 8 + 8 + i32(halo_row);
            lds_row = u32(r) + 8u + halo_row;
        }

        let halo_global_col = i32(wid.x) * 8 + i32(halo_col);

        let halo_sum = compute_horizontal_conv(halo_global_col, halo_global_row, border_mode);
        lds_h[lds_row * lds_stride + halo_col] = halo_sum;

        h_idx = h_idx + 64u;
    }

    workgroupBarrier();

    // ===== 阶段 3: 从 LDS 读取水平结果，计算垂直卷积 =====
    // LDS 行映射: lds_row = local_y + ky
    // 行 0 对应 global row = wid.y*8 - r，行 local_y+ky 对应 global row = wid.y*8 + local_y + ky - r
    if (global_x < i32(params.width) && global_y < i32(params.height)) {
        var v_sum: f32 = 0.0;
        let ks = i32(params.kernel_size);
        for (var ky = 0; ky < ks; ky = ky + 1) {
            let lds_row_idx = u32(local_y + ky);
            let val = lds_h[lds_row_idx * lds_stride + u32(local_x)];
            let weight = params.kernel[u32(ky)];
            v_sum = v_sum + val * weight;
        }
        let result = u32(clamp(v_sum, 0.0, 255.0));
        output[u32(global_y) * params.width + u32(global_x)] = result;
    }
}
