// GPU 2D 卷积着色器
// 支持三种边界模式（Zero/Clamp/Reflect）和三种通道模式（2D/水平1D/垂直1D）

struct ConvParams {
    width: u32,
    height: u32,
    kernel_size: u32,
    kernel_radius: u32,
    border_mode: u32,   // 0=Zero, 1=Clamp, 2=Reflect
    pass_mode: u32,     // 0=Full2D, 1=Horizontal1D, 2=Vertical1D
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

// 根据边界模式获取像素值
fn get_pixel(x: i32, y: i32, border_mode: u32) -> f32 {
    let w = i32(params.width);
    let h = i32(params.height);

    if (border_mode == 0u) {
        // Zero: 越界返回 0
        if (x < 0 || x >= w || y < 0 || y >= h) {
            return 0.0;
        }
        return f32(pixels[u32(y) * params.width + u32(x)]);
    }

    var fx = x;
    var fy = y;

    if (border_mode == 1u) {
        // Clamp: 钳制到边缘
        fx = clamp(x, 0, w - 1);
        fy = clamp(y, 0, h - 1);
    } else if (border_mode == 2u) {
        // Reflect: 镜像反射
        fx = reflect_coord(x, w);
        fy = reflect_coord(y, h);
    }

    return f32(pixels[u32(fy) * params.width + u32(fx)]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel_idx = gid.x;
    let total_pixels = params.width * params.height;

    if (pixel_idx >= total_pixels) {
        return;
    }

    let x = i32(pixel_idx % params.width);
    let y = i32(pixel_idx / params.width);

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

    // 钳制到 [0, 255] 并写入输出
    let result = u32(clamp(sum, 0.0, 255.0));
    output[pixel_idx] = result;
}
