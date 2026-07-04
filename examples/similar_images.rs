//! # similar_images — GPU 加速的相似图像检测 CLI 工具
//!
//! 模拟 czkawka 的相似图像检测流程，集成 gpgpu-tool 的 GPU 加速能力：
//!
//! 1. 扫描目录中的图像文件（PNG/JPEG/BMP/GIF/WebP）
//! 2. 加载图像（支持 GPU 双线性或 Lanczos3 高质量缩放）
//! 3. 使用 GPU 批量计算感知哈希
//! 4. 使用 GPU 汉明距离矩阵查找相似对
//! 5. 按相似组分组输出结果
//!
//! ## 用法
//!
//! ```bash
//! # 基本用法（默认: Gradient 算法, hash_size=8, tolerance=10, GPU 缩放）
//! cargo run --example similar_images --features czkawka-compat -- /path/to/images
//!
//! # 使用 Lanczos3 高质量缩放
//! cargo run --example similar_images --features czkawka-compat -- \
//!     /path/to/images --resize lanczos3 -s 32 -a gradient
//!
//! # 零拷贝 GPU 流式加载（最小化 CPU 内存，适合大批量处理）
//! cargo run --example similar_images --features czkawka-compat -- \
//!     /path/to/images --zero-copy -r
//!
//! # 参数说明
//! #   -t, --tolerance <N>     汉明距离容差 (默认: 10)
//! #   -a, --algorithm <ALGO>  哈希算法 (默认: gradient)
//! #   -s, --hash-size <N>     哈希尺寸 8/16/32/64 (默认: 8)
//! #   --resize <MODE>         缩放算法: gpu(默认) / lanczos3
//! #   --zero-copy             启用零拷贝 GPU 流式加载（最小化 CPU 内存）
//! #   --memory-budget <MB>    CPU 内存预算（MB，默认: 512）
//! #   -r, --recursive         递归扫描子目录
//! #   -v, --verbose           详细输出（显示哈希值）
//! #   -h, --help              帮助
//! ```

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
use gpgpu_tool::tasks::phasher::HashAlgorithm;
use gpgpu_tool::GpuError;
use gpgpu_tool::HashBytes;
use gpgpu_tool::MemoryManager;

// ── CLI 参数解析 ───────────────────────────────────────────────

/// 缩放算法选择
#[derive(Clone, Copy, PartialEq)]
enum ResizeMode {
    /// GPU 双线性插值（默认，速度快）
    Gpu,
    /// CPU Lanczos3 高质量缩放（精度高，速度较慢）
    Lanczos3,
}

struct CliArgs {
    directory: PathBuf,
    tolerance: u32,
    algorithm: HashAlgorithm,
    hash_size: u8,
    resize_mode: ResizeMode,
    /// 零拷贝 GPU 流式加载
    zero_copy: bool,
    /// CPU 内存预算（MB）
    memory_budget_mb: u64,
    /// 子批大小（流水线模式：每多少张图像批量上传 GPU）
    sub_batch_size: usize,
    /// 限制处理的图像数量（0 = 不限制）
    limit: usize,
    /// 禁用流水线优化（回退到逐张上传模式）
    no_pipeline: bool,
    recursive: bool,
    verbose: bool,
}

fn parse_algorithm(s: &str) -> Option<HashAlgorithm> {
    match s.to_lowercase().as_str() {
        "mean" => Some(HashAlgorithm::Mean),
        "median" => Some(HashAlgorithm::Median),
        "gradient" => Some(HashAlgorithm::Gradient),
        "block" | "blockhash" => Some(HashAlgorithm::Block),
        "vertgradient" | "vert_gradient" => Some(HashAlgorithm::VertGradient),
        "doublegradient" | "double_gradient" => Some(HashAlgorithm::DoubleGradient),
        _ => None,
    }
}

fn print_help() {
    println!("similar_images — GPU 加速的相似图像检测 CLI 工具");
    println!();
    println!("USAGE:");
    println!("    similar_images [OPTIONS] <DIRECTORY>");
    println!();
    println!("ARGS:");
    println!("    <DIRECTORY>    要扫描的图像目录");
    println!();
    println!("OPTIONS:");
    println!("    -t, --tolerance <N>     汉明距离容差 (默认: 10)");
    println!("    -a, --algorithm <ALGO>  哈希算法: mean/median/gradient/block/vertgradient/doublegradient (默认: gradient)");
    println!("    -s, --hash-size <N>     哈希尺寸: 8/16/32/64 (默认: 8, 对应 64-bit)");
    println!("        --resize <MODE>     缩放算法: gpu(默认) / lanczos3 (高质量CPU缩放)");
    println!("        --zero-copy             启用零拷贝 GPU 流式加载（流水线优化，最小化 CPU 内存）");
    println!("        --memory-budget <MB> CPU 内存预算 (MB, 默认: 512)");
    println!("        --sub-batch-size <N>   零拷贝流水线子批大小 (默认: 16, 范围 1-64)");
    println!("        --limit <N>            限制处理的图像数量 (默认: 0 = 不限制)");
    println!("        --no-pipeline          禁用流水线优化 (回退到逐张上传模式)");
    println!("    -r, --recursive         递归扫描子目录");
    println!("    -v, --verbose           详细输出（显示哈希值）");
    println!("    -h, --help              打印帮助信息");
    println!();
    println!("EXAMPLES:");
    println!("    similar_images /photos");
    println!("    similar_images /photos -t 5 -a block -s 16");
    println!("    similar_images /photos --resize lanczos3 -s 32 -a gradient");
    println!("    similar_images /photos --zero-copy -r  (零拷贝大批量处理)");
    println!("    similar_images /photos -r -v");
}

fn parse_args() -> Result<CliArgs, String> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return Err("缺少目录参数".to_string());
    }

    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        std::process::exit(0);
    }

    let mut directory = None;
    let mut tolerance: u32 = 10;
    let mut algorithm = HashAlgorithm::Gradient;
    let mut hash_size: u8 = 8;
    let mut resize_mode = ResizeMode::Gpu;
    let mut zero_copy = false;
    let mut memory_budget_mb: u64 = 512;
    let mut sub_batch_size: usize = 16;
    let mut limit: usize = 0;
    let mut no_pipeline = false;
    let mut recursive = false;
    let mut verbose = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-t" | "--tolerance" => {
                i += 1;
                if i >= args.len() {
                    return Err("--tolerance 需要参数".to_string());
                }
                tolerance = args[i]
                    .parse()
                    .map_err(|_| format!("无效的 tolerance 值: {}", args[i]))?;
            }
            "-a" | "--algorithm" => {
                i += 1;
                if i >= args.len() {
                    return Err("--algorithm 需要参数".to_string());
                }
                algorithm = parse_algorithm(&args[i])
                    .ok_or_else(|| format!("未知算法: {} (可选: mean/median/gradient/block/vertgradient/doublegradient)", args[i]))?;
            }
            "-s" | "--hash-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("--hash-size 需要参数".to_string());
                }
                hash_size = args[i]
                    .parse()
                    .map_err(|_| format!("无效的 hash-size 值: {}", args[i]))?;
                if !matches!(hash_size, 8 | 16 | 32 | 64) {
                    return Err(format!("hash-size 必须是 8/16/32/64, 得到: {}", hash_size));
                }
            }
            "--resize" => {
                i += 1;
                if i >= args.len() {
                    return Err("--resize 需要参数".to_string());
                }
                resize_mode = match args[i].to_lowercase().as_str() {
                    "gpu" => ResizeMode::Gpu,
                    "lanczos" | "lanczos3" => ResizeMode::Lanczos3,
                    _ => return Err(format!("未知缩放算法: {} (可选: gpu / lanczos3)", args[i])),
                };
            }
            "--zero-copy" => {
                zero_copy = true;
            }
            "--sub-batch-size" => {
                i += 1;
                if i >= args.len() {
                    return Err("--sub-batch-size 需要参数".to_string());
                }
                sub_batch_size = args[i]
                    .parse()
                    .map_err(|_| format!("无效的 sub-batch-size 值: {}", args[i]))?;
                if sub_batch_size < 1 || sub_batch_size > 64 {
                    return Err(format!("sub-batch-size 必须在 1-64 范围内, 得到: {}", sub_batch_size));
                }
            }
            "--limit" => {
                i += 1;
                if i >= args.len() {
                    return Err("--limit 需要参数".to_string());
                }
                limit = args[i]
                    .parse()
                    .map_err(|_| format!("无效的 limit 值: {}", args[i]))?;
            }
            "--no-pipeline" => {
                no_pipeline = true;
            }
            "--memory-budget" => {
                i += 1;
                if i >= args.len() {
                    return Err("--memory-budget 需要参数".to_string());
                }
                memory_budget_mb = args[i]
                    .parse()
                    .map_err(|_| format!("无效的 memory-budget 值: {}", args[i]))?;
            }
            "-r" | "--recursive" => {
                recursive = true;
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            arg if arg.starts_with('-') => {
                return Err(format!("未知选项: {}", arg));
            }
            _ => {
                if directory.is_none() {
                    directory = Some(PathBuf::from(&args[i]));
                } else {
                    return Err(format!("多余的参数: {}", args[i]));
                }
            }
        }
        i += 1;
    }

    let directory = directory.ok_or("缺少目录参数")?;

    if !directory.exists() {
        return Err(format!("目录不存在: {}", directory.display()));
    }

    if !directory.is_dir() {
        return Err(format!("不是目录: {}", directory.display()));
    }

    Ok(CliArgs {
        directory,
        tolerance,
        algorithm,
        hash_size,
        resize_mode,
        zero_copy,
        memory_budget_mb,
        sub_batch_size,
        limit,
        no_pipeline,
        recursive,
        verbose,
    })
}

// ── 图像扫描与加载 ─────────────────────────────────────────────

/// 支持的图像文件扩展名
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "gif", "webp", "tiff", "tif"];

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// 扫描目录中的图像文件
fn scan_directory(dir: &Path, recursive: bool) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if recursive {
        scan_directory_recursive(dir, &mut files);
    } else {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_image_file(&path) {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    files
}

fn scan_directory_recursive(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                scan_directory_recursive(&path, files);
            } else if is_image_file(&path) {
                files.push(path);
            }
        }
    }
}

/// 最大允许的图像像素尺寸（宽或高），超过则跳过
const MAX_DIMENSION: u32 = 20000;

/// 加载图像为 RGBA 格式（GPU 缩放路径）
///
/// 使用 ImageReader 设置大小限制，防止超大图片导致 OOM。
fn load_image_rgba(path: &Path) -> Result<(Vec<u8>, u32, u32), String> {
    let reader = image::ImageReader::open(path)
        .map_err(|e| format!("打开失败: {}: {}", path.display(), e))?;
    let mut reader = reader.with_guessed_format()
        .map_err(|e| format!("格式识别失败: {}: {}", path.display(), e))?;
    let mut limits = image::Limits::no_limits();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    reader.limits(limits);

    let img = reader.decode()
        .map_err(|e| format!("解码失败: {}: {}", path.display(), e))?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((rgba.into_raw(), w, h))
}

/// 加载图像为 DynamicImage（Lanczos3 路径）
fn load_image_dynamic(path: &Path) -> Result<image::DynamicImage, String> {
    let reader = image::ImageReader::open(path)
        .map_err(|e| format!("打开失败: {}: {}", path.display(), e))?;
    let mut reader = reader.with_guessed_format()
        .map_err(|e| format!("格式识别失败: {}: {}", path.display(), e))?;
    let mut limits = image::Limits::no_limits();
    limits.max_image_width = Some(MAX_DIMENSION);
    limits.max_image_height = Some(MAX_DIMENSION);
    reader.limits(limits);
    reader.decode()
        .map_err(|e| format!("解码失败: {}: {}", path.display(), e))
}

// ── 相似组分组（Union-Find）─────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let px = self.find(x);
        let py = self.find(y);
        if px == py {
            return;
        }
        if self.rank[px] < self.rank[py] {
            self.parent[px] = py;
        } else if self.rank[px] > self.rank[py] {
            self.parent[py] = px;
        } else {
            self.parent[py] = px;
            self.rank[px] += 1;
        }
    }
}

// ── 格式化辅助 ──────────────────────────────────────────────────

fn hash_to_hex(hash: &[u8]) -> String {
    hash.iter().map(|b| format!("{:02x}", b)).collect::<String>()
}

fn format_duration(ms: u64) -> String {
    if ms < 1000 {
        format!("{}ms", ms)
    } else {
        format!("{:.2}s", ms as f64 / 1000.0)
    }
}

// ── 主流程 ──────────────────────────────────────────────────────

fn main() {
    env_logger::init();

    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    };

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     similar_images — GPU 加速相似图像检测工具           ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. 扫描目录 ─────────────────────────────────────────────
    println!("[1/4] 扫描图像文件...");
    println!("      目录: {}", args.directory.display());
    println!("      递归: {}", if args.recursive { "是" } else { "否" });

    let scan_start = Instant::now();
    let mut image_files = scan_directory(&args.directory, args.recursive);
    let scan_elapsed = scan_start.elapsed();

    // 应用 --limit 限制
    if args.limit > 0 && args.limit < image_files.len() {
        image_files.truncate(args.limit);
    }

    if image_files.is_empty() {
        println!("\n      未找到图像文件。");
        println!("\n      支持的格式: PNG, JPEG, BMP, GIF, WebP, TIFF");
        return;
    }

    println!("      找到 {} 个图像文件 ({})", image_files.len(), format_duration(scan_elapsed.as_millis() as u64));

    // ── 2. 初始化 GPU 加速器 ────────────────────────────────────
    println!("\n[2/4] 初始化 GPU 加速器...");
    println!("      算法: {:?} (czkawka 名: {})", args.algorithm, args.algorithm.to_czkawka_name());
    println!("      哈希尺寸: {} ({}-bit)", args.hash_size, (args.hash_size as u32) * (args.hash_size as u32));
    println!("      容差: {}", args.tolerance);
    println!("      缩放算法: {}", match args.resize_mode {
        ResizeMode::Gpu => "GPU 双线性插值 (快速)",
        ResizeMode::Lanczos3 => "Lanczos3 高质量缩放 (CPU)",
    });
    println!("      零拷贝模式: {}", if args.zero_copy { format!("是 (流水线优化, 子批={})", args.sub_batch_size) } else { "否 (批量加载)".to_string() });
    println!("      内存预算: {} MB", args.memory_budget_mb);

    // 根据零拷贝模式选择加速器类型
    let accelerator = if args.zero_copy {
        match CzkawkaGpuAccelerator::new_with_gpu_resize(args.hash_size, args.algorithm) {
            Ok(acc) => {
                let ctx = acc.shared_context().lock().unwrap();
                println!("      GPU 适配器: {}", ctx.adapter_info());
                drop(ctx);
                acc
            }
            Err(GpuError::GpuUnavailable) => {
                eprintln!("\n错误: GPU 不可用，无法启动零拷贝 GPU 加速。");
                eprintln!("      零拷贝模式需要 GPU 支持。");
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("\n错误: GPU 初始化失败: {}", e);
                std::process::exit(3);
            }
        }
    } else {
        match CzkawkaGpuAccelerator::new(args.hash_size, args.algorithm) {
            Ok(acc) => {
                let ctx = acc.shared_context().lock().unwrap();
                println!("      GPU 适配器: {}", ctx.adapter_info());
                drop(ctx);
                acc
            }
            Err(GpuError::GpuUnavailable) => {
                eprintln!("\n错误: GPU 不可用，无法启动 GPU 加速。");
                eprintln!("      请确保系统有支持 Vulkan/Metal/DX12 的 GPU。");
                std::process::exit(2);
            }
            Err(e) => {
                eprintln!("\n错误: GPU 初始化失败: {}", e);
                std::process::exit(3);
            }
        }
    };

    // 初始化动态内存管理器
    let ctx_ref = accelerator.shared_context().lock().unwrap();
    let mut mem_manager = MemoryManager::from_context(&ctx_ref)
        .with_cpu_budget(args.memory_budget_mb * 1024 * 1024)
        .with_batch_range(2, 256);
    drop(ctx_ref);

    // ── 3. 流式分批加载图像并计算哈希 ─────────────────────────
    println!("\n[3/4] 流式分批加载 + GPU 感知哈希计算...");
    if args.zero_copy {
        println!("      模式: 零拷贝 GPU 流水线（CPU/GPU 双缓冲, 子批={}, 峰值 CPU 内存 = {} 张图像）",
            args.sub_batch_size, args.sub_batch_size);
    }

    let hash_start = Instant::now();

    let mut valid_files: Vec<PathBuf> = Vec::new();
    let mut load_errors: Vec<(PathBuf, String)> = Vec::new();
    let mut all_hashes: Vec<HashBytes> = Vec::new();

    let total = image_files.len();

    if args.zero_copy {
        // ═══════════════════════════════════════════════════════════
        // 零拷贝模式：逐张加载到 GPU 显存，最小化 CPU 内存
        // ═══════════════════════════════════════════════════════════
        let mut processed = 0;
        let mut offset = 0;

        while offset < total {
            // 动态计算批次大小（首次使用默认值，后续根据实际情况调整）
            let batch_size = if processed == 0 {
                mem_manager.batch_size()
            } else {
                mem_manager.batch_size()
            };

            let end = (offset + batch_size).min(total);
            let chunk = &image_files[offset..end];

            let batch_start = Instant::now();
            let batch_count = chunk.len();

            // 零拷贝 GPU 加载
            match args.resize_mode {
                ResizeMode::Gpu => {
                    // GPU 缩放路径
                    let result = if args.no_pipeline {
                        // 旧版：逐张上传，无流水线
                        accelerator.compute_hashes_zero_copy(
                            chunk.len(),
                            |i| { load_image_rgba(&chunk[i]) },
                        )
                    } else {
                        // 新版：流水线优化
                        accelerator.compute_hashes_zero_copy_pipelined(
                            chunk.len(),
                            |i| { load_image_rgba(&chunk[i]) },
                            args.sub_batch_size,
                        )
                    };

                    match result {
                        Ok(hashes) => {
                            let success_count = hashes.len();
                            for j in 0..success_count {
                                valid_files.push(chunk[j].clone());
                            }
                            all_hashes.extend(hashes);
                            mem_manager.report_success(batch_count);
                        }
                        Err(e) => {
                            eprintln!("\n      GPU 批次失败 (offset={}): {}", offset, e);
                            mem_manager.report_oom();
                        }
                    }
                }
                ResizeMode::Lanczos3 => {
                    // Lanczos3 路径
                    let result = if args.no_pipeline {
                        accelerator.compute_hashes_zero_copy_lanczos3(
                            chunk.len(),
                            |i| { load_image_dynamic(&chunk[i]) },
                        )
                    } else {
                        accelerator.compute_hashes_zero_copy_lanczos3_pipelined(
                            chunk.len(),
                            |i| { load_image_dynamic(&chunk[i]) },
                            args.sub_batch_size,
                        )
                    };

                    match result {
                        Ok(hashes) => {
                            let success_count = hashes.len();
                            for j in 0..success_count {
                                valid_files.push(chunk[j].clone());
                            }
                            all_hashes.extend(hashes);
                            mem_manager.report_success(batch_count);
                        }
                        Err(e) => {
                            eprintln!("\n      GPU 批次失败 (offset={}): {}", offset, e);
                            mem_manager.report_oom();
                        }
                    }
                }
            }

            let batch_elapsed = batch_start.elapsed();
            processed += batch_count;
            offset = end;

            if processed % 500 < batch_size || processed >= total {
                let throughput = if batch_elapsed.as_millis() > 0 {
                    (batch_count as f64 / batch_elapsed.as_secs_f64()).round() as u64
                } else {
                    0
                };
                println!("      进度: {}/{} ({} 哈希, {} 错误, {} 图/s, batch={})",
                    processed, total, all_hashes.len(), load_errors.len(),
                    throughput, mem_manager.batch_size());
                let _ = std::io::stdout().flush();
            }
        }
    } else {
        // ═══════════════════════════════════════════════════════════
        // 传统模式：批量加载到 CPU 内存（兼容原有逻辑）
        // ═══════════════════════════════════════════════════════════

        // 首先采样几张图像来估算批次大小
        let sample_size = total.min(20);
        let mut sample_widths: Vec<u32> = Vec::new();
        let mut sample_heights: Vec<u32> = Vec::new();
        for i in 0..sample_size {
            if let Ok(reader) = image::ImageReader::open(&image_files[i]) {
                if let Ok(reader) = reader.with_guessed_format() {
                    if let Ok(dims) = reader.into_dimensions() {
                        sample_widths.push(dims.0);
                        sample_heights.push(dims.1);
                    }
                }
            }
        }

        // 根据采样结果计算初始批次大小
        if !sample_widths.is_empty() {
            let avg_w = sample_widths.iter().sum::<u32>() / sample_widths.len() as u32;
            let avg_h = sample_heights.iter().sum::<u32>() / sample_heights.len() as u32;
            mem_manager.calculate_batch_size(avg_w, avg_h);
            println!("      采样: 平均尺寸 {}×{} → 初始 batch_size={}", avg_w, avg_h, mem_manager.batch_size());
        }

        let batch_size = mem_manager.batch_size();

        for (batch_idx, chunk) in image_files.chunks(batch_size).enumerate() {
            let batch_start = batch_idx * batch_size;

            match args.resize_mode {
                ResizeMode::Gpu => {
                    let mut rgba_images: Vec<Vec<u8>> = Vec::with_capacity(chunk.len());
                    let mut dims: Vec<(u32, u32)> = Vec::with_capacity(chunk.len());

                    for file in chunk {
                        match load_image_rgba(file) {
                            Ok((rgba, w, h)) => {
                                rgba_images.push(rgba);
                                dims.push((w, h));
                                valid_files.push(file.clone());
                            }
                            Err(e) => {
                                load_errors.push((file.clone(), e));
                            }
                        }
                    }

                    if !rgba_images.is_empty() {
                        match accelerator.compute_hashes(&rgba_images, &dims) {
                            Ok(batch_hashes) => {
                                all_hashes.extend(batch_hashes);
                                mem_manager.report_success(chunk.len());
                            }
                            Err(e) => {
                                eprintln!("\n      GPU 批次失败 (offset={}): {}", batch_start, e);
                                mem_manager.report_oom();
                            }
                        }
                    }
                    // rgba_images 和 dims 在此处自动 drop，释放内存
                }
                ResizeMode::Lanczos3 => {
                    let mut dynamic_images: Vec<image::DynamicImage> = Vec::with_capacity(chunk.len());

                    for file in chunk {
                        match load_image_dynamic(file) {
                            Ok(img) => {
                                dynamic_images.push(img);
                                valid_files.push(file.clone());
                            }
                            Err(e) => {
                                load_errors.push((file.clone(), e));
                            }
                        }
                    }

                    if !dynamic_images.is_empty() {
                        match accelerator.compute_hashes_lanczos3(&dynamic_images) {
                            Ok(batch_hashes) => {
                                all_hashes.extend(batch_hashes);
                                mem_manager.report_success(chunk.len());
                            }
                            Err(e) => {
                                eprintln!("\n      GPU 批次失败 (offset={}): {}", batch_start, e);
                                mem_manager.report_oom();
                            }
                        }
                    }
                    // dynamic_images 在此处自动 drop，释放内存
                }
            }

            let processed = batch_start + chunk.len();
            if processed % 500 < batch_size || processed == total {
                println!("      进度: {}/{} ({} 哈希, {} 错误, batch={})",
                    processed, total, all_hashes.len(), load_errors.len(), mem_manager.batch_size());
                let _ = std::io::stdout().flush();
            }
        }
    }

    let hash_elapsed = hash_start.elapsed();
    println!("\n      完成: {} 哈希, {} 错误 ({})",
        all_hashes.len(), load_errors.len(), format_duration(hash_elapsed.as_millis() as u64));
    println!("      内存管理: {}", mem_manager.summary());

    if all_hashes.is_empty() {
        eprintln!("错误: 没有成功计算的哈希。");
        for (file, err) in load_errors.iter().take(20) {
            eprintln!("  {}: {}", file.display(), err);
        }
        if load_errors.len() > 20 {
            eprintln!("  ... (共 {} 个错误)", load_errors.len());
        }
        std::process::exit(4);
    }

    if !load_errors.is_empty() && args.verbose {
        println!("      加载失败详情 (前 20 个):");
        for (file, err) in load_errors.iter().take(20) {
            println!("        {} : {}", file.display(), err);
        }
        if load_errors.len() > 20 {
            println!("        ... (共 {} 个错误)", load_errors.len());
        }
    }

    if args.verbose {
        println!("\n      哈希值 (前 50 个):");
        for (i, (file, hash)) in valid_files.iter().zip(all_hashes.iter()).enumerate().take(50) {
            let hex = hash_to_hex(hash.as_bytes());
            let preview = if hex.len() > 32 {
                format!("{}...", &hex[..32])
            } else {
                hex
            };
            println!("        [{}] {} => {}", i, file.file_name().unwrap_or_default().to_string_lossy(), preview);
        }
        if all_hashes.len() > 50 {
            println!("        ... (共 {} 个哈希)", all_hashes.len());
        }
    }

    // ── 4. GPU 相似匹配 ────────────────────────────────────────
    println!("\n[4/4] GPU 汉明距离相似匹配...");
    let match_start = Instant::now();

    let pairs = accelerator
        .find_similar_pairs(&all_hashes, args.tolerance)
        .expect("GPU 相似匹配失败");

    let match_elapsed = match_start.elapsed();
    println!("      匹配完成: {} 个相似对 ({})", pairs.len(), format_duration(match_elapsed.as_millis() as u64));

    if all_hashes.len() >= 2000 {
        println!("      (使用 GPU 端 threshold 过滤管线，N={} >= 2000)", all_hashes.len());
    } else {
        println!("      (使用 GPU 距离矩阵管线，N={} < 2000)", all_hashes.len());
    }

    // ── 结果分组与输出 ──────────────────────────────────────────
    println!("\n{}", "═".repeat(60));
    println!("  检测结果");
    println!("{}", "═".repeat(60));

    if pairs.is_empty() {
        println!("\n  未发现相似图像对 (容差={})", args.tolerance);
        println!("  尝试增大容差值 (-t) 来检测更多相似图像。");
    } else {
        let mut uf = UnionFind::new(valid_files.len());
        for &(parent, child, _) in &pairs {
            uf.union(parent as usize, child as usize);
        }

        let mut groups: HashMap<usize, Vec<(usize, PathBuf)>> = HashMap::new();
        for (i, file) in valid_files.iter().enumerate() {
            let root = uf.find(i);
            groups.entry(root).or_default().push((i, file.clone()));
        }

        let mut similar_groups: Vec<(usize, Vec<(usize, PathBuf)>)> = groups
            .into_iter()
            .filter(|(_, members)| members.len() > 1)
            .collect();

        similar_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        let total_grouped: usize = similar_groups.iter().map(|(_, m)| m.len()).sum();
        let resize_name = match args.resize_mode {
            ResizeMode::Gpu => "GPU缩放",
            ResizeMode::Lanczos3 => "Lanczos3",
        };
        let mode_name = if args.zero_copy { format!("零拷贝流水线(s={})", args.sub_batch_size) } else { "批量".to_string() };

        println!("\n  发现 {} 组相似图像 (共 {} 张图片，{} 个相似对)",
            similar_groups.len(), total_grouped, pairs.len());
        println!("  容差: {} | 算法: {:?} | 哈希: {}-bit | 缩放: {} | 模式: {}\n",
            args.tolerance, args.algorithm, (args.hash_size as u32) * (args.hash_size as u32), resize_name, mode_name);

        for (group_idx, (_, members)) in similar_groups.iter().enumerate() {
            println!("  ┌─ 组 {} ({} 张图片) {}", group_idx + 1, members.len(),
                "─".repeat(40usize.saturating_sub(20 + members.len().to_string().len())));

            let member_indices: HashSet<usize> = members.iter().map(|(i, _)| *i).collect();

            for &(my_idx, ref file) in members.iter() {
                let filename = file.file_name().unwrap_or_default().to_string_lossy();
                let parent_dir = file.parent()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default();

                let mut distances: Vec<u32> = Vec::new();
                for &(parent, child, dist) in &pairs {
                    let p = parent as usize;
                    let c = child as usize;
                    if member_indices.contains(&p) && member_indices.contains(&c) {
                        if p == my_idx || c == my_idx {
                            distances.push(dist);
                        }
                    }
                }

                let dist_info = if distances.is_empty() {
                    String::new()
                } else {
                    let min_dist = *distances.iter().min().unwrap_or(&0);
                    let max_dist = *distances.iter().max().unwrap_or(&0);
                    if min_dist == max_dist {
                        format!(" [距离={}]", min_dist)
                    } else {
                        format!(" [距离 {}-{}]", min_dist, max_dist)
                    }
                };

                if parent_dir.is_empty() {
                    println!("  │  {}{}", filename, dist_info);
                } else {
                    println!("  │  {}/{}{}", parent_dir, filename, dist_info);
                }
            }
            println!("  └{}\n", "─".repeat(50));
        }
    }

    // ── 统计摘要 ────────────────────────────────────────────────
    println!("{}", "─".repeat(60));
    println!("  统计摘要");
    println!("{}", "─".repeat(60));
    println!("  扫描文件:     {}", image_files.len());
    println!("  成功加载:     {}", valid_files.len());
    println!("  加载失败:     {}", load_errors.len());
    let resize_name = match args.resize_mode {
        ResizeMode::Gpu => "GPU缩放",
        ResizeMode::Lanczos3 => "Lanczos3",
    };
    let mode_name = if args.zero_copy { format!("零拷贝流水线(s={})", args.sub_batch_size) } else { "批量".to_string() };
    println!("  哈希计算:     {} ({:?}, {}, {})", format_duration(hash_elapsed.as_millis() as u64), args.algorithm, resize_name, mode_name);
    println!("  相似匹配:     {} ({})", format_duration(match_elapsed.as_millis() as u64), format!("{} pairs", pairs.len()));
    println!("  内存管理:     {}", mem_manager.summary());
    println!("  总耗时:       {}", format_duration((scan_elapsed + hash_elapsed + match_elapsed).as_millis() as u64));
    println!();
}
