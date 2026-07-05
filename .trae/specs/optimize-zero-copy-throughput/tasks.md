# Tasks

## Task 1: Cargo.toml feature 依赖更新
- [ ] 1.1: 修改 `czkawka-compat` feature 定义，增加 `parallel-cpu` 依赖（`czkawka-compat = ["image", "parallel-cpu"]`）
- [ ] 1.2: 可选：新增 `simd` feature，依赖 `wide` crate（`wide = { version = "0.7", optional = true }`）
- [ ] 1.3: 验证 `cargo build --features czkawka-compat` 编译通过，rayon 自动引入

## Task 2: rayon 并行图像解码 — GPU 缩放路径
- [ ] 2.1: 在 `czkawka_compat.rs` 中重构 `compute_hashes_zero_copy_pipelined` 的 CPU 工作线程
  - 将 `for i in 0..image_count` 串行循环改为：先收集 `sub_batch_size` 个索引，再使用 `rayon::par_iter` 并行调用 `loader(i)` + `rgba_to_grayscale_cpu()`
  - 并行结果按索引排序，保持与串行版本一致的输出顺序
  - 子批数据通过 `sync_channel(1)` 发送给主线程（背压机制不变）
- [ ] 2.2: 添加 `Send` bound 确保 rayon 闭包可跨线程（当前已有 `+ Send`，验证兼容性）
- [ ] 2.3: 编写单元测试验证并行解码结果与串行版本一致

## Task 3: rayon 并行图像解码 — Lanczos3 路径
- [ ] 3.1: 在 `czkawka_compat.rs` 中重构 `compute_hashes_zero_copy_lanczos3_pipelined` 的 CPU 工作线程
  - 同 Task 2，但并行执行 `loader(i)` + `img.resize_exact(Lanczos3)` + `grayscale()` + 像素收集
  - 注意 `image::DynamicImage` 是 `Send`，可跨线程传递
- [ ] 3.2: 编写单元测试验证 Lanczos3 并行路径结果一致性

## Task 4: SIMD 加速 RGBA→灰度转换
- [ ] 4.1: 在 `src/tasks/phasher.rs` 中新增 `rgba_to_grayscale_simd()` 函数
  - 使用 `wide` crate 的 `u8x16` / `u32x8` SIMD 类型
  - 一次处理 16 个像素：加载 64 字节 RGBA → 拆分 R/G/B 通道 → SIMD 乘法 + 加法 + 移位 → 打包输出
  - 处理尾部不足 16 像素的剩余部分（标量回退）
- [ ] 4.2: 将 `rgba_to_grayscale_cpu()` 改为分发器
  - `#[cfg(feature = "simd")]` → 调用 `rgba_to_grayscale_simd()`
  - 否则 → 调用标量实现（当前逻辑，重命名为 `rgba_to_grayscale_scalar`）
- [ ] 4.3: 编写单元测试验证 SIMD 路径与标量路径结果逐字节一致
  - 测试用例：纯色、渐变、随机像素、奇数像素数（尾部处理）

## Task 5: GPU 端 u8→u32 像素打包着色器
- [ ] 5.1: 在 `src/tasks/color_convert.wgsl` 中新增 `pack_u8_to_u32` 入口点
  - 输入：`array<u32>`（原始 u8 数据 reinterpret 为 u32，每 4 字节含 4 个灰度像素）
  - 输出：`array<u32>`（每个 u32 含 1 个灰度像素值）
  - workgroup_size = 256，每线程处理 4 个像素（1 个输入 u32 → 4 个输出 u32）
  - 参数：`pixel_count: u32`
- [ ] 5.2: 在 `src/tasks/phasher_util.rs` 中新增 `upload_raw_to_gpu()` 函数
  - 直接上传原始 u8 灰度数据到 GPU（数据量 = `pixel_count` 字节，向上对齐到 4 字节）
  - 分配 u32 输出 buffer（`pixel_count * 4` 字节）
  - Dispatch 打包着色器，生成 u32 格式 GpuBuffer
  - 返回 u32 GpuBuffer（与 `upload_image_to_gpu` 返回类型一致）
- [ ] 5.3: 在 `src/tasks/phasher.rs` 中新增 `preprocess_gpu_raw()` 方法
  - 接受原始 u8 数据，调用 `upload_raw_to_gpu()` + 可选 blur
  - 返回 `Vec<GpuBuffer>`（与 `preprocess_gpu` 签名一致）
- [ ] 5.4: 编写单元测试验证 GPU 打包结果与 CPU `pack_u8_to_u32()` 一致

## Task 6: 零拷贝流水线集成 GPU 端打包
- [ ] 6.1: 修改 `compute_hashes_zero_copy_pipelined` 和 `compute_hashes_zero_copy_lanczos3_pipelined`
  - CPU 线程发送灰度 u8 数据（不打包为 u32）到主线程
  - 主线程使用 `preprocess_gpu_raw()` 替代 `preprocess_gpu()` 上传 + 打包
- [ ] 6.2: 验证零拷贝模式端到端哈希结果不变
- [ ] 6.3: 基准测试对比 CPU 打包 vs GPU 打包的上传时间和总吞吐量

## Task 7: 异步 GPU 流水线 — 上传与计算重叠
- [ ] 7.1: 在 `czkawka_compat.rs` 中新增 `compute_hashes_zero_copy_async_pipelined()` 方法
  - 三阶段重叠流水线架构：
    - CPU 线程：rayon 并行解码 + 灰度转换 → sync_channel 发送子批
    - 主线程上传子批 N（`queue.write_buffer` 到 encoder A）
    - GPU 计算子批 N-1（dispatch 到 encoder B，与 encoder A 在同一 `queue.submit`）
  - 双 CommandEncoder 交替：submit 后立即创建新 encoder
  - 子批结果累积，全部完成后统一返回
- [ ] 7.2: 处理首尾边界情况
  - 第一个子批：仅上传，无计算
  - 最后一个子批：仅计算，无上传
  - 单子批：退化为串行模式
- [ ] 7.3: 编写单元测试验证异步流水线结果与非重叠版本一致
- [ ] 7.4: 基准测试对比异步流水线 vs 当前流水线的 GPU 空闲时间

## Task 8: CLI 参数与示例更新
- [ ] 8.1: 在 `examples/similar_images.rs` 中新增 `--threads <N>` 参数
  - 控制 rayon 线程数（默认 = `num_cpus - 1`）
  - 通过 `rayon::ThreadPoolBuilder::new().num_threads(N).build_global()` 配置
- [ ] 8.2: 新增 `--async-pipeline` 参数标志
  - 启用时调用 `compute_hashes_zero_copy_async_pipelined()`
  - 禁用时调用 `compute_hashes_zero_copy_pipelined()`（当前版本）
- [ ] 8.3: 更新帮助文本和示例用法

## Task 9: 基准测试
- [ ] 9.1: 新增 `benches/pipeline_throughput_bench.rs`
  - 测试矩阵：串行 vs rayon | 标量 vs SIMD | CPU 打包 vs GPU 打包 | 串行流水线 vs 异步流水线
  - 图像数量：100 / 500 / 2000
  - 图像尺寸：512×512 / 1920×1080 / 3840×2160
- [ ] 9.2: 运行基准测试，记录结果到 `benches/pipeline_optimization_report.md`
- [ ] 9.3: 对比优化前后吞吐量，验证预估加速比

## Task 10: 一致性测试
- [ ] 10.1: 新增 `tests/pipeline_optimization_test.rs`
  - 测试 rayon 并行解码 vs 串行解码 → 哈希结果一致
  - 测试 SIMD 灰度转换 vs 标量灰度转换 → 结果逐字节一致
  - 测试 GPU 端打包 vs CPU 端打包 → 后续哈希结果一致
  - 测试异步流水线 vs 串行流水线 → 哈希结果一致
- [ ] 10.2: 运行全部测试，确保无回归

# Task Dependencies

- [Task 2] depends on [Task 1] — rayon 并行解码需要 `parallel-cpu` feature
- [Task 3] depends on [Task 1] — 同上
- [Task 4] depends on [Task 1] — SIMD 需要 `simd` feature（可选依赖 `wide`）
- [Task 5] depends on nothing — GPU 着色器独立开发
- [Task 6] depends on [Task 5] — 集成 GPU 打包需要着色器和新上传函数就绪
- [Task 6] depends on [Task 2] — 修改流水线需要 rayon 版本作为基础
- [Task 7] depends on [Task 2] — 异步流水线在 rayon 并行版本基础上重构
- [Task 7] depends on [Task 6] — 异步流水线使用 GPU 端打包
- [Task 8] depends on [Task 7] — CLI 参数需要异步流水线方法就绪
- [Task 9] depends on [Task 8] — 基准测试需要 CLI 可运行
- [Task 10] depends on [Task 7] — 一致性测试覆盖所有优化路径
