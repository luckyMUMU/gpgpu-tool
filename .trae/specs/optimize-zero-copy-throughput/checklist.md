# Checklist

## Phase 1 — Cargo.toml 与 Feature 依赖（Task 1）
- [ ] `czkawka-compat` feature 定义中包含 `parallel-cpu` 依赖
- [ ] `simd` feature 可选引入 `wide` crate（或使用 `std::simd` nightly）
- [ ] `cargo build --features czkawka-compat` 编译通过，rayon 自动引入
- [ ] `cargo build`（无 feature）编译通过，不引入 rayon/wide

## Phase 2 — rayon 并行图像解码（Task 2-3）
- [ ] `compute_hashes_zero_copy_pipelined` CPU 工作线程使用 `rayon::par_iter` 并行解码
- [ ] `compute_hashes_zero_copy_lanczos3_pipelined` CPU 工作线程使用 `rayon::par_iter` 并行解码
- [ ] 并行结果按原始索引排序，顺序与串行版本一致
- [ ] `sync_channel(1)` 背压机制保持不变（CPU 最多领先 GPU 一个子批）
- [ ] `loader` 闭包的 `Send` bound 与 rayon 兼容
- [ ] 单元测试：并行解码 vs 串行解码 → 哈希结果一致
- [ ] 单元测试：Lanczos3 并行路径 vs 串行路径 → 哈希结果一致

## Phase 3 — SIMD 灰度转换（Task 4）
- [ ] `rgba_to_grayscale_simd()` 函数使用 `wide` crate 实现 SIMD 向量化
- [ ] 一次处理 16 个像素，尾部不足 16 像素时标量回退
- [ ] `rgba_to_grayscale_cpu()` 改为分发器（`simd` feature → SIMD 路径，否则 → 标量）
- [ ] 标量实现重命名为 `rgba_to_grayscale_scalar`（或内联在 `cfg` 分支中）
- [ ] SIMD 公式与标量一致：`(R*77 + G*150 + B*29) >> 8`
- [ ] 单元测试：纯色像素（R/G/B/白/黑）→ SIMD vs 标量结果一致
- [ ] 单元测试：渐变像素 → SIMD vs 标量结果一致
- [ ] 单元测试：奇数像素数（尾部处理）→ SIMD vs 标量结果一致
- [ ] 单元测试：4K 图像 → SIMD vs 标量结果逐字节一致

## Phase 4 — GPU 端像素打包（Task 5-6）
- [ ] `color_convert.wgsl` 新增 `pack_u8_to_u32` 入口点
- [ ] 着色器输入为 u32 reinterpret 的 u8 数据，输出为 u32 灰度像素
- [ ] workgroup_size = 256，每线程处理 4 个像素
- [ ] `phasher_util.rs` 新增 `upload_raw_to_gpu()` 函数
- [ ] 上传数据量 = `pixel_count` 字节（向上对齐 4 字节），非 `pixel_count * 4`
- [ ] `phasher.rs` 新增 `preprocess_gpu_raw()` 方法，签名与 `preprocess_gpu` 一致
- [ ] 零拷贝流水线使用 `preprocess_gpu_raw()` 替代 `preprocess_gpu()`
- [ ] 传统批量模式（`compute()`、`compute_from_rgba()`）保持 CPU 端打包不变
- [ ] 单元测试：GPU 打包 vs CPU `pack_u8_to_u32()` → 结果一致
- [ ] 单元测试：零拷贝模式端到端哈希结果不变
- [ ] 基准测试：CPU 打包 vs GPU 打包上传时间对比

## Phase 5 — 异步 GPU 流水线（Task 7）
- [ ] `compute_hashes_zero_copy_async_pipelined()` 方法实现三阶段重叠
- [ ] 双 CommandEncoder 交替使用（上传 encoder + 计算 encoder）
- [ ] 第一个子批：仅上传，无计算（边界正确）
- [ ] 最后一个子批：仅计算，无上传（边界正确）
- [ ] 单子批：退化为串行模式（边界正确）
- [ ] 子批结果按顺序累积，全部完成后统一返回
- [ ] 单元测试：异步流水线 vs 串行流水线 → 哈希结果一致
- [ ] 基准测试：异步流水线 GPU 空闲时间 < 串行流水线

## Phase 6 — CLI 与示例（Task 8）
- [ ] `--threads <N>` 参数控制 rayon 线程数
- [ ] `--async-pipeline` 参数启用异步流水线
- [ ] 帮助文本包含新参数说明
- [ ] 示例用法包含新参数的命令行示例
- [ ] 默认行为不变（`--threads` 默认 = `num_cpus - 1`，`--async-pipeline` 默认关闭）

## Phase 7 — 基准测试与验证（Task 9-10）
- [ ] `benches/pipeline_throughput_bench.rs` 覆盖所有优化路径组合
- [ ] 测试矩阵包含 3 种图像数量（100/500/2000）× 3 种尺寸（512p/1080p/4K）
- [ ] 基准测试结果记录到 `benches/pipeline_optimization_report.md`
- [ ] 优化前后吞吐量对比，验证预估加速比（目标 2-4×）
- [ ] `tests/pipeline_optimization_test.rs` 覆盖所有一致性测试
- [ ] `cargo test --features czkawka-compat` 全部通过，无回归
- [ ] `cargo clippy --features czkawka-compat` 无警告
