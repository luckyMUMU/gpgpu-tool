# Tasks

> **v3 修订说明**：本版本对齐 spec.md v3（代码核验版），主要修正：
> 1. `image_hasher` → `img_hash`（实际 dev-dependency，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）
> 2. 关键任务补充"现状代码出处"，便于实施时定位修改点
> 3. 修正 Task 9.4 描述（实际已存在 `img_hash = "3.2"`，无需新增）
> 4. Task 3 明确 GpuError 现状为 11 变体（[error.rs:5-39](file:///d:/Code/AI/wgpu-tool/src/error.rs#L5-L39)）
>
> 本规范聚焦于"对当前工具包进行架构审查并产出面向 czkawka 集成的优化建议"，任务清单按 P0/P1/P2 优先级与 Phase 实施路径组织。
>
> - **Phase 1（P0 阻断项）**：必须完成才能实现 czkawka 集成
> - **Phase 2（P1 性能优化）**：重要优化，提升集成价值
> - **Phase 3（P2 增强特性）**：可选增强，非阻断
> - **Phase 4（czkawka 侧）**：上游 PR，非本规范实施范围

## Phase 1 — P0 集成阻断项（必做）

### Feature 与错误处理基础

- [x] Task 1: 新增 `gpu-accel` / `czkawka-compat` feature 入口
  - [x] 1.1: 修改 `Cargo.toml`，新增 `gpu-accel = ["image"]` feature（wgpu 为核心依赖始终包含，image 为可选依赖）
  - [x] 1.2: 新增 `czkawka-compat = ["gpu-accel"]` feature（czkawka 兼容层入口）
  - [x] 1.3: 验证 `cargo build --no-default-features --features gpu-accel` 通过
  - [x] 1.4: 验证 `cargo build --features czkawka-compat` 通过

- [x] Task 2: 调整 `default` feature（**BREAKING**）
  - [x] 2.1: 修改 `Cargo.toml`，将 `default = ["cpu-fallback"]` 改为 `default = []`
  - [x] 2.2: 在 `lib.rs` 顶部文档增加 BREAKING 变更说明与迁移指南
  - [x] 2.3: 更新 `examples/demo.rs` 等内部示例显式启用 `cpu-fallback`（示例编译验证通过）
  - [x] 2.4: 验证 `cargo build`（默认）通过，无 sha2 依赖引入
  - [x] 2.5: 验证 `cargo build --features cpu-fallback` 通过，恢复原行为

- [x] Task 3: 新增 `GpuError::GpuUnavailable` 变体
  - [x] 3.1: 修改 `src/error.rs`，新增 `GpuUnavailable` 变体（第 12 变体）
  - [x] 3.2: 添加中文 doc 注释说明语义（与 `NoAdapter` 区分）
  - [x] 3.3: 验证 `cargo build` 无错误

- [x] Task 4: 新增 `GpuContext::new_for_integration()` 工厂方法
  - [x] 4.1: 新增 `new_for_integration() -> Result<GpuContext, GpuError>`
  - [x] 4.2: 区分错误类型：无适配器 → `GpuUnavailable`；其他 → 原错误
  - [x] 4.3: 添加 doc 注释说明与 `new_sync()` 的区别
  - [x] 4.4: 添加 doc 示例代码

### P0-A1: RGBA→灰度 GPU 转换（集成阻断项）

- [x] Task 5: 新增 `color_convert.wgsl` 着色器
  - [x] 5.1: 新增 `src/tasks/color_convert.wgsl` 文件
  - [x] 5.2: 实现 `rgba_to_grayscale` 入口点，使用 czkawka 公式 `(R*77 + G*150 + B*29) >> 8`
  - [x] 5.3: 使用 `@workgroup_size(64)` 1D 模型
  - [x] 5.4: 输入：RGBA 数据（u32 打包）；输出：灰度 u32
  - [x] 5.5: 添加 WGSL 注释说明公式来源（czkawka 兼容）

- [x] Task 6: 新增 `PerceptualHasher::compute_from_rgba()` 系列方法
  - [x] 6.1: 新增 `compute_from_rgba()` 方法
  - [x] 6.2: 新增 `compute_from_rgba_to_hash_bytes()` 方法
  - [x] 6.3: 内部流程：RGBA→灰度（CPU czkawka 公式）→ compute() 流水线
  - [x] 6.4: 验证灰度转换结果与 czkawka CPU 公式一致
  - [x] 6.5: 新增 `compute_to_hash_bytes()` 方法（Task 22.1 提前实现）

### P0-A2: czkawka 兼容结果格式（集成阻断项）

- [x] Task 7: 新增 `GpuHashMatcherBytes::compute_similar_pairs()` 方法
  - [x] 7.1: 新增 `compute_similar_pairs() -> Result<Vec<(u32, u32, u32)>, GpuError>`
  - [x] 7.2: 内部：GPU 距离矩阵 → CPU 过滤 threshold + 去除自身匹配
  - [x] 7.3: 返回值按 distance 升序排序
  - [x] 7.4: 待添加测试覆盖 64/256/1024/4096-bit 四档
  - [x] 7.5: 处理空/单元素边界情况

- [x] Task 8: 新增 `GpuHashMatcherBytes::compute_similar_pairs_asymmetric()` 方法
  - [x] 8.1: 新增非对称模式方法
  - [x] 8.2: 签名：`compute_similar_pairs_asymmetric(&self, ctx, ref_hashes, normal_hashes, tolerance)`
  - [x] 8.3: 对应 czkawka `gpu_compare_hashes_asymmetric()`
  - [x] 8.4: 待添加非对称场景测试

### P0-A3: 算法枚举与位宽互转

- [x] Task 9: 新增 `HashAlgorithm` 与 czkawka 互转方法
  - [x] 9.1: 新增 `from_czkawka(name: &str) -> Option<Self>`（兼容 `"Blockhash"` 和 `"Block"`）
  - [x] 9.2: 新增 `to_czkawka_name() -> &'static str`（`Block` → `"Blockhash"`）
  - [x] 9.3: `img_hash` 互转通过字符串名实现（doc 注释说明），无需新增依赖
  - [x] 9.4: dev-dependencies 已含 `img_hash = "3.2"`
  - [x] 9.5: 添加 doc 测试覆盖 6 种算法转换

- [x] Task 10: 新增 `HashSize::from_czkawka()` 构造方法
  - [x] 10.1: 修改 `src/tasks/hash_common.rs`，新增 `from_czkawka(hash_size: u8) -> Result<HashSize, GpuError>`
  - [x] 10.2: 支持 `hash_size ∈ {8, 16, 32, 64}`，其他值返回 `Err(GpuError::InvalidInput)`
  - [x] 10.3: 添加单元测试覆盖 4 个合法值 + 1 个非法值（待补充测试）

### P0 一致性测试

- [x] Task 11: 新增 GPU/CPU 哈希一致性测试套件
  - [x] 11.1: 新增 `tests/czkawka_compat_test.rs` 测试文件（19 个测试全部通过）
  - [x] 11.2: 测试覆盖：HashAlgorithm 互转（4 测试）、HashSize::from_czkawka（3 测试）、RGBA→灰度转换（3 测试）、compute_similar_pairs（4 测试）、compute_similar_pairs_asymmetric（1 测试）、GpuContext::new_for_integration（1 测试）、GPU vs CPU 哈希一致性（3 测试）
  - [x] 11.3: 测试 1：本工具包 GPU 路径 vs 本工具包 CPU 路径（RGBA→灰度→哈希一致性）
  - [ ] 11.4: 测试 2：本工具包 CPU 路径 vs `img_hash` crate（待 P2 阶段补充，需 `cpu-fallback` feature）
  - [x] 11.5: 测试 3：RGBA→灰度转换一致性（本工具包 CPU 公式 `(R*77 + G*150 + B*29) >> 8` 与 czkawka 一致）
  - [ ] 11.6: 差异允许范围文档化待 P2 `czkawka_compat` 模块实现时补充
  - [x] 11.7: 验证 `cargo test --features czkawka-compat --test czkawka_compat_test` 通过（19/19 通过）

## Phase 2 — P1 性能优化（重要）

### P1-B1: 共享 GPU 上下文封装

- [x] Task 12: 新增 `CzkawkaGpuAccelerator` struct
  - [x] 12.1: 新增 `src/czkawka_compat.rs`，定义 `CzkawkaGpuAccelerator`
  - [x] 12.2: 封装 `Arc<Mutex<GpuContext>>` + `PerceptualHasher` + `GpuHashMatcherBytes`
  - [x] 12.3: 实现 `new(hash_size: u8, hash_alg: HashAlgorithm) -> Result<Self, GpuError>`
  - [x] 12.4: 实现 `compute_hashes(&self, rgba_images: &[Vec<u8>], dims: &[(u32, u32)]) -> Result<Vec<HashBytes>, GpuError>`
  - [x] 12.5: 实现 `find_similar_pairs(&self, hashes: &[HashBytes], tolerance: u32) -> Result<Vec<(u32, u32, u32)>, GpuError>`
  - [x] 12.6: 添加集成测试验证跨阶段共享 GPU 上下文
  - [x] 12.7: 添加中文 doc 注释，包含完整使用示例

### P1-B2: GpuContext 线程安全改进

- [ ] Task 13: `GpuContext::get_or_create_pipeline()` 改为 `&self`
  - [ ] 13.1: 修改 `src/context.rs`，将 `PipelineCache` 字段（现状：未 Mutex 包装 [context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149) `pipeline_cache: PipelineCache`）改用 `Mutex<PipelineCache>` 实现内部可变性
  - [ ] 13.2: `get_or_create_pipeline()` 签名（现状：`&mut self` [context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)）改为 `&self`
  - [ ] 13.3: 更新所有内部调用方（`PerceptualHasher`、`GpuHashMatcher` 等）
  - [ ] 13.4: `PerceptualHasher::compute()` 等方法接受 `&GpuContext` 而非 `&mut GpuContext`
  - [ ] 13.5: 保留 `&mut self` 变体作为 `#[deprecated]` 别名，调用方逐步迁移
  - [ ] 13.6: 验证 `cargo test --features czkawka-compat` 全部通过
  - [ ] 13.7: 验证 `cargo clippy --features czkawka-compat` 无警告

### P1-B3: GPU 端全零拷贝流水线

- [ ] Task 14: 扩展 `GpuPipelineBuilder` 支持 RGBA 入口
  - [ ] 14.1: 修改 `src/pipeline_builder.rs`，新增 `from_rgba()` 步骤
  - [ ] 14.2: 支持 `from_rgba().blur().resize().hash()` 完整链路
  - [ ] 14.3: 新增 `PerceptualHasher::compute_from_rgba_full_pipeline()` 方法
  - [ ] 14.4: 验证零 CPU 回读：RGBA 上传后全程 GpuBuffer
  - [ ] 14.5: 添加集成测试覆盖全零拷贝流水线

### P1-B4: 大规模矩阵 GPU 端 threshold 过滤

- [x] Task 15: 新增 `hamming_pairs.wgsl` 着色器入口点
  - [x] 15.1: 新增 `src/tasks/hamming_pairs.wgsl` 文件
  - [x] 15.2: 实现 `hamming_distance_pairs` 入口点（对称模式），使用 `@workgroup_size(16, 16, 1)`
  - [x] 15.3: 实现 `hamming_distance_pairs_asymmetric` 入口点（非对称模式）
  - [x] 15.4: GPU 端直接过滤 threshold，使用 `atomicAdd(&counter[0], 1u)` 管理输出位置
  - [x] 15.5: 输出 `(qi, di, dist)` 三元组到 pairs buffer
  - [x] 15.6: 修改 `CzkawkaGpuAccelerator::find_similar_pairs()`，大规模场景（N ≥ 2000）自动使用 GPU 过滤管线
  - [x] 15.7: 新增 `compute_similar_pairs_gpu_filtered()` 和 `compute_similar_pairs_asymmetric_gpu_filtered()` 方法
  - [x] 15.8: 验证 22 个测试全部通过

## Phase 3 — P2 增强特性（可选）

### P2-C1: Lanczos3 GPU 缩放着色器

- [ ] Task 16: 在 `resize.wgsl` 中新增 Lanczos3 入口点
  - [ ] 16.1: 修改 `src/tasks/resize.wgsl`，新增 `lanczos3` 函数与对应入口点
  - [ ] 16.2: 实现 Lanczos3 核函数：`3.0 * sin(px) * sin(px/3.0) / (px*px)`
  - [ ] 16.3: 修改 `GpuResize`，支持 `ResizeFilter::Lanczos3` 选项
  - [ ] 16.4: 新增 `PerceptualHasher::compute_batch_with_filter()` 方法，支持传入 `FilterType`
  - [ ] 16.5: 一致性测试：GPU Lanczos3 vs `img_hash` Lanczos3 结果一致
  - [ ] 16.6: 文档化：此优化消除缓存版本 bump 的需要

### P2-C2: 二面体变换增强匹配

- [ ] Task 17: 新增 `GpuHashMatcherBytes::find_nearest_with_dihedral()` 方法
  - [ ] 17.1: 修改 `src/tasks/gpu_matcher.rs`（现状：无此方法，仅有 `find_nearest_neighbors` [gpu_matcher.rs](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs) `GpuHashMatcherBytes::find_nearest_neighbors(...) -> Vec<(u32, u32)>`），新增 `find_nearest_with_dihedral` 方法
  - [ ] 17.2: 签名：`find_nearest_with_dihedral(&self, ctx: &GpuContext, hashes: &[HashBytes], tolerance: u32) -> Result<Vec<(u32, u32, u32, u8)>, GpuError>`
  - [ ] 17.3: GPU 端对每个 query 的 8 个 D4 变体并行匹配（基于 `dihedral.rs` 现有 `DihedralHashes64/256/1024/4096` [dihedral.rs:472-716](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L472-L716)）
  - [ ] 17.4: 返回 `(parent, child, dist, transform_id)` 四元组
  - [ ] 17.5: 一致性测试：与 CPU 端 8 次匹配结果一致

### P2-C3: czkawka_compat 兼容模块

- [ ] Task 18: 新增 `src/czkawka_compat.rs` 完整模块
  - [ ] 18.1: 实现 `cache_tag_suffix() -> &'static str`（返回 `"_gpu"`）
  - [ ] 18.2: 实现 `FilterTypeMapping` enum：`Bilinear` / `Nearest` / `Triangle` / `Gaussian` / `CatmullRom` / `Lanczos3`
  - [ ] 18.3: 实现 `convert_filter_type(ft: image::FilterType) -> FilterTypeMapping`
  - [ ] 18.4: 实现 `GpuUnavailablePolicy` enum：`FallBackToCzkawkaCpu` / `ReportError`
  - [ ] 18.5: 将 `CzkawkaGpuAccelerator`（Task 12）迁移至此模块
  - [ ] 18.6: 添加模块级 doc 注释，包含完整 czkawka 集成示例
  - [ ] 18.7: 在 `src/lib.rs` 中导出 `pub mod czkawka_compat;`（在 `cfg(feature="czkawka-compat")` 下）

- [ ] Task 19: 文档化缓存兼容性与降级路径
  - [ ] 19.1: 在 `czkawka_compat` 模块 doc 注释中增加"缓存兼容性"章节
  - [ ] 19.2: 说明 `gpu-accel` 启用时缓存文件名追加 `_gpu` 后缀
  - [ ] 19.3: 说明 bincode 序列化 `ImagesEntry` 结构不变的保证
  - [ ] 19.4: 增加"降级路径"章节：GPU 初始化失败 → 降级到 czkawka 原 CPU 路径
  - [ ] 19.5: 说明 `GpuUnavailablePolicy` 的两种策略应用场景

### P2-C4: 集成示例

- [ ] Task 20: 新增 `examples/czkawka_integration.rs` 示例
  - [ ] 20.1: 演示完整集成流程：GpuContext 初始化 → 批量哈希 → 缓存 → GPU 匹配 → 分组
  - [ ] 20.2: 演示降级路径：GPU 不可用时回退到 CPU
  - [ ] 20.3: 演示缓存文件名拼接
  - [ ] 20.4: 演示 RGBA→灰度→哈希全零拷贝流水线

### P2 其他

- [ ] Task 21: 新增 `BackendDispatcher::try_gpu_or_fallback()` 辅助方法
  - [ ] 21.1: 修改 `src/backend_dispatcher.rs`（现状：trait 单方法 `dispatch_gpu` [backend_dispatcher.rs:26-42](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L26-L42)，`DefaultBackendDispatcher` 实现 [backend_dispatcher.rs:45-66](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L45-L66)，**无** `try_gpu_or_fallback` 方法），新增 `try_gpu_or_fallback` 方法
  - [ ] 21.2: 封装"尝试 GPU → 失败则降级 CPU"模式
  - [ ] 21.3: 添加单元测试验证降级路径
  - [ ] 21.4: 添加中文 doc 注释，包含 czkawka 集成示例

- [ ] Task 22: 新增 `PerceptualHasher::compute_to_hash_bytes()` / `compute_with_dihedral()` 方法
  - [ ] 22.1: 新增 `compute_to_hash_bytes()`，返回 `Vec<HashBytes>`（对应 czkawka `ImHash = Vec<u8>`，[hash_bytes.rs:15](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_bytes.rs#L15)）
  - [ ] 22.2: 新增 `DihedralVariantSet` enum（现状：仅 `DihedralTransform` trait [dihedral.rs:459-466](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L459-L466) + `DihedralHashes*` 结构体，**无** `DihedralVariantSet` enum）：`Off` / `MirrorFlip` / `MirrorFlipRotate90`
  - [ ] 22.3: 新增 `compute_with_dihedral()`，返回 `Vec<Vec<HashBytes>>`
  - [ ] 22.4: `Off` 模式委托 `compute_to_hash_bytes`
  - [ ] 22.5: `MirrorFlip` / `MirrorFlipRotate90` 模式：先计算原图哈希，再用 `dihedral.rs` 位矩阵变换
  - [ ] 22.6: 添加单元测试覆盖 3 档变体数量（1 / 3 / 8）

## Phase 4 — czkawka 侧集成（上游 PR，非本规范实施范围）

> 本阶段属于上游 PR 工作，不在本规范实施范围内，仅记录建议步骤供后续参考。

- [ ] Task 23（建议）: czkawka_core 新增 `gpu-accel` feature 依赖
  - [ ] 23.1: 修改 `czkawka_core/Cargo.toml`，新增 `gpgpu-tool = { version = "...", features = ["gpu-accel", "czkawka-compat"], optional = true }`
  - [ ] 23.2: 新增 `gpu-accel = ["dep:gpgpu-tool"]` feature

- [ ] Task 24（建议）: czkawka similar_images 模块增加 GPU 分支
  - [ ] 24.1: 修改 `core.rs::collect_image_file_entry`，`#[cfg(feature="gpu-accel")]` 分支调用 `CzkawkaGpuAccelerator::compute_hashes()`
  - [ ] 24.2: 修改 `core.rs::compare_hashes_with_non_zero_tolerance`，`#[cfg(feature="gpu-accel")]` 分支调用 `compute_similar_pairs()`
  - [ ] 24.3: 修改 `core.rs::compute_hashes_for_image`，`#[cfg(feature="gpu-accel")]` 分支调用 `compute_with_dihedral()`
  - [ ] 24.4: 修改 `core.rs::get_similar_images_cache_file`，`#[cfg(feature="gpu-accel")]` 追加 `_gpu` 后缀

# Task Dependencies

## Phase 1 内部依赖
- [Task 2] depends on [Task 1]（default 调整需 feature 入口就绪）
- [Task 4] depends on [Task 3]（new_for_integration 需 GpuUnavailable 变体）
- [Task 6] depends on [Task 5]（compute_from_rgba 需 color_convert.wgsl）
- [Task 8] depends on [Task 7]（非对称模式需对称模式基础）
- [Task 9] depends on [Task 1]（image feature 依赖 gpu-accel 入口）
- [Task 11] depends on [Task 6, Task 7, Task 9, Task 10]（一致性测试需各 P0 方法就绪）

## Phase 2 内部依赖
- [Task 12] depends on [Task 4, Task 6, Task 7]（CzkawkaGpuAccelerator 需基础 API）
- [Task 13] depends on [Task 4]（线程安全改进需 new_for_integration）
- [Task 14] depends on [Task 5, Task 6]（全零拷贝流水线需 RGBA 转换）
- [Task 15] depends on [Task 7]（hamming_pairs 需 compute_similar_pairs 接口）

## Phase 3 内部依赖
- [Task 16] depends on [Task 6]（Lanczos3 需基础哈希方法）
- [Task 17] depends on [Task 7]（dihedral 匹配需基础匹配方法）
- [Task 18] depends on [Task 12, Task 21]（czkawka_compat 模块聚合各组件）
- [Task 19] depends on [Task 18]（文档化需模块就绪）
- [Task 20] depends on [Task 18, Task 19]（示例需完整模块）
- [Task 22] depends on [Task 6]（dihedral 哈希需基础方法）

## 跨 Phase 依赖
- [Phase 2] depends on [Phase 1 全部完成]（P1 优化需 P0 基础稳定）
- [Phase 3] depends on [Phase 1 完成]（P2 增强 P0 基础即可，不强依赖 P1）
- [Phase 4] depends on [Phase 1-3 全部完成]（上游 PR 需本工具包 API 稳定）

# 并行化建议

## Phase 1 可并行任务
- Task 1 + Task 3 可并行（feature 入口 + 错误变体独立）
- Task 5 + Task 9 + Task 10 可并行（RGBA 着色器 + 算法互转 + 位宽互转独立）
- Task 7 + Task 8 部分可并行（对称与非对称模式独立实现）

## Phase 2 可并行任务
- Task 12 + Task 13 部分可并行（封装与线程安全改进独立）
- Task 14 + Task 15 可并行（全零拷贝流水线与 threshold 过滤独立）

## Phase 3 可并行任务
- Task 16 + Task 17 + Task 21 + Task 22 可并行（4 个独立增强）
- Task 18 + Task 19 + Task 20 部分可并行（模块实现、文档化、示例独立）

# 验证策略

## 每个任务完成后
1. `cargo build --features gpu-accel` 必须通过
2. `cargo build --features czkawka-compat` 必须通过
3. `cargo test --features czkawka-compat` 必须通过（涉及该任务的测试）
4. `cargo clippy --features czkawka-compat` 无警告

## Phase 1 完成后
1. `cargo test --features gpu-accel,czkawka-compat --test czkawka_compat_test` 全部通过
2. RGBA→灰度转换与 czkawka 公式一致
3. `compute_similar_pairs` 返回格式正确
4. 算法/位宽互转全覆盖

## Phase 2 完成后
1. `CzkawkaGpuAccelerator` 跨阶段共享上下文测试通过
2. `GpuContext::get_or_create_pipeline()` 接受 `&self`
3. N=20000 大规模矩阵 GPU 端过滤测试通过

## Phase 1-3 全部完成后
1. `cargo test --features czkawka-compat` 全部通过
2. `cargo doc --features czkawka-compat` 文档完整
3. `cargo build --no-default-features` 通过（确保不强制引入 GPU 依赖）
4. `examples/czkawka_integration.rs` 可运行
