# Checklist

> **v3 修订说明**：本版本对齐 spec.md v3（代码核验版）与 tasks.md v3，主要修正：
> 1. `image_hasher` → `img_hash`（实际 dev-dependency，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）
> 2. GpuError 现状 11 变体（[error.rs:5-39](file:///d:/Code/AI/wgpu-tool/src/error.rs#L5-L39)）
> 3. 关键检查点补充代码出处，便于实施时定位验证点
>
> 验证 czkawka GPU 加速集成规范 v3 的实施完成度。
> 检查点对齐 spec.md v3 的 P0/P1/P2 优化建议与 tasks.md v3 的 4 阶段任务分解。
> 覆盖新增关键差距：RGBA→灰度转换、czkawka 兼容结果格式、GpuContext 线程安全、GPU 端 threshold 过滤。

---

## Phase 1 — P0 集成阻断项验证

### P0-F0: Feature 与错误处理基础

- [ ] `Cargo.toml` 新增 `gpu-accel = ["dep:wgpu", "image"]` feature 入口
- [ ] `Cargo.toml` 新增 `czkawka-compat = ["gpu-accel"]` feature 入口
- [ ] `default` feature 从 `["cpu-fallback"]` 改为 `[]`（**BREAKING**）
- [ ] `lib.rs` 顶部文档包含 BREAKING 变更说明与迁移指南
- [ ] `cargo build --no-default-features` 通过（不强制引入 GPU 依赖）
- [ ] `cargo build --no-default-features --features gpu-accel` 通过
- [ ] `cargo build --features czkawka-compat` 通过
- [ ] `cargo build --features cpu-fallback` 通过（恢复原默认行为）
- [ ] `examples/demo.rs` 等内部示例显式启用 `cpu-fallback`
- [ ] `GpuError::GpuUnavailable` 变体已添加，含中文 doc 注释（现状：[error.rs:5-39](file:///d:/Code/AI/wgpu-tool/src/error.rs#L5-L39) 11 变体，新增为第 12 变体）
- [ ] `GpuContext::new_for_integration()` 工厂方法已实现
- [ ] 无 GPU 适配器时返回 `GpuUnavailable`（非其他错误类型）
- [ ] `new_for_integration()` 与 `new_sync()` 区别已在 doc 注释说明
- [ ] `cargo clippy --features czkawka-compat` 无警告

### P0-A1: RGBA→灰度 GPU 转换（集成阻断项核心）

- [ ] `src/tasks/color_convert.wgsl` 文件已创建
- [ ] `rgba_to_grayscale` 入口点使用 czkakka 公式 `(R*77 + G*150 + B*29) >> 8`
- [ ] 使用 `@workgroup_size(64)` 1D 模型，每线程处理一个像素
- [ ] 输入：RGBA 数据；输出：灰度 u8（u32 打包）
- [ ] WGSL 注释说明公式来源（czkawka 兼容）
- [ ] `PerceptualHasher::compute_from_rgba()` 方法已实现（现状：`compute()` 仅接受灰度 [phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)，**无** `compute_from_rgba` 方法）
- [ ] `PerceptualHasher::compute_from_rgba_to_hash_bytes()` 方法已实现
- [ ] 内部流程：RGBA 上传 → GPU 灰度转换 → GPU blur（可选）→ GPU resize → GPU hash
- [ ] 灰度转换结果与 czkawka CPU 公式一致（位级一致或允许差异 ≤ 1bit）
- [ ] 单元测试覆盖 RGBA→灰度→哈希全流程
- [ ] 6 种算法 × 4 种位宽组合下 RGBA 输入与灰度输入产生相同哈希

### P0-A2: czkawka 兼容结果格式（集成阻断项核心）

- [ ] `GpuHashMatcherBytes::compute_similar_pairs()` 方法已实现（现状：`GpuHashMatcherFacadeBytes` 通过 trait `HashMatcherBytes::find_similar` 返回 `Vec<Vec<MatchResultBytes>>` [gpu_matcher.rs:865-977](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L865-L977)，**无** `compute_similar_pairs` 方法返回三元组）
- [ ] 返回类型为 `Vec<(u32, u32, u32)>`（parent_idx, child_idx, distance）
- [ ] 内部过滤 `distance > tolerance` 的对
- [ ] 内部过滤 `parent_idx == child_idx`（自身匹配）
- [ ] 返回值按 distance 升序排序
- [ ] 64/256/1024/4096-bit 四档位宽测试通过
- [ ] 空 hashes / 单元素 hashes 边界情况测试通过
- [ ] `GpuHashMatcherBytes::compute_similar_pairs_asymmetric()` 方法已实现
- [ ] 非对称模式签名：`(ref_hashes, normal_hashes, tolerance)`
- [ ] 对应 czkawka `gpu_compare_hashes_asymmetric()`（参考文件夹 vs 普通文件夹）
- [ ] 非对称模式单元测试覆盖
- [ ] 返回三元组格式与 czkawka `connect_results_simplified()` 输入兼容

### P0-A3: 算法枚举与位宽互转

- [ ] `HashAlgorithm::from_czkawka(name: &str)` 已实现，兼容 `"Blockhash"` 和 `"Block"`（现状：[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51) `HashAlgorithm` enum 6 变体 + Pdq feature-gated）
- [ ] `HashAlgorithm::to_czkawka_name() -> &'static str` 已实现（`Block` → `"Blockhash"`）
- [ ] `HashAlgorithm::from_img_hash_alg()` 已实现（`cfg(feature="image")` 下，对应 czkawka 实际依赖 `img_hash`，**注**：v2 误写为 `image_hasher`）
- [ ] `HashAlgorithm::to_img_hash_alg()` 已实现（`cfg(feature="image")` 下）
- [ ] `dev-dependencies` 已含 `img_hash = "3.2"`（[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)，无需新增）
- [ ] 6 种算法的字符串名 + enum 往返转换单元测试通过
- [ ] `HashSize::from_czkawka(hash_size: u8)` 已实现（现状：`HashSize::new(size: u32)` 无校验 [hash_common.rs:56-58](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L56-L58)）
- [ ] 支持 `hash_size ∈ {8, 16, 32, 64}`
- [ ] 非法值返回 `Err(GpuError::InvalidInput)`
- [ ] 4 个合法值 + 1 个非法值的单元测试通过

### P0-A4: 一致性测试套件

- [ ] `tests/czkawka_compat_test.rs` 测试文件已创建
- [ ] 测试矩阵覆盖：6 种算法 × 4 种位宽 × 3 种几何不变性档位
- [ ] 测试 1：本工具包 GPU 路径 vs 本工具包 CPU 路径
- [ ] 测试 2：本工具包 CPU 路径 vs `img_hash` crate（czkawka 实际依赖 [Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39) `img_hash = "3.2"`，**注**：v2 误写为 `image_hasher`）
- [ ] 测试 3：RGBA→灰度转换一致性（本工具包 GPU vs czkawka CPU 公式 `(R*77 + G*150 + B*29) >> 8`）
- [ ] 差异允许范围 ≤ 5%，差异原因文档化在 `czkawka_compat` 模块
- [ ] `cargo test --features gpu-accel,czkawka-compat --test czkawka_compat_test` 通过

---

## Phase 2 — P1 性能优化验证

### P1-B1: CzkawkaGpuAccelerator 跨阶段封装

- [ ] `CzkawkaGpuAccelerator` struct 已定义
- [ ] 内部封装 `Arc<Mutex<GpuContext>>` + `PerceptualHasher` + `GpuHashMatcherBytes`
- [ ] `new(hash_size: u8, hash_alg: HashAlgorithm) -> Result<Self, GpuError>` 已实现
- [ ] `compute_hashes(&self, rgba_images, dims) -> Result<Vec<HashBytes>, GpuError>` 已实现
- [ ] `find_similar_pairs(&self, hashes, tolerance) -> Result<Vec<(u32, u32, u32)>, GpuError>` 已实现
- [ ] 集成测试验证跨阶段共享 GPU 上下文（哈希阶段 + 匹配阶段共用同一 GpuContext）
- [ ] 中文 doc 注释包含完整使用示例
- [ ] GpuContext 在哈希阶段和匹配阶段之间不被重建

### P1-B2: GpuContext 线程安全改进

- [ ] `src/context.rs` 中 `PipelineCache` 改用 `Mutex<PipelineCache>` 实现内部可变性（现状：字段未 Mutex 包装 [context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149) `pipeline_cache: PipelineCache`）
- [ ] `GpuContext::get_or_create_pipeline()` 签名从 `&mut self` 改为 `&self`（现状：[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487) 接受 `&mut self`）
- [ ] 所有内部调用方已更新（`PerceptualHasher`、`GpuHashMatcher` 等）
- [ ] `PerceptualHasher::compute()` 等方法接受 `&GpuContext` 而非 `&mut GpuContext`
- [ ] 保留 `&mut self` 变体作为 `#[deprecated]` 别名（向后兼容）
- [ ] `cargo test --features czkawka-compat` 全部通过
- [ ] `cargo clippy --features czkawka-compat` 无警告
- [ ] 多线程并发调用 `get_or_create_pipeline()` 不产生数据竞争

### P1-B3: GPU 端全零拷贝流水线

- [ ] `GpuPipelineBuilder` 新增 `from_rgba()` 步骤
- [ ] 支持 `from_rgba().blur().resize().hash()` 完整链路
- [ ] `PerceptualHasher::compute_from_rgba_full_pipeline()` 方法已实现
- [ ] 验证零 CPU 回读：RGBA 上传后全程 GpuBuffer
- [ ] 集成测试覆盖全零拷贝流水线
- [ ] 中间结果（灰度、模糊、缩放）全程保持在 GPU 端

### P1-B4: 大规模矩阵 GPU 端 threshold 过滤

- [ ] `src/tasks/hamming_pairs.wgsl` 文件已创建（或在 `hamming.wgsl` 中新增入口点；现状：[hamming.wgsl](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl) 已有 3 入口点，**无** `hamming_distance_pairs`）
- [ ] `hamming_distance_pairs` 入口点使用 `@workgroup_size(16, 16, 1)`
- [ ] GPU 端直接过滤 threshold，使用 `atomicAdd(&output_count[0], 1u)` 管理输出位置
- [ ] 输出 `vec3<u32>(qi, di, dist)` 三元组到 pairs buffer
- [ ] `GpuHashMatcherBytes::compute_similar_pairs()` 大规模场景（N ≥ 阈值）自动使用此管线
- [ ] N=20000 时避免下载 1.6GB 完整矩阵（性能测试验证）
- [ ] GPU 端过滤结果与 CPU 端过滤结果一致（正确性测试）
- [ ] 输出 pairs buffer 大小有上限保护（避免 atomicAdd 溢出）

---

## Phase 3 — P2 增强特性验证

### P2-C1: Lanczos3 GPU 缩放着色器

- [ ] `src/tasks/resize.wgsl` 新增 `lanczos3` 函数与对应入口点
- [ ] Lanczos3 核函数实现正确：`3.0 * sin(px) * sin(px/3.0) / (px*px)`
- [ ] `GpuResize` 支持 `ResizeFilter::Lanczos3` 选项
- [ ] `PerceptualHasher::compute_batch_with_filter()` 方法已实现，支持传入 `FilterType`
- [ ] 一致性测试：GPU Lanczos3 vs `img_hash` Lanczos3 结果一致
- [ ] 文档说明：此优化消除缓存版本 bump 的需要

### P2-C2: 二面体变换增强匹配

- [ ] `GpuHashMatcherBytes::find_nearest_with_dihedral()` 方法已实现（现状：**无** 此方法，仅有 `find_nearest_neighbors` 返回 `Vec<(u32, u32)>`）
- [ ] 签名返回 `Vec<(u32, u32, u32, u8)>`（parent, child, dist, transform_id）
- [ ] GPU 端对每个 query 的 8 个 D4 变体并行匹配
- [ ] 一致性测试：与 CPU 端 8 次匹配结果一致
- [ ] `transform_id` 与 D4 群 8 种变换的索引对应关系文档化

### P2-C3: czkawka_compat 模块

- [ ] `src/czkawka_compat.rs` 模块文件已创建（`cfg(feature="czkawka-compat")` 下）
- [ ] `cache_tag_suffix()` 返回 `"_gpu"`
- [ ] `FilterTypeMapping` enum 已定义，含 6 种变体（Bilinear + 5 种 czkawka 滤波器）
- [ ] `convert_filter_type()` 函数已实现，映射 czkawka 5 种 FilterType
- [ ] `GpuUnavailablePolicy` enum 已定义，含 `FallBackToCzkawkaCpu` / `ReportError`
- [ ] `CzkawkaGpuAccelerator`（Task 12）已迁移至此模块
- [ ] 模块级 doc 注释包含完整 czkawka 集成示例
- [ ] `src/lib.rs` 中导出 `pub mod czkawka_compat;`（在 `cfg(feature="czkawka-compat")` 下）
- [ ] 模块 doc 包含"缓存兼容性"章节
- [ ] 模块 doc 包含"降级路径"章节
- [ ] 模块 doc 说明 bincode 序列化 `ImagesEntry` 结构不变的保证

### P2-C4: 集成示例

- [ ] `examples/czkawka_integration.rs` 示例已创建
- [ ] 演示完整集成流程：GpuContext 初始化 → 批量哈希 → 缓存 → GPU 匹配 → 分组
- [ ] 演示降级路径：GPU 不可用时回退到 CPU
- [ ] 演示缓存文件名拼接（含 `_gpu` 后缀）
- [ ] 演示 RGBA→灰度→哈希全零拷贝流水线

### P2 其他增强

- [ ] `BackendDispatcher::try_gpu_or_fallback()` 辅助方法已实现（现状：trait 单方法 `dispatch_gpu` [backend_dispatcher.rs:26-42](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L26-L42)，**无** `try_gpu_or_fallback`）
- [ ] 封装"尝试 GPU → 失败则降级 CPU"模式
- [ ] 单元测试验证降级路径
- [ ] 中文 doc 注释包含 czkawka 集成示例
- [ ] `PerceptualHasher::compute_to_hash_bytes()` 已实现，返回 `Vec<HashBytes>`（对应 czkawka `ImHash = Vec<u8>`，[hash_bytes.rs:15](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_bytes.rs#L15)）
- [ ] `DihedralVariantSet` enum 已定义，含 `Off` / `MirrorFlip` / `MirrorFlipRotate90` 三档（现状：**无** 此 enum，仅有 `DihedralTransform` trait [dihedral.rs:459-466](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L459-L466) + `DihedralHashes*` 结构体）
- [ ] `DihedralVariantSet::variant_count()` 返回 1 / 3 / 8
- [ ] `PerceptualHasher::compute_with_dihedral()` 已实现
- [ ] `Off` 模式返回每图 1 个 `HashBytes`
- [ ] `MirrorFlip` 模式返回每图 3 个 `HashBytes`
- [ ] `MirrorFlipRotate90` 模式返回每图 8 个 `HashBytes`
- [ ] 单元测试覆盖 3 档变体数量

---

## Phase 4 — czkawka 侧集成（上游 PR，非本规范实施范围）

> 本阶段属于上游 PR 工作，不在本规范实施范围内，仅记录建议步骤供后续参考。

- [ ] czkawka_core `Cargo.toml` 新增 `gpgpu-tool` 可选依赖与 `gpu-accel` feature（建议项）
- [ ] czkawka `core.rs::collect_image_file_entry` 增加 `#[cfg(feature="gpu-accel")]` 分支（建议项）
- [ ] czkawka `core.rs::compare_hashes_with_non_zero_tolerance` 增加 GPU 分支（建议项）
- [ ] czkawka `core.rs::compute_hashes_for_image` 增加 GPU 分支（建议项）
- [ ] czkawka `core.rs::get_similar_images_cache_file` 追加 `_gpu` 后缀（建议项）

---

## 整体验证（Phase 1-3 完成后）

### 构建验证

- [ ] `cargo build --no-default-features` 通过（默认不引入 GPU 依赖）
- [ ] `cargo build --features cpu-fallback` 通过（恢复原默认行为）
- [ ] `cargo build --features gpu-accel` 通过
- [ ] `cargo build --features czkawka-compat` 通过
- [ ] `cargo test --features czkawka-compat` 全部通过
- [ ] `cargo clippy --features czkawka-compat` 无警告
- [ ] `cargo doc --features czkawka-compat` 文档完整，无缺失链接

### 集成可行性验证（对照 spec.md 能力对照表）

- [ ] 6 种哈希算法与 czkawka `img_hash::HashAlg` 1:1 对应（验证互转测试，czkawka 实际依赖 `img_hash = "3.2"` [Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）
- [ ] 4 档哈希位宽与 czkawka hash_size 1:1 对应（验证 `from_czkawka` 测试）
- [ ] `HashBytes`(`Vec<u8>`) 与 czkawka `ImHash = Vec<u8>` 同签名（验证字节布局测试）
- [ ] 64/256/1024/4096-bit 四档位宽字节长度正确（8/32/128/512 字节）
- [ ] D4 群 8 种二面体变换与 czkawka `MirrorFlipRotate90` 对应
- [ ] GPU 距离矩阵 + 最近邻可替换 czkawka BK-tree
- [ ] GPU 不可用时可通过 `GpuUnavailable` 降级到 czkawka 原 CPU 路径
- [ ] 缓存文件名 `_gpu` 后缀隔离 CPU/GPU 缓存
- [ ] **RGBA→灰度转换与 czkawka CPU 公式一致**（新增阻断项验证）
- [ ] **`compute_similar_pairs` 返回 `(parent_idx, child_idx, distance)` 元组**（新增阻断项验证）
- [ ] **CzkawkaGpuAccelerator 跨阶段共享 GpuContext**（新增 P1 验证）
- [ ] **N=20000 大规模矩阵 GPU 端过滤避免 1.6GB 下载**（新增 P1 验证）

### 性能验证

- [ ] RGBA→灰度 GPU 转换 vs CPU rayon 转换：大图（≥ 4K）GPU 胜出
- [ ] 全零拷贝流水线 vs CPU 回读流水线：吞吐量提升 ≥ 2x
- [ ] N=20000 大规模矩阵：GPU 端 threshold 过滤 vs 完整下载 + CPU 过滤，时间提升 ≥ 5x
- [ ] 端到端（10K 图像）：哈希 + 匹配总耗时 ≤ czkawka CPU 路径的 1/10（28x 目标）

---

## 风险缓解验证

- [ ] GPU/CPU 哈希结果不一致风险：缓存文件名 `_gpu` 后缀隔离已实现
- [ ] GPU 不可用环境构建失败风险：`gpu-accel` 默认关闭已验证
- [ ] **RGBA→灰度公式不一致风险**：GPU 着色器公式 `(R*77 + G*150 + B*29) >> 8` 与 czkawka 一致
- [ ] **结果格式不兼容风险**：`compute_similar_pairs` 元组格式与 czkawka 期望一致
- [ ] **GpuContext 跨阶段共享风险**：`Arc<Mutex<GpuContext>>` 模式测试通过
- [ ] **大规模矩阵显存溢出风险**：GPU 端 threshold 过滤 + atomicAdd 上限保护
- [ ] 二面体变换语义差异风险：位矩阵变换 vs 像素翻转后重哈希一致性测试通过
- [ ] 缓存向后兼容风险：bincode 序列化 `ImagesEntry` 结构不变已文档化
- [ ] rayon 与 GPU 协同风险：GPU 阶段不用 rayon 已文档化
- [ ] 缩放滤镜差异风险：差异允许范围（≤ 5%）已文档化，Lanczos3 GPU 着色器（P2）可消除差异

---

## czkawka GPU 路径兼容性验证（兼容两种情况）

> spec.md 采用"兼容两种情况"策略：无论 czkawka 是否已有 wgpu_compute_engine 模块，本工具包均提供独立可用的 GPU 加速路径。

- [ ] 情况 A（czkawka 无 GPU 路径）：本工具包作为唯一 GPU 加速方案
- [ ] 情况 B（czkawka 已有 wgpu_compute_engine）：本工具包可作为替代实现，API 兼容
- [ ] 缓存版本兼容性策略：保守起见 bump 缓存版本（100 → 120 或 czkawka 实际值）
- [ ] GPU/CPU 缓存通过 `_gpu` 文件名后缀物理隔离
- [ ] `GpuUnavailable` 时调用方可选择 `FallBackToCzkawkaCpu` 或 `ReportError` 策略

---

## 文档完整性验证

- [ ] `spec.md` v3 包含完整分析报告（架构审查 + czkawka 集成可行性对照），每条"现状"声明标注代码出处 `[文件:行号]`
- [ ] `spec.md` v3 包含优化建议清单（P0/P1/P2 三档）
- [ ] `spec.md` v3 包含关键差距表（RGBA 转换、结果格式、上下文生命周期、大规模矩阵）
- [ ] `spec.md` v3 包含风险与缓解表
- [ ] `spec.md` v3 包含集成实施路径（Phase 1-4）
- [ ] `spec.md` v3 性能估算澄清：778x 加速为观测值（非测试断言），CPU 1711s 为抽样推算（非实测全量），来源 [cache_gpu_matcher_test.rs:301,311,318](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L301-L318) `eprintln!` 输出
- [ ] `spec.md` v3 已修正 v2 错误：`image_hasher` → `img_hash`、GpuError 9 变体 → 11 变体、`try_gpu_or_fallback` → `dispatch_gpu`
- [ ] `tasks.md` v3 任务拆分到最小可执行单元，含子任务（24 个任务），关键任务标注现状代码出处
- [ ] `tasks.md` v3 含任务依赖关系与并行化建议
- [ ] `checklist.md` v3 覆盖所有 Requirement 与 Scenario
- [ ] 所有新增公共 API 含中文 doc 注释
- [ ] BREAKING 变更（`default` feature 调整）有迁移指南
- [ ] `czkawka_compat` 模块 doc 注释包含完整 czkawka 集成示例
- [ ] `CzkawkaGpuAccelerator` doc 注释包含跨阶段共享上下文示例
