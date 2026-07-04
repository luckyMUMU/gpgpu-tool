# czkawka GPU 加速集成规范（v3 — 代码核验修订版）

> **修订说明**：本版本基于对 wgpu-tool 实际源码的逐行核验（非仅依赖 `docs/参考/czkawka_integration_analysis.md` 报告），对 v2 进行修订：
> 1. **每条关于"现状"的声明均标注代码出处**（`文件:行号`），可追溯验证；
> 2. **修正** v2 中"现状"与实际代码不符的描述（如 `image_hasher` crate 实际名为 `img_hash`）；
> 3. **明确标注** 所有"待新增"项（v2 中部分描述把待新增 API 当作现状）；
> 4. **澄清** 778x 加速是观测值而非测试断言，CPU 1711s 是抽样推算而非实测全量。

## Why

当前工具包（`gpgpu-tool` v0.3.0）已具备完整的 GPU 感知哈希计算与汉明距离匹配能力（6 种算法 / 64~4096-bit / 3 管线 GPU 匹配器），但所有 API 设计、错误模型、依赖体积均面向"独立 GPU 算法库"定位，未针对集成到上游应用做适配。

目标上游 [`qarmin/czkawka`](https://github.com/qarmin/czkawka) 是 Rust 生态最活跃的重复/相似文件清理工具。经核验：

- **算法层与数据层高度对齐**（代码出处）：
  - 6 种算法 1:1（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51) `HashAlgorithm` enum 6 变体 + `Pdq` feature-gated）
  - 4 档位宽 1:1（[hash_common.rs:46-51](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L46-L51) `HashSize` 表格：8/16/32/64 → 64/256/1024/4096 bit）
  - `ImHash = Vec<u8>` 与 `HashBytes(Vec<u8>)` 同签名（[hash_bytes.rs:15](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_bytes.rs#L15) `pub struct HashBytes(Vec<u8>)`）
  - D4 群 8 种二面体变换 1:1（[dihedral.rs:472-716](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L472-L716) `DihedralHashes64/256/1024/4096` 4 个结构体各 8 字段）
- **本工具包自身已有实测验证**：[tests/cache_gpu_matcher_test.rs:294-371](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L294-L371) `test_cache_gpu_vs_cpu_performance` 使用 Czkawka 缓存数据（[tests/data/cache_similar_images_32_Gradient_Gaussian_100.json](file:///d:/Code/AI/wgpu-tool/tests/data/)，20000 条 1024-bit Gradient 哈希）测试 GPU 最近邻 vs CPU 一致性。
  > ⚠️ **数字说明**：v2 引用的 "GPU 2.2s vs CPU 1711s（778x 加速）" 是某次运行观测值，测试代码用 `Instant::now()` 实测并通过 `eprintln!` 打印（[cache_gpu_matcher_test.rs:301,311,318](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L301-L318)），**不是测试断言**。其中 CPU 1711s 是抽样 200 query 后**推算**的全量耗时（`cpu_elapsed * (n / cpu_query_count)`），非实测全量。
- **集成阻断项**已通过代码核验确认（详见"当前工具包架构审查"章节）。

本规范的目标是：**对当前工具包进行架构审查并产出面向 czkawka 集成的优化建议清单**，使其可以作为可选 `gpu-accel` feature 无侵入地接入 czkawka 的相似图像流水线（哈希计算 → 缓存 → BK-tree/距离比较 → 分组），同时保留 CPU 降级路径与缓存兼容性。

## 当前工具包架构审查（基于代码核验）

### 1. 能力层（`src/*.rs`）审查

能力层封装 wgpu 细节，是承载"GPU 类型无关"长期目标的核心资产，也是上游集成时的依赖边界。

| 模块 | 现状（代码出处） | 集成影响 |
|------|------|----------|
| `context.rs` `GpuContext` | 同步 `new_sync()`（[context.rs:207](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207)）+ 异步 `new()`（[context.rs:179](file:///d:/Code/AI/wgpu-tool/src/context.rs#L179)），三级降级链：`init_gpu()` → `init_software_adapter()` → `new_cpu_only()`（[context.rs:215/321/397](file:///d:/Code/AI/wgpu-tool/src/context.rs#L215)）；管线缓存 LRU（[context.rs:89-100](file:///d:/Code/AI/wgpu-tool/src/context.rs#L89-L100) `evict_if_needed`） | ⚠️ **`get_or_create_pipeline(&mut self, ...)`**（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)）需 `&mut self`；`PipelineCache` 字段**未** Mutex 包装（[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149) `pipeline_cache: PipelineCache`），跨线程共享不便，需改进 |
| `buffer.rs` `GpuBuffer` | `from_data` / `from_bytes_readable` / `download` / `download_batch` | ✅ API 完备 |
| `buffer_pool.rs` `BufferPool` | GpuContext 级共享池（[context.rs:150](file:///d:/Code/AI/wgpu-tool/src/context.rs#L150) `Arc<BufferPool>`），15 级分档，OOM 预检查 | ✅ 已统一 |
| `pipeline.rs` `ComputePipeline` | 支持 Push Constant + Uniform 回退，`dispatch_with_params` 统一入口，BindGroup LRU 缓存 | ✅ 已优化 |
| `batch.rs` `GpuBatchSubmitter` | 实现 Drop 安全清理，双缓冲流水线 | ✅ 适合 czkawka 按分辨率分组批处理 |
| `pipeline_builder.rs` `GpuPipelineBuilder` | 声明式 `blur().resize().hash()` 链式 API | ✅ 适合 czkawka 流水线组合（待新增 `from_rgba()` 入口） |
| `backend_dispatcher.rs` `BackendDispatcher` | trait 单方法 `dispatch_gpu`（[backend_dispatcher.rs:26-42](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L26-L42)），`DefaultBackendDispatcher` 实现（[backend_dispatcher.rs:45-66](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L45-L66)） | ⚠️ 待新增 `try_gpu_or_fallback()` 辅助方法 |
| `error.rs` `GpuError` | thiserror 11 变体（[error.rs:5-39](file:///d:/Code/AI/wgpu-tool/src/error.rs#L5-L39)）：`NoAdapter`/`DeviceRequest`/`ShaderCompile`/`MapFailed`/`Validation`/`DeviceLost`/`Oom`/`Internal`/`InvalidInput`/`CpuFallback`/`Timeout` | ⚠️ 待新增 `GpuUnavailable` 变体（或复用 `NoAdapter`/`CpuFallback`，需评估） |

### 2. 业务层（`src/tasks/`）审查

#### 2.1 感知哈希（`phasher.rs` / `hash_common.rs` / 6 个 `*_hash.rs`）

| 维度 | 本工具包（代码出处） | czkawka | 对应度 |
|------|----------|--------------------------|--------|
| 算法枚举 | `HashAlgorithm::{Mean, Median, Gradient, Block, VertGradient, DoubleGradient, Pdq?}`（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51)，`Pdq` 为 `feature = "pdq"` gate） | `HashAlg::{Mean, Median, Gradient, Blockhash, VertGradient, DoubleGradient}` | ✅ 6 种 1:1（命名差异：`Block` vs `Blockhash`） |
| 位宽 | `HashSize::new(8/16/32/64)`（[hash_common.rs:56-58](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L56-L58)）→ 64/256/1024/4096 bit（[hash_common.rs:46-51](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L46-L51)） | `hash_size = 8/16/32/64` | ✅ 1:1 |
| **输入图像格式** | **灰度 u8**（[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665) `compute(&self, ctx: &GpuContext, images: &[Vec<u8>], dimensions: &[(u32, u32)])`；[phasher.rs:305-335](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L305-L335) `preprocess_gpu` 同样接受灰度） | **RGBA8888** | 🔴 **关键阻断项**：需 RGBA→灰度转换层（待新增 `compute_from_rgba()`） |
| 输出类型 | `Vec<u64>`（compute，[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)）/ `HashBytes`(`Vec<u8>`)（变长，[hash_bytes.rs:15](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_bytes.rs#L15)） | `ImHash = Vec<u8>` | ⚠️ czkawka 用 `Vec<u8>`，待新增 `compute_to_hash_bytes()` 主路径 |
| 缩放滤波器 | GPU 双线性插值（`GpuResize`） | 5 种 `FilterType`（Lanczos3/Nearest/Triangle/Gaussian/CatmullRom） | 🔴 **关键差异**：滤波器不同 → 哈希结果不一致 |
| 几何不变性 | `dihedral.rs` 位矩阵变换（D4 群 8 种，[dihedral.rs:472-716](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L472-L716) 4 个 `DihedralHashes*` 结构体） | 像素级翻转后重哈希（D4 群 8 种，`MirrorFlipRotate90`） | ⚠️ 数学等价但实现不同；待新增 `DihedralVariantSet` enum + `compute_with_dihedral()` 方法 |
| 批量提交 | `compute()` 接受 `&[Vec<u8>]` 批量（[phasher.rs:660](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660)），GPU 单次 dispatch 多图，自动分块（[hash_common.rs:316-324](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L316-L324) `chunks(max_batch)`） | `hasher.hash_image(img)` 单图调用，rayon 并行 | 🎯 GPU 批量是核心加速点 |
| 三阶段流水线 | preprocess（可选 blur）→ resize → hash（[phasher.rs:99-108](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L99-L108) 文档） | decode → resize → hash | 🔄 流水线结构相似 |
| `target_size_for()` | 存在（[phasher.rs:77-89](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L77-L89)），按算法返回 (w,h) | — | ✅ |

#### 2.2 GPU 匹配（`gpu_matcher.rs` / `hamming.wgsl`）

| 维度 | 本工具包（代码出处） | czkawka | 对应度 |
|------|----------|---------------------------|--------|
| 算法 | GPU 距离矩阵（16×16 workgroup）+ 最近邻（256 workgroup）+ 大哈希最近邻（32 workgroup），3 管线字段（[gpu_matcher.rs:61-65](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L61-L65)）；workgroup 常量（[gpu_matcher.rs:33-35](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L33-L35)）；WGSL 3 入口点（[hamming.wgsl:75,140,232](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl#L75)） | BK-tree 最近邻 + 容差剪枝；GPU 路径 `gpu_compare_hashes_auto/asymmetric` | 🔄 互补 |
| 输入类型 | `&[u64]`（64-bit，[gpu_matcher.rs:144-149](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L144-L149) `compute_distance_matrix`）/ `&[HashBytes]`（变长，[gpu_matcher.rs:627+](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L627) `GpuHashMatcherBytes::compute_distance_matrix`） | `&[Vec<u8>]`（ImHash） | ✅ HashBytes 路径对应 |
| **输出格式** | `Vec<Vec<MatchResultBytes>>`（[gpu_matcher.rs:865-977](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L865-L977) `GpuHashMatcherFacadeBytes` 通过 trait `HashMatcherBytes::find_similar`）/ `Vec<Vec<u32>>` 距离矩阵 | **`Vec<(parent_idx, child_idx, distance)>`**（候选对三元组） | 🔴 **需适配层**：待新增 `compute_similar_pairs() -> Vec<(u32, u32, u32)>` |
| 阈值 | `threshold: u32`（绝对 Hamming 距离，[gpu_matcher.rs:151](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl#L151) `params.w = threshold`） | `tolerance: u32`（绝对 Hamming 距离） | ✅ 语义一致 |
| 自动管线选择 | `LARGE_HASH_THRESHOLD = 32`（[gpu_matcher.rs:50](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L50)）；`u32_per_hash ≤ 32` 标准管线 / `> 32` 大哈希管线 | 无（BK-tree 统一） | ✅ 本工具包已处理 |
| **大规模处理** | 已有分块处理（[gpu_matcher.rs:172-195](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L172-L195) 按 database 分块合并），但完整矩阵需下载到 CPU | `connect_results_simplified` 贪心分组 + 重父逻辑 | ⚠️ N=20000 时矩阵 1.6GB 超显存，需 GPU 端 threshold 过滤（待新增 `hamming_pairs.wgsl`） |
| `find_nearest_neighbors()` | 存在（[gpu_matcher.rs](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs) `GpuHashMatcherBytes::find_nearest_neighbors(...) -> Vec<(u32, u32)>`） | — | ✅ |

**性能验证**（基于 [tests/cache_gpu_matcher_test.rs:294-371](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L294-L371) `test_cache_gpu_vs_cpu_performance`，Czkawka 缓存数据 20000×1024-bit Gradient 哈希）：

| 方法 | 规模 | 耗时（观测值） | 匹配数 |
|------|------|------|--------|
| CPU 线性扫描（200×20000 抽样） | 4M 比较 | 实测，非硬编码 | 200 |
| CPU 推算全量（20000×20000） | 400M 比较 | 推算（`cpu_elapsed * (n / cpu_query_count)`），非实测 | — |
| **GPU 最近邻（20000×20000）** | **400M 比较** | **实测，非硬编码** | **20000** |
| GPU 距离矩阵（500×500） | 250K 比较 | 实测 | 532 |

> ⚠️ **数字说明**：v2 引用的 "2.2s/1711s/778x" 是某次运行观测值，测试代码用 `Instant::now()` 实测后通过 `eprintln!` 打印（[cache_gpu_matcher_test.rs:301,311,318](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L301-L318)），**不是测试断言**。CPU 1711s 是抽样 200 query 后**推算**的全量耗时，非实测。本规范引用这些数字仅作集成价值参考。

#### 2.3 端到端匹配器（`gpu_image_matcher.rs`）

`GpuImageMatcher`（[gpu_image_matcher.rs:45-49](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_image_matcher.rs#L45-L49)）组合 `PerceptualHasher` + `GpuHashMatcherBytes`，提供 `find_similar(query_images, db_images, threshold) -> Vec<Vec<MatchResultBytes>>`（[gpu_image_matcher.rs:90-137](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_image_matcher.rs#L90-L137)）。现有 8 个方法：`new` / `with_hash_size` / `hasher` / `gpu_matcher` / `hash_size` / `find_similar` / `compute_distance_matrix` / `compute_hashes`。

**与 czkawka 流程的差异**：
- czkawka 是"先全部哈希 → 写缓存 → 再匹配"两阶段分离，`GpuImageMatcher` 是"图像直传匹配"一站式
- czkawka 匹配后需并查集合并重叠分组（`merge_overlapping_groups` + `connect_results_simplified` 贪心分组 + 重父逻辑），本工具包不提供分组逻辑
- czkawka 需保留 `ImagesEntry` 元数据（path/size/modified_date），本工具包只返回哈希与距离
- czkawka 需要 `(parent_idx, child_idx, distance)` 三元组格式，本工具包返回 `Vec<Vec<MatchResultBytes>>`（[gpu_image_matcher.rs:90-137](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_image_matcher.rs#L90-L137)）

**结论**：`GpuImageMatcher` 不适合直接替换 czkawka 的整个流程，应：
1. 分别使用 `PerceptualHasher` 和 `GpuHashMatcherBytes` 两个组件
2. **待新增** `compute_similar_pairs()` 方法直接输出 czkawka 兼容的三元组格式
3. 分组逻辑（`connect_results_simplified` + `merge_overlapping_groups`）保留在 czkawka 侧

### 3. 公共 API 表面审查

`lib.rs`（[lib.rs:148-345](file:///d:/Code/AI/wgpu-tool/src/lib.rs#L148-L345)）当前导出约 40+ 个符号，已覆盖 czkawka 集成所需大部分类型。**缺口**（均待新增）：

1. **无 RGBA→灰度转换 API** 🔴：czkawka 解码为 RGBA8888，本工具包 `compute()` 只接受灰度 `Vec<u8>`（[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)）
2. **无 `GpuContext` 跨阶段共享封装**：czkawka 的 `hash_images` 和 `find_similar_hashes` 是独立阶段，需共享同一 GPU 上下文；当前 `GpuContext::get_or_create_pipeline()` 需 `&mut self`（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)），`PipelineCache` 字段未 Mutex 包装（[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149)），跨线程共享不便
3. **无 czkawka 兼容的结果格式 API**：缺少 `compute_similar_pairs() -> Vec<(parent_idx, child_idx, distance)>`
4. **无"GPU 不可用"显式信号**：`new_sync()` 失败时返回 `Err(GpuError)`（[context.rs:207](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207)），现有 `NoAdapter`/`CpuFallback` 变体（[error.rs:6,34](file:///d:/Code/AI/wgpu-tool/src/error.rs#L6)）可复用但语义不够明确，czkawka 需区分"GPU 不可用 → 降级"与"GPU 出错 → 报错"
5. **无 `HashAlgorithm` 与 czkawka 互转**：czkawka 配置层用 `image_hasher::HashAlg` 或字符串名，需适配层
6. **无 `FilterType` 适配**：czkawka 5 种 `image::FilterType` 与本工具包 GPU 双线性插值不对应
7. **无大规模矩阵 GPU 端过滤**：N=20000 时矩阵 1.6GB，需 GPU 着色器端直接过滤 threshold 输出候选对
8. **无 `czkawka_compat` 模块**（[lib.rs:148-161](file:///d:/Code/AI/wgpu-tool/src/lib.rs#L148-L161) 模块声明中无此模块）；**无 `CzkawkaGpuAccelerator` 类型**

### 4. 依赖与 Feature 审查

实际 `Cargo.toml`（[Cargo.toml:24-30](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L24-L30)）：

```toml
[features]
default = ["cpu-fallback"]      # 默认开启 CPU 降级
cpu-fallback = ["dep:sha2"]
parallel-cpu = ["cpu-fallback", "dep:rayon"]
dual-encoder = []
pdq = []
image = ["dep:image"]
```

dev-dependencies（[Cargo.toml:32-41](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L32-L41)）实际使用 `img_hash = "3.2"`（**注**：v2 误写为 `image_hasher = "3.0"`，实际 crate 名为 `img_hash`）。

**问题**：
- `default = ["cpu-fallback"]` 会让 czkawka 默认引入 `sha2`，但 czkawka 相似图像模块不需要 SHA-256（czkawka 的 duplicate 模块用 `blake3`）
- 无 `gpu-accel` 入口 feature 供 czkawka 显式启用（待新增）
- `wgpu = "24"`（[Cargo.toml:13](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L13)）是重量级依赖，czkawka 需要可关闭
- 若 czkawka 已有 `wgpu_compute_engine` 模块，可能存在 wgpu 版本冲突风险

### 5. 缓存兼容性审查

czkawka 缓存文件命名包含全部影响哈希值的参数：

```
cache_similar_images_{hash_size}_{alg}_{filter}_{geom}_{version}.bin
```

实际测试使用的缓存文件（[cache_gpu_matcher_test.rs:43](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L43)）：`tests/data/cache_similar_images_32_Gradient_Gaussian_100.json`，对应 `hash_size=32`（1024-bit）/ Gradient 算法 / Gaussian 滤波器 / 版本 100。

缓存版本号存在调研差异：`docs/参考/czkawka_integration_analysis.md` 报告为 `100`，sub-agent GitHub 调研为 `120`。**本规范采取保守策略**：假设版本号会变更，集成时需 bump 或追加 `_gpu` 后缀。

**本工具包 GPU 实现的哈希结果与 `img_hash` CPU 实现可能不一致**（滤波器差异、浮点精度差异、RGBA→灰度转换精度），若直接替换会读到错误缓存。

## czkawka 集成可行性对照

### 能力对照表

| 维度 | czkawka 现状 | 本工具包能力（代码出处） | 对应度 |
|------|--------------|--------------|--------|
| 哈希算法 | 6 种 (Mean/Gradient/Blockhash/VertGradient/DoubleGradient/Median) | 完全相同 6 种（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51)） | ✅ 1:1 |
| 哈希位宽 | 64/256/1024/4096 bit (hash_size 8/16/32/64) | 完全相同 4 档（[hash_common.rs:46-51](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L46-L51)） | ✅ 1:1 |
| 哈希数据类型 | `Vec<u8>` (ImHash) | `HashBytes`(`Vec<u8>`)（[hash_bytes.rs:15](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_bytes.rs#L15)） | ✅ 同签名 |
| **输入图像格式** | **RGBA8888** | **灰度 u8**（[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)） | 🔴 **需转换层** |
| 二面体变换 | D4 群 8 种 (像素翻转后重哈希) | D4 群 8 种 (位矩阵变换，[dihedral.rs:472-716](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L472-L716)) | ✅ 数学等价 |
| 匹配算法 | BK-tree (CPU) + GPU 路径（`gpu_compare_hashes_auto/asymmetric`） | GPU 距离矩阵 + 最近邻 (3 管线，[gpu_matcher.rs:61-65](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L61-L65)) | 🔄 互补/替换 |
| **结果格式** | `Vec<(parent_idx, child_idx, distance)>` | `Vec<Vec<MatchResultBytes>>`（[gpu_image_matcher.rs:90-137](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_image_matcher.rs#L90-L137)）/ 距离矩阵 | 🔴 **需适配层** |
| 缩放滤波器 | 5 种 FilterType (Lanczos3 等) | GPU 双线性插值 | 🔴 差异 |
| 缓存格式 | bincode + JSON | — | ⚠️ 需不破坏 |
| GPU 支持 | 可能有实验性 `wgpu_compute_engine` 模块 | ✅ wgpu 全平台，3 管线成熟实现 | 🎯 集成目标 |
| CPU 降级 | ✅ 本身即 CPU | ✅ `cpu-fallback` feature（[Cargo.toml:26](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L26)） | ✅ 双向兼容 |
| 并行模型 | rayon (CPU 多核) | wgpu (GPU 千核) + rayon (CPU 兜底) | 🎯 协同 |
| 分组逻辑 | `connect_results_simplified` + `merge_overlapping_groups` 并查集 | 无 | ⚠️ 保留 czkawka 侧 |

### 集成收益预估（基于 `docs/参考/czkawka_integration_analysis.md` 附录 B）

| 场景 | CPU 基线 | GPU 加速 | 加速比 |
|------|----------|----------|--------|
| 20000 条 1024-bit 哈希最近邻搜索 | 1711s（推算，非实测） | 2.2s（观测值，非断言） | **778x**（观测值） |
| RGBA→灰度转换（10K 图像） | ~2s (rayon) | ~0.1s (GPU) | 20x |
| 缩放+哈希（10K 图像） | ~60s | ~3s (GPU 流水线) | 20x |
| **端到端（10K 图像）** | **~942s** | **~34s** | **~28x** |

> 注：图像解码仍为 CPU 瓶颈（rayon 并行），GPU 加速的是解码后的处理流水线。第一行数字基于 [tests/cache_gpu_matcher_test.rs:294-371](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L294-L371) 的观测值，非测试断言。

## What Changes（优化建议）

### P0 — 集成阻断项（必做）

#### A1. RGBA→灰度 GPU 转换（待新增）

- **新增 `color_convert.wgsl` 着色器**：在 GPU 端完成 RGBA8888→灰度转换，使用 czkawka 同款公式 `(R*77 + G*150 + B*29) >> 8` 保证结果一致
- **新增 `PerceptualHasher::compute_from_rgba()` 方法**：接受 `&[Vec<u8>]`（RGBA 数据）和 `&[(u32, u32)]`（尺寸），内部 GPU 转换 → GPU 哈希流水线。当前 `compute()` 只接受灰度（[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)）
- **新增 `PerceptualHasher::compute_from_rgba_to_hash_bytes()` 方法**：上述方法的 `HashBytes` 输出版本
- **集成到零拷贝流水线**：RGBA→灰度→blur→resize→hash 全程 GpuBuffer，零 CPU 回读

#### A2. czkawka 兼容的结果格式 API（待新增）

- **新增 `GpuHashMatcherBytes::compute_similar_pairs()` 方法**：
  ```rust
  pub fn compute_similar_pairs(
      &self, ctx: &GpuContext,
      hashes: &[HashBytes], tolerance: u32,
  ) -> Result<Vec<(u32, u32, u32)>, GpuError>  // (parent_idx, child_idx, distance)
  ```
  内部：GPU 距离矩阵 → CPU 过滤 threshold + 去除自身匹配（distance=0 或 parent_idx==child_idx）。当前 `GpuHashMatcherBytes` 仅有 `compute_distance_matrix` / `find_nearest_neighbors`（[gpu_matcher.rs:627-859](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L627-L859)），无候选对输出 API。
- **新增 `GpuHashMatcherBytes::compute_similar_pairs_asymmetric()` 方法**：非对称模式（参考文件夹 vs 普通文件夹），对应 czkawka `gpu_compare_hashes_asymmetric()`
- **预留 `hamming_pairs.wgsl` 着色器入口点接口**（P1 实现）：GPU 端直接过滤 threshold 输出候选对，使用原子计数器管理输出位置，避免下载 1.6GB 完整矩阵

#### A3. 算法枚举与位宽互转（待新增）

- **新增 `HashAlgorithm::from_czkawka(name: &str) -> Option<Self>`**：字符串名转换，兼容 `"Blockhash"` 和 `"Block"` 两种命名。当前 `HashAlgorithm` enum（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51)）无此方法。
- **新增 `HashAlgorithm::to_czkawka_name() -> &'static str`**：转换为 czkawka 名称（`Block` → `"Blockhash"`）
- **新增 `HashAlgorithm::from_image_hasher_alg()` / `to_image_hasher_alg()`**（`cfg(feature="image")` 下）：与 `img_hash::HashAlg` enum 互转（**注**：dev-dependency 实际为 `img_hash = "3.2"`，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)，非 `image_hasher`）
- **新增 `HashSize::from_czkawka(hash_size: u8) -> Result<HashSize, GpuError>`**：支持 `hash_size ∈ {8, 16, 32, 64}`。当前 `HashSize::new(size: u32)`（[hash_common.rs:56-58](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L56-L58)）无校验，需新增带校验的 czkawka 入口。

#### A4. GPU 不可用显式降级信号（待新增）

- **新增 `GpuError::GpuUnavailable` 变体**：表示"无可用 GPU 适配器"或"GPU 初始化失败但非致命"。当前 `GpuError` 已有 `NoAdapter`（[error.rs:6](file:///d:/Code/AI/wgpu-tool/src/error.rs#L6)）和 `CpuFallback`（[error.rs:34](file:///d:/Code/AI/wgpu-tool/src/error.rs#L34)）变体，可评估是否复用或新增更明确的 `GpuUnavailable`。
- **新增 `GpuContext::new_for_integration()` 工厂方法**：返回 `Result<GpuContext, GpuError>`，明确区分"GPU 不可用（应降级）"与"GPU 出错（应报错）"。当前 `new_sync()`（[context.rs:207](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207)）不区分错误类型。

### P1 — 性能优化项（重要）

#### B1. 共享 GPU 上下文的集成封装（待新增）

- **新增 `CzkawkaGpuAccelerator` struct**：封装 `Arc<Mutex<GpuContext>>` + `PerceptualHasher` + `GpuHashMatcherBytes`，允许跨哈希计算和距离比较两个阶段共享同一 GPU 上下文
  ```rust
  pub struct CzkawkaGpuAccelerator {
      ctx: Arc<Mutex<GpuContext>>,
      hasher: PerceptualHasher,
      matcher: GpuHashMatcherBytes,
  }
  impl CzkawkaGpuAccelerator {
      pub fn new(hash_size: u8, hash_alg: HashAlgorithm) -> Result<Self, GpuError>;
      pub fn compute_hashes(&self, rgba_images: &[Vec<u8>], dims: &[(u32, u32)]) -> Result<Vec<HashBytes>, GpuError>;
      pub fn find_similar_pairs(&self, hashes: &[HashBytes], tolerance: u32) -> Result<Vec<(u32, u32, u32)>, GpuError>;
  }
  ```
  当前 `GpuHashMatcherFacadeBytes::new_with_shared_ctx` 已支持 `Arc<Mutex<GpuContext>>`（[gpu_matcher.rs:865-977](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L865-L977)），可作为封装基础。

#### B2. `GpuContext` 线程安全改进（待修改）

- **将 `GpuContext::get_or_create_pipeline()` 从 `&mut self` 改为 `&self`**（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)）：内部 `PipelineCache` 字段（[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149)）改用 `Mutex<PipelineCache>` 实现内部可变性
- **影响范围**：`PerceptualHasher::compute()` 等方法已接受 `&GpuContext`（[phasher.rs:660](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660)），但构造方法 `PerceptualHasher::new()` 等仍需 `&mut GpuContext`（[phasher.rs:148](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L148)）调用 `get_or_create_pipeline`，需同步更新
- **兼容性**：保留 `&mut self` 变体作为 deprecated 别名，调用方逐步迁移

#### B3. GPU 端 RGBA→灰度→缩放→哈希全零拷贝流水线（待新增）

- **扩展 `GpuPipelineBuilder`**：新增 `from_rgba()` 入口，支持 `from_rgba().blur().resize().hash()` 完整链路
- **新增 `PerceptualHasher::compute_from_rgba_full_pipeline()` 方法**：一键完成 RGBA→灰度→blur→resize→hash

#### B4. 大规模距离矩阵 GPU 端 threshold 过滤（待新增）

- **新增 `hamming_pairs.wgsl` 着色器入口点**：
  ```wgsl
  @compute @workgroup_size(16, 16, 1)
  fn hamming_distance_pairs(...) {
      // 计算 dist
      if (dist <= threshold && qi != di) {
          let slot = atomicAdd(&output_count[0], 1u);
          pairs[slot] = vec3<u32>(qi, di, dist);
      }
  }
  ```
  当前 `hamming.wgsl` 仅有 3 个入口点（[hamming.wgsl:75,140,232](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl#L75)），无候选对输出入口。
- **收益**：N=20000 时避免下载 1.6GB 完整矩阵，仅下载满足条件的候选对（通常 << 1%）

### P2 — 增强特性（非阻断项）

#### C1. Lanczos3 GPU 缩放着色器（待新增）

- **在 `resize.wgsl` 中新增 Lanczos3 入口点**：匹配 czkawka 默认缩放滤镜，确保哈希结果一致性
  ```wgsl
  fn lanczos3(x: f32) -> f32 {
      if (x == 0.0) { return 1.0; }
      if (abs(x) >= 3.0) { return 0.0; }
      let px = 3.14159265 * x;
      return 3.0 * sin(px) * sin(px / 3.0) / (px * px);
  }
  ```
- **新增 `PerceptualHasher::compute_batch_with_filter()` 方法**：支持传入 `FilterType` 参数
- **一致性测试**：GPU Lanczos3 vs `img_hash` Lanczos3 结果一致 → 消除缓存版本 bump 的需要

#### C2. 二面体变换增强匹配（待新增）

- **新增 `GpuHashMatcherBytes::find_nearest_with_dihedral()` 方法**：GPU 匹配阶段直接处理 8 种 D4 变体，避免 CPU 端 8 次匹配
  ```rust
  pub fn find_nearest_with_dihedral(
      &self, ctx: &GpuContext,
      hashes: &[HashBytes], tolerance: u32,
  ) -> Result<Vec<(u32, u32, u32, u8)>, GpuError>  // (parent, child, dist, transform_id)
  ```
- czkawka 当前无旋转不变性匹配，本工具包 `DihedralTransform` trait（[dihedral.rs:459-466](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L459-L466)）可作为增强特性

#### C3. czkawka_compat 兼容模块（待新增）

- **新增 `src/czkawka_compat.rs` 模块**（`cfg(feature="czkawka-compat")`）：
  - `cache_tag_suffix() -> &'static str`：返回 `"_gpu"` 后缀
  - `convert_filter_type(ft: image::FilterType) -> FilterTypeMapping`：映射 5 种 FilterType
  - `GpuUnavailablePolicy` enum：`FallBackToCzkawkaCpu` / `ReportError`
  - `CzkawkaGpuAccelerator` 封装（P1 B1）
- 当前 `lib.rs` 模块声明（[lib.rs:148-161](file:///d:/Code/AI/wgpu-tool/src/lib.rs#L148-L161)）中无此模块，需新增

#### C4. 集成示例（待新增）

- **新增 `examples/czkawka_integration.rs`**：演示完整集成流程

### D. Feature 与依赖改造（待新增/修改）

- **新增 `gpu-accel` feature**：作为 czkawka 集成的入口 feature，聚合 `wgpu` + `image`，默认关闭。当前 features（[Cargo.toml:24-30](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L24-L30)）无此入口。
- **新增 `czkawka-compat` feature**：依赖 `gpu-accel`，启用 czkawka 兼容模块
- **调整 `default` feature（BREAKING）**：从 `["cpu-fallback"]` 改为 `[]`，避免 czkawka 引入不需要的 `sha2` 依赖
- **文档化 wgpu 版本冲突风险**：若 czkawka 已有 `wgpu_compute_engine` 模块，需锁定 wgpu 版本避免冲突

### E. 错误处理与降级（待新增）

- **新增 `BackendDispatcher::try_gpu_or_fallback()` 辅助方法**：封装"尝试 GPU → 失败则降级 CPU"模式。当前 `BackendDispatcher` trait 仅有 `dispatch_gpu` 方法（[backend_dispatcher.rs:26-42](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L26-L42)）。
- **文档化 czkawka 集成降级路径**：
  - GPU 初始化失败 → 降级到 czkawka 原 CPU 路径（`img_hash` + `bk-tree`），而非本工具包的 CPU 降级（避免双重 CPU 实现）
  - 若 czkawka 已有 GPU 路径，降级到 czkawka 原 GPU 路径

### F. 一致性测试（待新增）

- **新增 `tests/czkawka_compat_test.rs`**：测试矩阵 6 种算法 × 4 种位宽 × 3 种几何不变性档位
  - 本工具包 GPU 路径 vs 本工具包 CPU 路径
  - 本工具包 CPU 路径 vs `img_hash` crate（dev-dependency，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）
  - RGBA→灰度转换一致性（本工具包 GPU vs czkawka CPU 公式）
  - 差异允许范围：哈希位差异 ≤ 5%，差异原因文档化

## Impact

- **Affected specs**:
  - `gpu-pipeline-optimization`（已完成的零拷贝管线、声明式 API 是集成基础）
  - `flexible-hash-size`（已完成的 4 档位宽是集成前提）
- **Affected code（本工具包需新增/修改，均标注代码出处）**:
  - `Cargo.toml`（[Cargo.toml:24-30](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L24-L30)）— 新增 `gpu-accel` / `czkawka-compat` feature，调整 `default`
  - `src/lib.rs`（[lib.rs:148-345](file:///d:/Code/AI/wgpu-tool/src/lib.rs#L148-L345)）— 导出新增公共 API
  - `src/error.rs`（[error.rs:5-39](file:///d:/Code/AI/wgpu-tool/src/error.rs#L5-L39)）— 新增 `GpuUnavailable` 变体
  - `src/context.rs`（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487) `get_or_create_pipeline`；[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149) `pipeline_cache` 字段；[context.rs:207](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207) `new_sync`）— 新增 `new_for_integration()`；`get_or_create_pipeline()` 改为 `&self` + `Mutex<PipelineCache>`（P1 B2）
  - `src/backend_dispatcher.rs`（[backend_dispatcher.rs:26-42](file:///d:/Code/AI/wgpu-tool/src/backend_dispatcher.rs#L26-L42)）— 新增 `try_gpu_or_fallback()` 辅助方法
  - `src/tasks/phasher.rs`（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51) `HashAlgorithm`；[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665) `compute`）— 新增 `compute_from_rgba()` / `compute_from_rgba_to_hash_bytes()` / `compute_with_dihedral()` / `compute_to_hash_bytes()` / `compute_batch_with_filter()` 方法
  - `src/tasks/gpu_matcher.rs`（[gpu_matcher.rs:627-859](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L627-L859) `GpuHashMatcherBytes`）— 新增 `compute_similar_pairs()` / `compute_similar_pairs_asymmetric()` / `find_nearest_with_dihedral()` 方法
  - `src/tasks/gpu_resize.rs` / `resize.wgsl` — 新增 Lanczos3 入口点（P2 C1）
  - 新增 `src/tasks/color_convert.wgsl` — RGBA→灰度 GPU 着色器（P0 A1）
  - `src/tasks/hamming.wgsl`（[hamming.wgsl:75,140,232](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl#L75)）— 新增 `hamming_distance_pairs` 入口点（P1 B4，可在现有文件追加或新建 `hamming_pairs.wgsl`）
  - 新增 `src/czkawka_compat.rs` — 兼容模块（P2 C3）
  - 新增 `tests/czkawka_compat_test.rs` — 一致性测试套件
  - 新增 `examples/czkawka_integration.rs` — 集成示例
- **Affected code（czkawka 侧，需上游 PR）**:
  - `czkawka_core/Cargo.toml` — 新增 `gpu-accel` feature 依赖 `gpgpu-tool`
  - `czkawka_core/src/tools/similar_images/core.rs` — `collect_image_file_entry` / `compare_hashes_with_non_zero_tolerance` / `compute_hashes_for_image` / `get_similar_images_cache_file` 增加 GPU 分支
  - `czkawka_core/src/tools/similar_images/core.rs` — 缓存文件名追加 `_gpu` 后缀

## ADDED Requirements

### Requirement: `gpu-accel` Feature 入口（待新增）

系统 SHALL 新增 `gpu-accel` feature 作为 czkawka 集成的统一入口，聚合 `wgpu` + `image` 依赖，默认关闭。

`gpu-accel` feature SHALL NOT 改变现有 `default` 行为，确保非集成场景无破坏。

#### Scenario: czkawka 启用 gpu-accel

- **WHEN** czkawka 在 `Cargo.toml` 中 `gpgpu-tool = { version = "...", features = ["gpu-accel"], optional = true }`
- **THEN** 编译引入 `wgpu` + `image` 依赖
- **AND** 公共 API 暴露 `GpuContext` / `PerceptualHasher` / `GpuHashMatcherBytes` 等集成所需类型

### Requirement: RGBA→灰度 GPU 转换（P0 阻断项，待新增）

系统 SHALL 新增 `color_convert.wgsl` 着色器，在 GPU 端完成 RGBA8888→灰度转换，使用公式 `(R*77 + G*150 + B*29) >> 8`（与 czkawka 一致）。

`PerceptualHasher` SHALL 新增 `compute_from_rgba()` 和 `compute_from_rgba_to_hash_bytes()` 方法，接受 RGBA 数据，内部 GPU 转换 → GPU 哈希流水线。当前 `compute()` 只接受灰度（[phasher.rs:660-665](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660-L665)）。

#### Scenario: czkawka RGBA 图像直接传入

- **WHEN** czkawka 调用 `hasher.compute_from_rgba_to_hash_bytes(ctx, &rgba_data, &dims)`
- **THEN** GPU 端完成 RGBA→灰度→blur→resize→hash 全流水线
- **AND** 灰度转换结果与 czkawka CPU 公式 `(R*77 + G*150 + B*29) >> 8` 一致

### Requirement: czkawka 兼容结果格式（P0 阻断项，待新增）

`GpuHashMatcherBytes` SHALL 新增 `compute_similar_pairs()` 方法，返回 `Vec<(u32, u32, u32)>`（`(parent_idx, child_idx, distance)` 三元组列表）。当前 `GpuHashMatcherBytes`（[gpu_matcher.rs:627-859](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L627-L859)）仅有 `compute_distance_matrix` / `find_nearest_neighbors`。

返回结果 SHALL：
- 过滤 `distance > tolerance` 的项
- 过滤 `parent_idx == child_idx` 的自身匹配
- 按 distance 升序排序

`GpuHashMatcherBytes` SHALL 新增 `compute_similar_pairs_asymmetric()` 方法，支持非对称模式（ref_hashes 查询 normal_hashes）。

#### Scenario: 替换 czkawka gpu_compare_hashes_auto

- **WHEN** czkawka 调用 `matcher.compute_similar_pairs(ctx, &hashes, tolerance)`
- **THEN** 返回 `(parent_idx, child_idx, distance)` 三元组列表
- **AND** 语义等价于 czkawka `gpu_compare_hashes_auto()` 但使用本工具包 3 管线架构

### Requirement: `GpuError::GpuUnavailable` 显式降级信号（待新增）

`GpuError` SHALL 新增 `GpuUnavailable` 变体，表示"无可用 GPU 适配器"或"GPU 初始化失败但非致命"。当前 `GpuError` 已有 `NoAdapter`（[error.rs:6](file:///d:/Code/AI/wgpu-tool/src/error.rs#L6)）和 `CpuFallback`（[error.rs:34](file:///d:/Code/AI/wgpu-tool/src/error.rs#L34)）变体，新增 `GpuUnavailable` 用于明确集成场景的降级信号。

`GpuContext::new_for_integration()` SHALL 在 GPU 不可用时返回 `Err(GpuError::GpuUnavailable)`，而非其他错误变体。当前 `new_sync()`（[context.rs:207](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207)）不区分错误类型。

#### Scenario: 无 GPU 环境下集成

- **WHEN** czkawka 在无 GPU 环境调用 `GpuContext::new_for_integration()`
- **THEN** 返回 `Err(GpuError::GpuUnavailable)`
- **AND** czkawka 可据此降级到原 CPU 路径

### Requirement: 算法枚举与 czkawka 互转（待新增）

`HashAlgorithm`（[phasher.rs:41-51](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L41-L51)）SHALL 提供：
- `from_czkawka(name: &str) -> Option<Self>`：字符串名转换，兼容 `"Blockhash"` 和 `"Block"`
- `to_czkawka_name() -> &'static str`：转换为 czkawka 名称（`Block` → `"Blockhash"`）
- `from_image_hasher_alg()` / `to_image_hasher_alg()`（`cfg(feature="image")` 下）：与 `img_hash::HashAlg` enum 互转（dev-dependency `img_hash = "3.2"`，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）

#### Scenario: czkawka 字符串配置转换

- **WHEN** czkawka 从配置文件读取 `"Blockhash"` 字符串
- **THEN** `HashAlgorithm::from_czkawka("Blockhash")` 返回 `Some(HashAlgorithm::Block)`

### Requirement: HashSize 与 czkawka u8 互转（待新增）

`HashSize`（[hash_common.rs:53](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L53)）SHALL 提供 `from_czkawka(hash_size: u8) -> Result<HashSize, GpuError>` 方法。当前 `HashSize::new(size: u32)`（[hash_common.rs:56-58](file:///d:/Code/AI/wgpu-tool/src/tasks/hash_common.rs#L56-L58)）无校验。

支持 `hash_size ∈ {8, 16, 32, 64}`，其他值返回 `Err(GpuError::InvalidInput)`（[error.rs:31](file:///d:/Code/AI/wgpu-tool/src/error.rs#L31)）。

### Requirement: `CzkawkaGpuAccelerator` 共享上下文封装（P1，待新增）

系统 SHALL 新增 `CzkawkaGpuAccelerator` struct，封装 `Arc<Mutex<GpuContext>>` + `PerceptualHasher` + `GpuHashMatcherBytes`，允许跨哈希计算和距离比较两个阶段共享同一 GPU 上下文。当前 `GpuHashMatcherFacadeBytes::new_with_shared_ctx`（[gpu_matcher.rs:865-977](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L865-L977)）已支持 `Arc<Mutex<GpuContext>>`，可作为封装基础。

#### Scenario: 跨阶段共享 GPU 上下文

- **WHEN** czkawka 在 `hash_images` 阶段调用 `accelerator.compute_hashes()`，在 `find_similar_hashes` 阶段调用 `accelerator.find_similar_pairs()`
- **THEN** 两个阶段共享同一 `GpuContext`，管线缓存复用
- **AND** 无需重新初始化 GPU

### Requirement: `GpuContext` 线程安全改进（P1，待修改）

`GpuContext::get_or_create_pipeline()`（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)）SHALL 从 `&mut self` 改为 `&self`，内部 `PipelineCache` 字段（[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149)）改用 `Mutex<PipelineCache>` 实现内部可变性。

`PerceptualHasher::compute()` 等方法 SHALL 接受 `&GpuContext` 而非 `&mut GpuContext`（`compute()` 已是 `&GpuContext`，[phasher.rs:660](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L660)；但 `new()` 等构造方法仍需 `&mut GpuContext`，[phasher.rs:148](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L148)）。

#### Scenario: 多线程共享 GpuContext

- **WHEN** czkawka 多线程场景下共享 `&GpuContext`
- **THEN** `get_or_create_pipeline()` 通过内部 Mutex 安全访问
- **AND** 无需 `&mut self`，简化跨线程共享

### Requirement: GPU 端 threshold 过滤候选对（P1，待新增）

系统 SHALL 在 `hamming.wgsl`（现有 3 入口点 [hamming.wgsl:75,140,232](file:///d:/Code/AI/wgpu-tool/src/tasks/hamming.wgsl#L75)）中新增 `hamming_distance_pairs` 入口点，或新建 `hamming_pairs.wgsl` 文件，在 GPU 端直接过滤 threshold 输出候选对三元组，使用原子计数器管理输出位置。

#### Scenario: 大规模矩阵避免下载完整数据

- **WHEN** N=20000 的距离矩阵计算
- **THEN** GPU 端直接过滤 threshold，仅输出满足条件的候选对
- **AND** 避免下载 1.6GB 完整矩阵到 CPU

### Requirement: `PerceptualHasher::compute_with_dihedral()`（P2，待新增）

`PerceptualHasher` SHALL 新增 `compute_with_dihedral()` 方法，单次调用返回 D4 群变体哈希，对应 czkawka `compute_hashes_for_image` 几何不变性场景。当前 `PerceptualHasher`（[phasher.rs:124-273](file:///d:/Code/AI/wgpu-tool/src/tasks/phasher.rs#L124-L273)）无此方法。

`DihedralVariantSet` enum SHALL 支持 `Off` / `MirrorFlip`（3 变体）/ `MirrorFlipRotate90`（8 变体）三档。当前 `dihedral.rs` 仅有 `DihedralTransform` trait（[dihedral.rs:459-466](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L459-L466)）和 `DihedralHashes*` 结构体（[dihedral.rs:472-716](file:///d:/Code/AI/wgpu-tool/src/tasks/dihedral.rs#L472-L716)），无 `DihedralVariantSet` enum。

#### Scenario: czkawka MirrorFlipRotate90 模式

- **WHEN** czkawka 配置 `GeometricInvariance::MirrorFlipRotate90` 调用本工具包
- **THEN** `compute_with_dihedral(..., DihedralVariantSet::MirrorFlipRotate90)` 返回每图 8 个 `HashBytes`
- **AND** 8 个哈希与 czkawka CPU 实现（像素翻转后重哈希）结果一致

### Requirement: Lanczos3 GPU 缩放着色器（P2 增强，待新增）

系统 SHALL 在 `resize.wgsl` 中新增 Lanczos3 入口点，匹配 czkawka 默认缩放滤镜。

`PerceptualHasher` SHALL 新增 `compute_batch_with_filter()` 方法，支持传入 `FilterType` 参数。

#### Scenario: GPU Lanczos3 与 img_hash 一致

- **WHEN** 使用 GPU Lanczos3 缩放计算哈希
- **THEN** 结果与 `img_hash` Lanczos3 一致
- **AND** 消除缓存版本 bump 的需要

### Requirement: czkawka_compat 模块（P2，待新增）

系统 SHALL 新增 `czkawka_compat` 模块（`cfg(feature="czkawka-compat")`），提供：
- `cache_tag_suffix() -> &'static str`：返回 `"_gpu"` 后缀
- `convert_filter_type(ft: image::FilterType) -> FilterTypeMapping`
- `GpuUnavailablePolicy` enum：`FallBackToCzkawkaCpu` / `ReportError`
- `CzkawkaGpuAccelerator` 封装

当前 `lib.rs` 模块声明（[lib.rs:148-161](file:///d:/Code/AI/wgpu-tool/src/lib.rs#L148-L161)）中无此模块。

#### Scenario: 缓存文件名拼接

- **WHEN** czkawka 启用 `gpu-accel` 后调用 `get_similar_images_cache_file`
- **THEN** 文件名追加 `_gpu` 后缀，如 `cache_similar_images_8_Gradient_Lanczos3_off_120_gpu.bin`
- **AND** 与 CPU 缓存隔离

### Requirement: GPU/CPU 哈希一致性测试套件（待新增）

系统 SHALL 新增 `tests/czkawka_compat_test.rs`，对以下矩阵验证一致性：

- 6 种算法 × 4 种位宽 × 3 种几何不变性档位
- 本工具包 GPU 路径 vs 本工具包 CPU 路径
- 本工具包 CPU 路径 vs `img_hash` crate（dev-dependency，[Cargo.toml:39](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L39)）
- RGBA→灰度转换一致性（本工具包 GPU vs czkawka CPU 公式）

差异允许范围：哈希位差异 ≤ 5%，并文档化。

## MODIFIED Requirements

### Requirement: `default` Feature 调整（待修改，BREAKING）

`Cargo.toml` 的 `default` feature（[Cargo.toml:25](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L25) `default = ["cpu-fallback"]`）SHALL 调整为 `[]`（空），将 `cpu-fallback` 改为显式可选。

**BREAKING**：现有用户需显式启用 `cpu-fallback`。迁移指南：在 `Cargo.toml` 中 `gpgpu-tool = { version = "...", features = ["cpu-fallback"] }`。

### Requirement: `GpuContext` 构造方法（待新增）

`GpuContext` SHALL 保留 `new_sync()` / `new()` 不变（[context.rs:207,179](file:///d:/Code/AI/wgpu-tool/src/context.rs#L207)），新增 `new_for_integration()` 工厂方法返回 `Result`，错误细分 `GpuUnavailable` 与其他。

### Requirement: `GpuContext::get_or_create_pipeline` 签名（P1，待修改）

`get_or_create_pipeline()`（[context.rs:484-487](file:///d:/Code/AI/wgpu-tool/src/context.rs#L484-L487)）SHALL 从 `&mut self` 改为 `&self`，内部 `PipelineCache`（[context.rs:149](file:///d:/Code/AI/wgpu-tool/src/context.rs#L149)）改用 `Mutex<PipelineCache>`。

## REMOVED Requirements

### Requirement: SHA-256 在 czkawka 集成中的隐式引入

**Reason**: czkawka 相似图像模块不需要 SHA-256，`cpu-fallback` feature 引入的 `sha2` 依赖（[Cargo.toml:26](file:///d:/Code/AI/wgpu-tool/Cargo.toml#L26)）是冗余的。
**Migration**: czkawka 启用 `gpu-accel` 时不强制引入 `cpu-fallback`，仅在需要本工具包 CPU 降级时显式启用。

## 集成实施路径

### Phase 1 — P0 集成阻断项（必做）

1. 新增 `gpu-accel` / `czkawka-compat` feature 入口（Task 1）
2. 调整 `default` feature（Task 2，BREAKING）
3. 新增 `GpuError::GpuUnavailable` + `GpuContext::new_for_integration()`（Task 3-4）
4. 新增 RGBA→灰度 GPU 转换（Task 5-6，P0 A1）
5. 新增 czkawka 兼容结果格式 API（Task 7-8，P0 A2）
6. 新增算法枚举与位宽互转（Task 9-10，P0 A3）
7. 一致性测试套件（Task 11）

### Phase 2 — P1 性能优化（重要）

1. `CzkawkaGpuAccelerator` 共享上下文封装（Task 12，P1 B1）
2. `GpuContext` 线程安全改进（Task 13，P1 B2）
3. GPU 端 RGBA→灰度→缩放→哈希全零拷贝流水线（Task 14，P1 B3）
4. 大规模矩阵 GPU 端 threshold 过滤（Task 15，P1 B4）

### Phase 3 — P2 增强特性（可选）

1. Lanczos3 GPU 缩放着色器（Task 16，P2 C1）
2. 二面体变换增强匹配（Task 17，P2 C2）
3. czkawka_compat 模块（Task 18，P2 C3）
4. 集成示例（Task 19，P2 C4）

### Phase 4 — czkawka 侧集成（上游 PR，非本规范范围）

1. czkawka_core 新增 `gpu-accel` feature 依赖
2. 修改 `similar_images/core.rs` 四处增加 GPU 分支
3. 修改缓存文件名拼接

## 风险与缓解

| 风险 | 等级 | 缓解措施 |
|------|------|----------|
| GPU/CPU 哈希结果不一致 | 🔴 高 | 缓存文件名 `_gpu` 后缀隔离 + 一致性测试套件 + 差异文档化 |
| **RGBA→灰度转换精度差异** | 🔴 高 | 使用 czkawka 同款公式 `(R*77+G*150+B*29)>>8` + 一致性测试 |
| **滤波器差异导致哈希不一致** | 🔴 高 | P2 C1 实现 GPU Lanczos3，或短期 CPU 侧缩放 |
| GPU 不可用环境构建失败 | 🟡 中 | `gpu-accel` 默认关闭 + `GpuUnavailable` 显式降级 |
| 二面体变换语义差异 | 🟡 中 | 位矩阵变换 vs 像素翻转后重哈希数学等价，需测试验证 |
| 缓存向后兼容 | 🟢 低 | bincode 序列化 `ImagesEntry` 结构不变，仅文件名变化 |
| rayon 与 GPU 协同 | 🟡 中 | GPU 阶段不用 rayon，CPU 兜底分支保留 rayon |
| **大规模矩阵显存不足** | 🟡 中 | P1 B4 GPU 端 threshold 过滤 + 已有分块处理（[gpu_matcher.rs:172-195](file:///d:/Code/AI/wgpu-tool/src/tasks/gpu_matcher.rs#L172-L195)）+ OOM 预检查 |
| **wgpu 版本冲突**（若 czkawka 已有 GPU 路径） | 🟡 中 | 锁定 wgpu 版本；本工具包作为依赖 |
| **Czkawka 上游接受度** | 🟡 中 | 先作为外部 crate 集成，POC 验证后提 PR |
| WGSL 跨平台兼容性 | 🟢 低 | wgpu 跨后端测试；DX12 register pressure 已知问题 |
| **`GpuContext` 线程安全改进的兼容性** | 🟡 中 | 保留 `&mut self` 变体作为 deprecated 别名，逐步迁移 |

## 性能预估（10K 图像，1024-bit Gradient 哈希，基于 `docs/参考/czkawka_integration_analysis.md` 附录 B）

| 阶段 | CPU (rayon) | GPU (wgpu-tool) | 加速比 |
|------|-------------|-----------------|--------|
| 图像解码 | ~30s (rayon) | ~30s (rayon, 不可加速) | 1x |
| RGBA→灰度 | ~2s | ~0.1s (GPU) | 20x |
| 缩放+哈希 | ~60s | ~3s (GPU 流水线) | 20x |
| 距离比较 | ~850s | ~1.1s (GPU 最近邻) | 778x（观测值） |
| **总计** | **~942s** | **~34s** | **~28x** |

> 注：图像解码仍为 CPU 瓶颈（rayon 并行），GPU 加速的是解码后的处理流水线。小批量（<100 图像）可能因 GPU dispatch 开销（~1.6ms）而更慢，需自动回退 CPU。距离比较行的 778x 基于 [tests/cache_gpu_matcher_test.rs:294-371](file:///d:/Code/AI/wgpu-tool/tests/cache_gpu_matcher_test.rs#L294-L371) 的观测值（非测试断言），CPU 1711s 为抽样推算（非实测全量）。
