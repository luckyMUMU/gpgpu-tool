## Context

本项目为全新 Rust 库，目标是将 wgpu 的 GPU 计算能力以通用、可扩展的方式暴露给上层业务。wgpu 作为跨平台图形与计算 API，支持 Vulkan、Metal、DirectX 12 和 WebGPU，但直接使用 wgpu 编写计算任务需要处理设备初始化、着色器编译、缓冲区管理、命令编码等大量底层细节。

参考多份设计方案后，本设计采用**三层架构**：
- **能力层**（Capability）：封装 wgpu 底层，提供设备管理、缓冲区、管线缓存、分块等通用能力
- **业务层**（Business）：基于能力层实现具体算法（如 SHA-256），每个业务模块自包含
- **公共 API**：面向用户的简洁入口

使新增计算任务只需关注 WGSL 着色器逻辑与数据编排。

## Goals / Non-Goals

**Goals:**
- 提供统一的 GPU 计算运行时，隐藏 wgpu 底层复杂性
- 支持多种计算任务并行执行，充分利用 GPU 多队列与多线程能力
- 实现首个业务模块 SHA-256，验证架构可行性与性能收益
- API 设计兼顾同步（阻塞）与异步（非阻塞）两种调用模式
- 提供完善的错误处理与资源生命周期管理
- 支持任意长度输入的分块处理，突破单缓冲区大小限制

**Non-Goals:**
- 不支持图形渲染（纯计算管线）
- 不实现通用自动并行化（需业务层显式指定并行粒度）
- 不覆盖 WebAssembly 目标（后续迭代考虑）
- 不提供 FFI 或 CLI 接口（纯 Rust 库）
- V1 不实现 BufferPool、BindGroup 缓存与 ExecutionGraph（V2 规划）

## Decisions

### 1. 分层架构：能力层 + 业务层 + 公共 API
- **选择**：三层分离，能力层提供通用 GPU 能力，业务层自包含算法实现，公共 API 面向用户
- **理由**：
  - 能力层只关心"如何在 GPU 上运行计算"，业务层只关心"计算什么"
  - 新增算法（如矩阵乘法、卷积）无需改动能力层代码
  - 业务 struct 持有 pipeline 引用，自包含且易于理解（参考 mimo.md）
  - 符合开闭原则：对扩展开放，对修改封闭
- **替代方案**：单体式设计（所有逻辑在一个 crate 中）——  rejected，因为会导致代码耦合、难以扩展

### 2. 能力层内部模块划分
- **选择**：能力层细分为 `context`、`buffer`、`pipeline_cache`、`chunker` 四个核心模块
- **理由**：
  - `context` 管理设备生命周期、队列与管线缓存；`buffer` 管理显存上传下载；`pipeline_cache` 管理着色器编译缓存；`chunker` 管理超大输入分批
  - 每个模块职责单一，符合单一职责原则
  - 便于独立测试和后续替换实现
- **替代方案**：六模块划分（device/buffer/shader/pipeline/kernel/command）—— rejected，kernel/command 模块在 V1 可由业务层直接调用能力层 API 替代，过度抽象增加复杂度

### 3. GpuContext 持有 PipelineCache
- **选择**：`PipelineCache` 作为 `GpuContext` 的内部字段，用户通过 `ctx.get_or_create_pipeline()` 获取管线
- **理由**：
  - 用户只需创建 `GpuContext`，pipeline 缓存自动管理，无需额外管理 PipelineCache 生命周期
  - API 更简洁：`Sha256Computer::new(&ctx)` 即可，无需传入独立的 cache 引用
  - PipelineCache 的生命周期与 GpuContext 绑定，ctx Drop 时缓存自动清理
- **替代方案**：PipelineCache 作为独立对象 —— rejected，增加用户管理负担，且 PipelineCache 必须与 Device 同生命周期

### 4. 业务层模式：struct 持有 Arc<ComputePipeline>
- **选择**：业务 struct（如 `Sha256Computer`）持有 `Arc<ComputePipeline>`，通过 `GpuContext::get_or_create_pipeline()` 获取
- **理由**：
  - PipelineCache 内部使用 `HashMap<Key, Arc<ComputePipeline>>`，业务 struct 持有 Arc 副本实现共享所有权
  - 业务 struct Drop 时 Arc 引用计数减少，但不影响缓存中的 pipeline
  - 参考 mimo.md 的实现模式，简单直接，每个业务模块自包含
- **替代方案**：业务 struct 持有 `&ComputePipeline` 引用 —— rejected，需要生命周期标注，API 侵入性强

### 5. PipelineCache 以 shader 内容为 Key
- **选择**：`PipelineCache` 以 `(shader_source_hash, workgroup_size)` 为 key 缓存 `Arc<ComputePipeline>`
- **理由**：
  - 比单纯 `TypeId` 更精确：同一类型但不同 WGSL 源码或 workgroup_size 会生成不同 pipeline
  - 支持运行时动态着色器（如根据参数生成不同 WGSL）
  - 参考 qwen.md 的 PipelineCache 设计，但简化 key 结构
- **替代方案**：`TypeId` 作为 key —— rejected，同一类型不同参数会共享错误 pipeline
- **碰撞风险**：V1 使用 fxhash（快速，碰撞概率极低，shader 源码通常 < 10KB），V2 可升级为 SHA-256 hash

### 6. 异步映射 + pollster 顶层封装
- **选择**：能力层内部使用 `map_async` 异步映射 staging buffer，通过 pollster 在同步 API 顶层封装阻塞等待
- **理由**：
  - `map_async` 是 wgpu 的标准异步映射 API，回调触发需要 `device.poll()`
  - wgpu 当前版本中，`map_async` 回调必须通过 `device.poll()` 触发，无法完全避免 poll 调用
  - 同步 API 内部使用 `device.poll(wgpu::Maintain::Wait)` 等待映射完成（这是 wgpu 生态的通用做法）
  - 异步 API 返回 Future，用户可在 tokio/async-std 中通过 `spawn_blocking` 包装同步调用
- **替代方案**：完全杜绝 poll(Wait) —— rejected，wgpu 当前版本不支持纯事件驱动的 map_async 回调

### 7. 使用 wgpu + pollster 实现同步/异步双模式
- **选择**：底层使用 wgpu 的异步 API，通过 pollster 提供同步封装，同时暴露原生 async API
- **理由**：
  - wgpu 原生为异步设计，直接阻塞等待会损失性能
  - pollster 轻量（无额外运行时依赖），适合需要同步调用的场景
  - 暴露 async API 允许用户集成到 tokio/async-std 等运行时
- **替代方案**：强制要求 tokio 运行时 —— rejected，因为会增加不必要的依赖，且并非所有用户都需要完整异步运行时

### 8. 缓冲区管理采用 RAII 封装（非泛型）
- **选择**：由 `GpuBuffer` 包装 wgpu 的 `Buffer`（存储 `wgpu::Buffer` + `byte_size`），通过 Rust 所有权系统管理 GPU 内存生命周期
- **理由**：
  - GPU 内存是稀缺资源，必须确保及时释放
  - 非泛型设计避免泛型传染，bytemuck::cast_slice 在 from_data() 和 download() 时使用
  - 参考 mimo.md 的实现，简单直接
- **替代方案**：`GpuBuffer<T>` 泛型设计 —— rejected，泛型会传染到所有使用 GpuBuffer 的 API，增加复杂度且收益有限

### 9. Chunker 通用分批 vs 业务层分块
- **选择**：能力层 `Chunker` 只负责按 `max_storage_buffer_binding_size` 和 `max_compute_workgroups_per_dimension` 进行通用分批；SHA-256 的 64-byte block 拆分由业务层自行处理
- **理由**：
  - 通用分批（按 buffer 大小切分）是所有 GPU 计算任务的共同需求，属于能力层
  - SHA-256 的 block 拆分是算法特定的，不属于通用能力
  - 两种分块概念不应混淆：Chunker 处理"一批数据太大放不进 GPU"的问题，业务层处理"一条消息需要多轮处理"的问题
- **替代方案**：Chunker 同时处理两种分块 —— rejected，违反单一职责，且不同算法的分块逻辑完全不同

### 10. SHA-256 多 block 实现方案
- **选择**：使用单个 WGSL shader，通过 uniform 参数 `block_mode` 控制处理模式：
  - `mode=0`（INIT）：处理第一个 block，使用标准初始哈希值 H0-H7
  - `mode=1`（UPDATE）：处理中间 block，从输入缓冲区读取前一轮的中间哈希状态
  - `mode=2`（FINAL）：处理最后一个 block（含填充），输出最终哈希
- **理由**：
  - 单 shader 方案减少管线数量和编译开销
  - 通过 uniform 参数切换模式，避免创建多个 pipeline
  - CPU 侧按顺序逐 block dispatch，每轮传递中间状态
- **替代方案**：三个独立 shader（init/update/final）—— rejected，增加管线数量和管理复杂度

### 11. ComputePipeline 职责划分
- **选择**：`ComputePipeline` 负责执行（dispatch），`PipelineCache`（通过 GpuContext）负责创建和缓存
- **理由**：
  - ComputePipeline 只封装 dispatch 逻辑（构建 bind group、编码命令、提交队列）
  - PipelineCache::get_or_create() 内部调用 ComputePipeline 的底层创建方法
  - 职责清晰：PipelineCache 是工厂+缓存，ComputePipeline 是执行器
- **调用链**：`ctx.get_or_create_pipeline(wgsl, workgroup_size)` → PipelineCache 查找/创建 → 返回 `Arc<ComputePipeline>` → 业务 struct 持有并调用 `pipeline.dispatch()`

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| GPU 驱动兼容性问题导致 wgpu 初始化失败 | 提供明确的错误类型（`NoAdapter`、`DeviceLost` 等），并支持 fallback 到 CPU 实现（后续迭代） |
| SHA-256 在 GPU 上的加速比不如预期（因数据依赖） | 基准测试覆盖不同数据量（小消息 vs 大批量），文档明确说明适用场景 |
| WGSL 着色器编写复杂，调试困难 | 提供着色器编译错误透传，开发阶段支持 SPIR-V 交叉验证 |
| GPU 内存限制（缓冲区大小上限） | `Chunker` 根据 `adapter.limits()` 动态分批，避免 VRAM OOM |
| 跨平台行为差异（不同后端表现不一致） | CI 覆盖 Windows/Linux/macOS，使用 wgpu 的验证层捕获问题 |
| PipelineCache 缓存泄漏 | GpuContext Drop 时自动清理；提供 `ctx.clear_pipeline_cache()` 手动清理接口 |
| 内部异步实现中 poll(Wait) 的必要性 | wgpu 当前版本限制，同步 API 必须使用 poll(Wait)；异步用户通过 spawn_blocking 包装 |
| 多 block SHA-256 分块引入额外同步开销 | 基准测试对比单 block vs 多 block 性能，文档说明适用场景 |
| PipelineCache hash 碰撞风险 | V1 使用 fxhash（碰撞概率极低），V2 可升级为 SHA-256 hash |
| 每次 dispatch 创建新 BindGroup | V1 可接受，V2 引入 BindGroup 缓存优化 |
| 每次下载创建新 staging buffer | V1 可接受，V2 引入 BufferPool 复用 |

## Migration Plan

本项目为全新库，无迁移需求。发布流程：
1. 本地验证通过 `cargo test` 与 `cargo bench`
2. GitHub Actions CI 通过（Windows/Linux/macOS）
3. 发布至 crates.io（后续迭代）

## Open Questions

1. 是否需要支持多 GPU 设备选择？（初期单 GPU 足够，后续可扩展）
2. 异步 API 是否应集成 `tokio` 的 `spawn_blocking` 模式以避免阻塞运行时线程？
3. V2 是否引入 `ComputeOp` trait 作为更高层抽象，替代当前 struct 模式？
4. V2 是否引入 `ResourcePool`（Size-Class + Slab）替代直接分配？

## 演进路线

| 阶段 | 能力目标 | 架构变更点 |
|------|----------|------------|
| **V1 (MVP)** | 单 Kernel 执行、CPU 侧预处理、PipelineCache、Chunker 分批、SHA-256 跑通 | 能力层四模块定型；业务层 struct 模式；pollster 同步封装 |
| **V2** | 多 Kernel 链式编排、BufferPool/BindGroup 缓存、更高层 trait 抽象（ComputeOp）、多队列异步 | 引入 `GpuCommandBuffer` Builder；ResourcePool 升级；可选 ComputeOp trait |
| **V3** | 动态 Specialization、Auto-Tuning、Profiling 集成、跨厂商降级策略 | 运行时集成启发式调参；Fallback 至 CPU 实现 |
