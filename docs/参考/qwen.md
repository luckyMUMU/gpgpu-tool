以下为目标库的**纯架构设计方案**，严格剥离具体代码实现，聚焦分层边界、核心抽象、数据流调度、资源策略与 RX 5700 适配要点。面向中高级工程师，可直接作为技术评审与迭代基线。

---
## 一、整体架构分层

```
┌─────────────────────────────────────────────────────────┐
│                   Public API (对外契约)                  │
│  Accelerator::execute::<Op>(input) -> Future<Output>    │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              Runtime Layer (运行时能力层)                │
│  • GpuContext      : Adapter/Device/Queue/Limits 管理   │
│  • ResourcePool    : Storage/Staging Buffer 复用分配器   │
│  • PipelineCache   : ShaderModule + ComputePipeline 缓存│
│  • CommandScheduler: Encode → Submit → Sync → Readback  │
│  • Chunker         : 超大输入分块与 VRAM 边界控制        │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              Contract Layer (内核契约层)                 │
│  trait ComputeOp {                                      │
│      type Input; type Output;                           │
│      fn shader() -> ShaderSource;                       │
│      fn bind_layouts() -> Vec<BindGroupLayoutEntry>;    │
│      fn workgroup_size() -> [u32; 3];                   │
│      fn dispatch_grid(len: usize) -> [u32; 3];          │
│      fn buffer_sizes(input: &Input) -> (u64, u64);      │
│      fn serialize(input: &Input) -> Vec<u8>;            │
│      fn deserialize(raw: &[u8], input: &Input) -> Output;│
│  }                                                      │
└───────────────┬─────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              Business Layer (业务内核层)                 │
│  • ops::sha256::Sha256Op                                │
│  • ops::regex::RegexMatchOp (规划)                      │
│  • ops::codec::H264DecodeOp (规划)                      │
│  仅依赖标准库 + bytemuck/encase，零 wgpu 依赖            │
└─────────────────────────────────────────────────────────┘
```

**设计原则**
- **单向依赖**：Business → Contract → Runtime，严禁反向或跨层调用
- **无状态内核**：`ComputeOp` 实例不持有 GPU 资源，仅描述计算特征与数据转换规则
- **运行时托管**：所有 `wgpu` 对象生命周期、同步屏障、内存复用由 Runtime 统一接管
- **异步优先**：对外暴露 `Future`，内部基于 `map_async` + 事件循环，杜绝 `poll(Wait)` 阻塞

---
## 二、核心抽象与契约设计

| 抽象 | 职责 | 关键设计点 |
|------|------|------------|
| `ComputeOp` | 算法与运行时的解耦契约 | 通过泛型关联类型定义输入输出；强制实现序列化/反序列化；暴露调度维度与 Buffer 尺寸估算 |
| `ResourcePool` | GPU 内存分配与复用 | 采用 Size-Class + Slab 策略；区分 Storage/Staging 池；支持碎片合并与水位告警 |
| `PipelineCache` | 管线编译去重 | 以 `(shader_hash, bind_layout_hash, workgroup_size)` 为 Key；支持热更新与 LRU 淘汰 |
| `Chunker` | 超大负载分片 | 根据 `max_compute_workgroups_per_dimension` 与 `max_storage_buffer_binding_size` 动态切分；支持流式提交 |
| `ExecutionGraph` (V2) | 多 Kernel 依赖调度 | DAG 描述 Kernel 间数据依赖；自动插入 Barrier 与 Copy 节点；支持异步计算队列并行 |

---
## 三、数据流与执行模型

```
CPU Input
   │
   ▼ serialize() + padding (CPU侧)
   │
   ▼ ResourcePool.acquire_storage()
   │
   ▼ Queue.write_buffer() / Staging → Storage Copy
   │
   ▼ PipelineCache.get_or_create() → BindGroup 构建
   │
   ▼ CommandEncoder.begin_compute_pass()
   │     set_pipeline / set_bind_group / dispatch_workgroups
   ▼
   ▼ Encoder.copy_buffer_to_buffer(Storage → Staging)
   │
   ▼ Queue.submit()
   │
   ▼ Staging.map_async(MapMode::Read) + oneshot 通知
   │
   ▼ deserialize() → CPU Output
   │
   ▼ ResourcePool.release() (RAII 或显式归还)
```

**调度策略**
- 默认单队列顺序提交，保证正确性与调试友好性
- 支持 `BatchExecutor` 合并多个独立 `ComputeOp` 调用至同一 CommandBuffer，降低提交开销
- 大负载自动触发 `Chunker` 分片，分片间通过 `Fence` 或 `map_async` 链式同步，避免 VRAM OOM

---
## 四、资源管理与生命周期

| 资源类型 | 分配策略 | 回收机制 | 安全边界 |
|----------|----------|----------|----------|
| Storage Buffer | Size-Class 池化分配，按需向上对齐至 256B | 显式 `release()` 或 `Drop` 钩子归还 | 严禁跨 `ComputeOp` 共享未同步 Buffer |
| Staging Buffer | 独立池，强制 `MAP_READ | COPY_DST` | 读回完成后立即归还 | `get_mapped_range()` 生命周期严格限定在闭包内 |
| ShaderModule | 首次加载编译，哈希缓存 | LRU 淘汰或手动 `invalidate()` | 编译失败直接返回 `ShaderError`，不降级 |
| ComputePipeline | 依赖 Shader + BindLayout 缓存 | 随缓存策略回收 | 布局不匹配时拒绝创建，抛出 `LayoutMismatch` |
| CommandEncoder | 每次执行按需创建，提交后丢弃 | 自动 Drop | 禁止跨线程共享 Encoder |

**内存对齐契约**
- 契约层强制要求 `serialize()` 输出满足 `std430` 对齐规则
- 提供 `AlignmentValidator` 调试钩子，在 `debug_assertions` 下校验偏移与 stride
- 推荐业务层使用 `bytemuck` / `encase` 进行布局推导，运行时不介入具体结构体定义

---
## 五、RX 5700 (RDNA1) 专项适配策略

| 维度 | 设计对策 |
|------|----------|
| **Wavefront 尺寸** | RDNA1 为 32。契约层建议 `workgroup_size` 为 32 的整数倍（推荐 256），运行时提供 `OccupancyEstimator` 告警非对齐配置 |
| **驱动后端** | 强制锁定 Vulkan Backend。Linux 环境优先探测 `radv`，若检测到 `amdvlk` 输出警告日志；Windows 依赖 Adrenalin 稳定版 |
| **硬件 Limits** | 启动时动态拉取 `adapter.limits()`，覆盖 `max_compute_workgroups_per_dimension`、`max_storage_buffer_binding_size`、`min_storage_buffer_offset_alignment`。`Chunker` 依此动态分片 |
| **寄存器与缓存** | RDNA1 VGPR 较紧张，契约层建议避免大型 `var<private>` 数组；常量表（如 SHA256 K 表）推荐走 Uniform/Storage Buffer 利用 L1 广播，而非 `const` 展开 |
| **异步计算队列** | RX 5700 支持独立 Compute Queue Family。V1 默认使用通用队列，V2 开放 `QueueSelector` 策略，允许计算与拷贝/图形队列并行 |
| **Profiling 对接** | 预留 `TimestampQueryPool` 插槽，支持导出 RGP 兼容的 `.rgp` 捕获标记；暴露 `tracing` span 覆盖 encode/submit/map 阶段 |

---
## 六、可扩展性与演进路线

| 阶段 | 能力目标 | 架构变更点 |
|------|----------|------------|
| **V1 (MVP)** | 单 Kernel 异步执行、CPU 侧预处理、基础 Buffer 复用、SHA-256 跑通 | 契约层定型；Runtime 完成单路径调度；Chunker 支持线性分片 |
| **V2** | 多 Kernel DAG 调度、GPU 侧预处理（如 Padding Kernel）、Slab 分配器、多队列异步 | 引入 `ExecutionGraph`；ResourcePool 升级；Queue 路由策略；支持跨 Op 数据零拷贝 |
| **V3** | 动态 Specialization、Auto-Tuning、RGP/TimeStamp 集成、跨厂商降级策略 | 契约层增加 `specialization_constants()`；运行时集成启发式调参；Fallback 至 CPU 或低精度管线 |

---
## 七、关键设计权衡

| 决策点 | 选择 | 理由 |
|--------|------|------|
| `wgpu` vs `vulkano`/`ash` | `wgpu` | WebGPU 标准抽象、跨平台一致、错误信息友好、Compute 支持成熟；更适合“GPU 作为通用加速器”的定位 |
| Trait 契约 vs 宏/代码生成 | Trait | 避免 proc-macro 调试黑洞；类型安全且易于单元测试；业务层可独立编译与分发 |
| CPU Padding vs GPU Padding | V1 CPU / V2 GPU | CPU 实现简单且正确性易验证；GPU Padding 需额外 Kernel 与同步，适合带宽瓶颈场景，放 V2 |
| 同步 Poll vs 异步 Map | 异步 `map_async` | 避免阻塞调用线程；符合现代异步 Rust 生态；便于集成到 tokio/async-std 服务中 |
| 静态 Limits vs 动态探测 | 动态探测 + Chunker | 硬件碎片化严重；硬编码 Limits 易导致 OOM 或 Dispatch 越界；动态分片保证可移植性 |

---
## 八、观测、调试与错误处理

- **错误分类**：`DeviceLost` / `Oom` / `Validation` / `ShaderCompile` / `LayoutMismatch` / `ExecutionTimeout`，全部实现 `std::error::Error` 与 `thiserror`
- **验证层集成**：`debug` 构建自动注入 `VK_LAYER_KHRONOS_validation` + `wgpu` 内部校验；提供 `ValidationReporter` 结构化日志
- **指标暴露**：内置 `metrics` 钩子，统计 `buffer_alloc_count`、`pipeline_cache_hit_rate`、`dispatch_latency_ms`、`vram_usage_peak`
- **安全边界**：运行时全部 `unsafe` 集中在 Buffer 映射与指针转换处，通过 `bytemuck::pod_read_unaligned` 与生命周期约束隔离；业务层零 `unsafe`

---
该方案可直接作为 crate 初始化与团队分工的基线。若需进一步输出：
- `ComputeOp` 契约的详细方法签名与泛型约束规范
- `Chunker` 分片算法与 VRAM 水位控制策略
- V2 `ExecutionGraph` 的 DAG 调度与 Barrier 插入规则
- RX 5700 RGP 捕获标记注入规范

可指定模块，我将提供对应的架构细化与接口契约定义。