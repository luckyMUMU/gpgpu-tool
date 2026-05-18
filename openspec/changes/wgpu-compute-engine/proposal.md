## Why

CPU 密集型计算任务（如密码学哈希、矩阵运算、图像处理等）在现代多核 CPU 上已接近性能瓶颈，而 GPU 具备大规模并行计算能力却未被充分利用。本项目旨在构建一个基于 wgpu 的 Rust 计算引擎，将 GPU 的并行加速能力以统一、易用的接口暴露给 CPU 端应用，使任意 Rust 程序都能通过简单的 API 调用获得 GPU 加速。选择 wgpu 作为底层，是因为它是跨平台的（支持 Vulkan/Metal/DX12/WebGPU），能覆盖桌面与 Web 环境。

## What Changes

- 新建 Rust crate `wgpu-compute-engine`，作为 GPU 计算能力的统一封装层
- 引入能力层（Capability Layer）与业务实现层（Business Layer）的分层架构：
  - 能力层：负责 wgpu 设备管理、缓冲区生命周期、管线编译缓存、通用分批等基础设施
  - 业务层：基于能力层实现具体计算任务（如 SHA-256 哈希），每个业务模块自包含（持有 Arc<ComputePipeline>）
- 实现首个业务模块 `sha256`：支持对任意长度输入数据进行并行 SHA-256 哈希计算（单 block + 多 block）
- 提供同步与异步两种 API 风格，便于集成到不同应用场景
- 附带单元测试与基准测试，验证正确性与加速比

## Capabilities

### New Capabilities
- `compute-engine-core`: 核心能力层，提供 GpuContext（设备/队列/管线缓存）、GpuBuffer（缓冲区管理）、Chunker（通用分批）等通用基础设施
- `sha256-compute`: 基于核心能力层的 SHA-256 并行哈希计算业务实现，支持单 block 与多 block 消息处理

### Modified Capabilities
- 无（本项目为全新库，不涉及现有能力变更）

## Impact

- 新增 Rust crate，无现有代码破坏
- 依赖 `wgpu`、`pollster`、`bytemuck`、`thiserror`、`log` 等 crate
- 目标平台：支持 Vulkan/Metal/DX12 的桌面系统；WebAssembly 待后续扩展
- 对外暴露的 API 为 Rust 库接口，暂不提供 FFI 或 CLI
