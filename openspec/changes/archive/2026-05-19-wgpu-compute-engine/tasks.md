## 1. 项目初始化与依赖配置

- [x] 1.1 初始化 Rust crate：`cargo init --lib`，配置 `Cargo.toml`（名称、版本、edition = "2021"）
- [x] 1.2 添加核心依赖：`wgpu`、`pollster`、`bytemuck`、`thiserror`、`log`
- [x] 1.3 添加开发依赖：`sha2`（用于交叉验证）、`rand`（用于随机测试数据）、`criterion`（用于基准测试）、`env_logger`、`hex`
- [x] 1.4 配置项目目录结构：`src/`（能力层模块）、`src/tasks/`（业务层）、`tests/`（集成测试）、`examples/`（示例）

## 2. 能力层核心实现（Capability Layer）

### 2.1 错误类型定义
- [x] 2.1.1 定义 `GpuError` 枚举，覆盖 NoAdapter、DeviceRequest、MapFailed、ShaderCompile、Validation、DeviceLost、Oom 等场景
- [x] 2.1.2 实现 `std::error::Error` 与 `thiserror` 派生

### 2.2 GpuContext 设备上下文（含 PipelineCache）
- [x] 2.2.1 实现 `GpuContext` 结构体，封装 wgpu 的 Instance、Adapter、Device、Queue、Limits，以及内部 PipelineCache
- [x] 2.2.2 实现 `GpuContext::new()` 异步构造函数，自动选择 HighPerformance 适配器
- [x] 2.2.3 实现 `GpuContext::new_sync()` 同步封装（基于 pollster）
- [x] 2.2.4 实现 `GpuContext::adapter_info()` 返回适配器信息字符串
- [x] 2.2.5 实现 `GpuContext::limits()` 暴露硬件限制
- [x] 2.2.6 实现 `GpuContext::get_or_create_pipeline(wgsl, workgroup_size)` 从内部 PipelineCache 获取或创建 Arc<ComputePipeline>
- [x] 2.2.7 实现 `GpuContext::clear_pipeline_cache()` 手动清理管线缓存

### 2.3 GpuBuffer 缓冲区管理（非泛型）
- [x] 2.3.1 定义 `BufferUsage` 枚举（Storage / Uniform）
- [x] 2.3.2 实现 `GpuBuffer` 结构体，封装 wgpu Buffer 与 byte_size（非泛型）
- [x] 2.3.3 实现 `GpuBuffer::from_data()` 从 CPU 数据创建 GPU 缓冲区（内部使用 bytemuck::cast_slice）
- [x] 2.3.4 实现 `GpuBuffer::empty()` 创建指定大小的空缓冲区（用于输出）
- [x] 2.3.5 实现 `GpuBuffer::write()` 更新已有缓冲区内容
- [x] 2.3.6 实现 `GpuBuffer::download()` 同步下载（staging buffer + map_async + poll(Wait)）
- [x] 2.3.7 内部自动使用 staging buffer 进行 CPU-GPU 数据传输

### 2.4 ComputePipeline 计算管线封装
- [x] 2.4.1 实现 `ComputePipeline` 结构体，封装 `wgpu::ComputePipeline` 和 `wgpu::BindGroupLayout`
- [x] 2.4.2 实现 `ComputePipeline::create()` 底层创建方法（编译 shader、推导 layout）
- [x] 2.4.3 实现 `ComputePipeline::dispatch()` 执行计算调度（构建 bind group、编码命令、提交队列）
- [x] 2.4.4 支持自定义 entry_point（默认 "main"）

### 2.5 Chunker 通用分批工具
- [x] 2.5.1 实现 `Chunker` 结构体，持有 `wgpu::Limits` 引用
- [x] 2.5.2 实现 `Chunker::compute_batches()` 根据输入大小和 buffer 限制计算分批策略
- [x] 2.5.3 支持按 `max_storage_buffer_binding_size` 和 `max_compute_workgroups_per_dimension` 分批
- [x] 2.5.4 返回分批元数据（偏移、大小、dispatch 尺寸）供业务层迭代处理

## 3. SHA-256 业务层实现（Business Layer）

### 3.1 WGSL 着色器（单 shader + block_mode）
- [x] 3.1.1 编写 SHA-256 WGSL 着色器，支持 block_mode uniform 参数（INIT=0 / UPDATE=1）
- [x] 3.1.2 定义 bindings：messages（storage read）、hashes（storage read_write）、params（uniform，含 message_count + block_mode）
- [x] 3.1.3 实现完整的 64 轮压缩函数（含 K 表、位运算原语）
- [x] 3.1.4 INIT 模式：使用标准初始哈希值 H0-H7
- [x] 3.1.5 UPDATE 模式：从输入缓冲区读取前一轮中间哈希状态
- [x] 3.1.6 使用 `@workgroup_size(256)` 匹配 RDNA1 Wavefront 尺寸

### 3.2 Sha256Computer 业务 struct
- [x] 3.2.1 实现 `Sha256Computer` 结构体，持有 `Arc<ComputePipeline>`
- [x] 3.2.2 实现 `Sha256Computer::new(&ctx)` 从 GpuContext 内部 PipelineCache 获取或创建 pipeline
- [x] 3.2.3 实现 `Sha256Computer::compute()` 同步批量哈希接口（单 block 批量并行 + 多 block 逐条处理）
- [x] 3.2.4 处理空输入边界情况（立即返回空结果，不触发 GPU 调用）

### 3.3 单 block SHA-256（≤55 字节）
- [x] 3.3.1 CPU 侧实现 `pad_single_block()` 填充函数（FIPS 180-4 §5.1.1）
- [x] 3.3.2 单 block 批量并行：所有消息打包为一个 input buffer，一次 dispatch

### 3.4 多 block SHA-256（>55 字节）
- [x] 3.4.1 CPU 侧实现消息分块逻辑：将长消息拆分为多个 64-byte block
- [x] 3.4.2 所有轮次统一使用 UPDATE 模式，中间哈希状态通过 input buffer 传递
- [x] 3.4.3 验证多 block 结果与 `sha2` crate 一致

### 3.5 正确性验证
- [x] 3.5.1 使用 NIST CAVP 测试向量验证（空字符串、"abc"、长消息等）
- [x] 3.5.2 使用 `sha2` crate 进行交叉验证（批量、56 字节、64 字节、1KB）
- [x] 3.5.3 测试边界：55 字节（单 block 最大）、56 字节（多 block 最小）、64 字节、1KB

## 4. 性能基准与优化

- [x] 4.1 集成 `criterion` 编写 SHA-256 基准测试，对比 GPU 与单线程 CPU 实现
- [x] 4.2 测试大批量场景（1000/10000 条 32 字节消息），记录加速比
- [x] 4.3 测试小批量场景（1/10/100 条），评估调度开销影响
- [x] 4.4 验证 PipelineCache 缓存效果（首次编译 vs 缓存命中执行时间）
- [x] 4.5 对比单 block vs 多 block 性能差异
- [x] 4.6 根据基准结果调整 workgroup_size（256 为 RDNA1 最优）

## 5. 性能优化（基于基准测试结果）

### 5.1 BufferPool 缓冲区复用
- [x] 5.1.1 实现 `BufferPool` 结构体，采用 Size-Class 策略按 2 的幂次分档
- [x] 5.1.2 实现 `BufferPool::acquire()` 从池中获取或创建缓冲区
- [x] 5.1.3 实现 `BufferPool::release()` 归还缓冲区到池（每档上限 8 个）
- [x] 5.1.4 在 `GpuBuffer` 中添加 `from_raw()` 和 `into_raw()` 支持池化
- [x] 5.1.5 `Sha256Computer` 内部持有 `BufferPool`，单 block/多 block 路径均使用池化缓冲区

### 5.2 多 block 消息批量处理优化
- [x] 5.2.1 将多 block 消息从逐条处理改为批量收集后统一处理
- [x] 5.2.2 使用 BufferPool 复用多 block 路径的 input/output 缓冲区
- [x] 5.2.3 保持正确性：多 block 消息仍按顺序逐 block 计算（SHA-256 数据依赖）

### 5.3 异步批量提交 API（能力层，算法无关）
- [x] 5.3.1 设计通用 `GpuBatchSubmitter` 结构体，不绑定具体算法
- [x] 5.3.2 实现 `BatchJob` 描述任意计算任务（input/output/params/pipeline/dispatch）
- [x] 5.3.3 实现 `GpuBatchSubmitter::submit()` 非阻塞提交（仅编码命令，不创建 staging）
- [x] 5.3.4 实现 `GpuBatchSubmitter::wait_all()` 统一等待并读取所有结果
- [x] 5.3.5 在 `Sha256Computer` 中实现 `batch_submitter()` 便捷接口
- [x] 5.3.6 实现 `Sha256BatchSubmitter::submit()` 支持单 block 消息批量异步提交
- [x] 5.3.7 实现 `Sha256BatchSubmitter::wait_all()` 返回 (原始索引, 哈希值) 列表

### 5.4 优化效果验证
- [x] 5.4.1 重新运行 `cargo bench` 获取优化后数据
- [x] 5.4.2 对比优化前后：单 block 批量、多 block 批量、workgroup_size、GPU 开销拆解
- [x] 5.4.3 添加异步批量基准测试：sync_10x_single vs async_10x_single
- [x] 5.4.4 验证异步批量提交加速比：10 次单条 4.3x，100 次单条 6.3x，10 次批量 4.4x
- [x] 5.4.5 生成优化报告，记录瓶颈根因与后续优化方向

## 6. 文档与发布准备

- [x] 6.1 为公共 API 编写 rustdoc 文档（`GpuContext`、`GpuBuffer`、`ComputePipeline`、`Sha256Computer`）
- [x] 6.2 在 `README.md` 中添加项目简介、架构说明、使用示例、性能数据
- [x] 6.3 编写 `examples/demo.rs` 演示批量 SHA-256 计算
- [x] 6.4 编写基准测试 `benches/sha256_bench.rs`
- [x] 6.5 运行 `cargo clippy` 与 `cargo fmt`，确保代码风格一致
- [x] 6.6 运行 `cargo test`，确认全部通过
