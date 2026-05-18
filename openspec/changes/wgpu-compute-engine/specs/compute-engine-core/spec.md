## ADDED Requirements

### Requirement: GPU 设备初始化与管理
计算引擎 SHALL 提供统一的 GPU 设备初始化接口，自动适配当前平台可用的最佳后端（Vulkan/Metal/DX12）。

#### Scenario: 成功初始化
- **WHEN** 用户调用 `GpuContext::new()`
- **THEN** 引擎自动请求 HighPerformance 适配器（Adapter）并创建设备（Device）与队列（Queue）
- **AND** 返回的上下文实例可用于后续计算任务提交

#### Scenario: 无可用 GPU 设备
- **WHEN** 当前平台不存在支持计算管线的 GPU 设备
- **THEN** 返回明确的错误 `GpuError::NoAdapter`
- **AND** 错误信息包含平台与请求的后端类型

### Requirement: 硬件 Limits 动态探测
计算引擎 SHALL 在初始化时动态拉取适配器 limits，并暴露给业务层用于分批决策。

#### Scenario: 获取 Limits
- **WHEN** 用户调用 `GpuContext::limits()`
- **THEN** 返回 `wgpu::Limits`，包含 `max_storage_buffer_binding_size`、`max_compute_workgroups_per_dimension`、`min_storage_buffer_offset_alignment` 等关键限制

### Requirement: 计算管线缓存
`GpuContext` SHALL 内部持有 `PipelineCache`，以 `(shader_source_hash, workgroup_size)` 为 key 缓存 `Arc<ComputePipeline>`，避免同一 shader 的重复编译。

#### Scenario: 首次请求编译并缓存
- **WHEN** 业务层首次调用 `ctx.get_or_create_pipeline(wgsl, workgroup_size)`
- **THEN** 引擎编译 WGSL 源码并创建 `ComputePipeline`
- **AND** 将 `Arc<ComputePipeline>` 缓存至内部 PipelineCache 并返回

#### Scenario: 缓存命中跳过编译
- **WHEN** 业务层再次调用 `ctx.get_or_create_pipeline()` 请求相同 WGSL + workgroup_size 组合
- **THEN** 引擎直接从缓存返回 `Arc<ComputePipeline>` 副本
- **AND** 跳过着色器编译步骤

#### Scenario: 手动清理缓存
- **WHEN** 用户调用 `ctx.clear_pipeline_cache()`
- **THEN** 所有缓存的 pipeline 被释放
- **AND** 下次请求将重新编译

### Requirement: 计算管线封装与调度
计算引擎 SHALL 提供 `ComputePipeline` 封装，支持从 WGSL 源码创建、自动推导 bind group layout、执行 dispatch。

#### Scenario: 从 WGSL 创建管线
- **WHEN** PipelineCache 内部调用 `ComputePipeline::create(device, wgsl_source)`
- **THEN** 自动编译 WGSL、推导 bind group layout、创建计算管线
- **AND** 返回 `ComputePipeline` 实例

#### Scenario: 执行计算调度
- **WHEN** 业务层调用 `pipeline.dispatch(ctx, bindings, workgroups)`
- **THEN** 自动构建 bind group、编码 compute pass、提交队列
- **AND** GPU 开始执行计算

#### Scenario: 着色器编译失败
- **WHEN** 提供的 WGSL 源码存在语法或类型错误
- **THEN** 返回 `GpuError::ShaderCompile`
- **AND** 错误信息包含 wgpu 提供的详细编译日志

### Requirement: 缓冲区生命周期管理（非泛型）
计算引擎 SHALL 提供 `GpuBuffer` 类型（非泛型）封装 wgpu 的 `Buffer`，通过 RAII 模式确保 GPU 内存及时释放，并支持 CPU-GPU 数据双向传输。

#### Scenario: 创建并上传数据
- **WHEN** 用户调用 `GpuBuffer::from_data(ctx, data, usage)`
- **THEN** 在 GPU 上创建对应用途的缓冲区（Storage 或 Uniform）
- **AND** 将 CPU 数据上传至该缓冲区

#### Scenario: 同步读取计算结果
- **WHEN** 用户调用 `buffer.download(ctx)`
- **THEN** 使用 staging buffer + map_async + poll(Wait) 读取数据
- **AND** 返回 `Vec<u8>`

#### Scenario: 缓冲区自动释放
- **WHEN** `GpuBuffer` 实例离开作用域被 Drop
- **THEN** 底层 wgpu `Buffer` 被释放，GPU 内存归还

### Requirement: 通用分批
计算引擎 SHALL 提供 `Chunker` 工具，根据硬件 limits 动态计算分批策略，将超出缓冲区限制的输入拆分为多个批次。

#### Scenario: 按 buffer 大小分批
- **WHEN** 输入数据大小超过 `max_storage_buffer_binding_size`
- **AND** 用户调用 `Chunker::compute_batches(input_size, limits)`
- **THEN** 返回多个批次元数据（偏移、大小、dispatch 尺寸）
- **AND** 每个批次可独立提交 GPU 处理

#### Scenario: 按 dispatch 维度分批
- **WHEN** 工作组数量超过 `max_compute_workgroups_per_dimension`
- **THEN** Chunker 自动拆分为多个 dispatch

### Requirement: 错误处理与诊断
计算引擎 SHALL 定义统一的错误类型 `GpuError`，覆盖设备初始化、着色器编译、缓冲区操作、队列提交等全生命周期的失败场景。

#### Scenario: 设备丢失
- **WHEN** GPU 设备因驱动重置或超时丢失
- **THEN** 返回 `GpuError::DeviceLost`
- **AND** 错误信息包含建议的恢复操作（如重新初始化上下文）

#### Scenario: 内存不足
- **WHEN** GPU 显存不足导致缓冲区分配失败
- **THEN** 返回 `GpuError::Oom`
- **AND** 错误信息包含当前请求大小与可用限制
