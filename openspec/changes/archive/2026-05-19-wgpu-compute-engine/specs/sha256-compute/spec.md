## ADDED Requirements

### Requirement: 批量 SHA-256 哈希计算
SHA-256 计算模块 SHALL 基于能力层，支持对多组独立输入数据并行计算 SHA-256 哈希值。

#### Scenario: 批量哈希成功
- **WHEN** 用户提供 `Vec<Vec<u8>>` 形式的多个输入消息
- **AND** 调用 `Sha256Computer::compute(&ctx, &messages)`
- **THEN** 每个消息被分配至独立的 compute shader 调用
- **AND** 返回与输入顺序一致的 `Vec<[u8; 32]>` 哈希结果

#### Scenario: 空输入处理
- **WHEN** 用户传入空批次（`Vec::new()`）
- **THEN** 立即返回空结果向量，不触发 GPU 调用

#### Scenario: 单条消息哈希
- **WHEN** 用户调用 `Sha256Computer::compute(&ctx, &[single_input])`
- **THEN** 内部转换为单元素批次执行
- **AND** 返回 `Vec<[u8; 32]>` 包含单一哈希结果

### Requirement: 单 block 消息支持（≤55 字节）
SHA-256 计算模块 SHALL 支持长度不超过 55 字节的消息，在 CPU 侧执行标准填充后通过单轮 GPU dispatch 计算哈希。

#### Scenario: 标准长度消息
- **WHEN** 输入消息长度小于等于 55 字节（单 block 填充）
- **THEN** CPU 侧执行填充（0x80 + 长度追加，FIPS 180-4 §5.1.1）
- **AND** 使用 INIT 模式（block_mode=0）单轮 GPU dispatch 计算并输出正确哈希

#### Scenario: 空消息哈希
- **WHEN** 输入消息长度为 0
- **THEN** 按标准填充为单个 64-byte block
- **AND** 输出空字符串的 SHA-256 哈希值（e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855）

### Requirement: 多 block 消息支持（>55 字节）
SHA-256 计算模块 SHALL 支持长度超过 55 字节的消息，通过多轮 GPU dispatch 逐 block 处理，使用 block_mode uniform 参数控制处理模式。

#### Scenario: 双 block 消息
- **WHEN** 输入消息长度为 56 字节（需要 2 个 block）
- **THEN** CPU 侧拆分为 2 个 64-byte block（第一个含原始数据，第二个含填充）
- **AND** 第一轮 GPU dispatch 使用 INIT 模式（block_mode=0）计算第一个 block
- **AND** 第二轮 GPU dispatch 使用 FINAL 模式（block_mode=2）处理含填充的最后一个 block
- **AND** 最终结果与标准 SHA-256 一致

#### Scenario: 长消息多轮处理
- **WHEN** 输入消息长度为 1KB（需要多个 block）
- **THEN** CPU 侧将消息拆分为多个 64-byte block
- **AND** 第一轮使用 INIT 模式（block_mode=0）
- **AND** 中间轮使用 UPDATE 模式（block_mode=1），传递前一轮中间哈希状态
- **AND** 最终轮使用 FINAL 模式（block_mode=2），处理含填充的最后一个 block
- **AND** 最终结果与标准 SHA-256 一致

### Requirement: Sha256Computer 业务 struct
SHA-256 计算模块 SHALL 提供 `Sha256Computer` 结构体，持有 `Arc<ComputePipeline>`，通过 `GpuContext` 的 PipelineCache 获取。

#### Scenario: 创建 Sha256Computer
- **WHEN** 用户调用 `Sha256Computer::new(&ctx)`
- **THEN** 从 `ctx.get_or_create_pipeline()` 获取或创建 SHA-256 pipeline
- **AND** 返回持有 `Arc<ComputePipeline>` 的 `Sha256Computer` 实例

#### Scenario: PipelineCache 缓存效果
- **WHEN** 连续创建多个 `Sha256Computer` 实例
- **THEN** 第二个及以后的实例直接复用缓存的 pipeline
- **AND** 不触发重复着色器编译

### Requirement: 结果正确性验证
SHA-256 计算模块 SHALL 保证 GPU 计算结果与标准 SHA-256 算法（如 `sha2` crate 或已知测试向量）逐位一致。

#### Scenario: NIST 测试向量验证
- **WHEN** 使用 NIST CAVP 提供的 SHA-256 已知答案测试向量（如空字符串、"abc"、长消息等）
- **THEN** GPU 计算结果与预期哈希值完全一致

#### Scenario: 随机数据交叉验证
- **WHEN** 生成多组随机长度（0 ~ 10KB）的随机数据
- **THEN** GPU 结果与 `sha2` crate 的 CPU 实现结果逐字节相等

### Requirement: 性能基准
SHA-256 计算模块 SHALL 在批量处理场景下展现相对于单线程 CPU 实现的性能优势，具体加速比取决于 GPU 型号与批次大小。

#### Scenario: 大批量加速
- **WHEN** 批次大小 >= 1000 条消息，每条消息长度 64 字节
- **THEN** GPU 执行时间应显著低于单线程 CPU 执行时间（目标：>= 5x 加速比，视硬件而定）

#### Scenario: 小批量开销
- **WHEN** 批次大小 < 10 条消息
- **THEN** 文档明确说明此时 GPU 调度开销可能抵消加速收益，建议小批量使用 CPU 实现
