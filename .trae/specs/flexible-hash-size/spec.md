# 灵活 Hash Size 规范

## Why

当前感知哈希体系存在三层硬编码限制：`HashAlgorithm::target_size()` 固定网格尺寸、`HashBits` 枚举仅支持 64/128/256 位、WGSL 着色器 `array<u32, 8>` 上限 256 位。作为通用组件，用户无法选择 hash_size=16/32/64 等网格尺寸，也无法获得超过 256 位的哈希输出。需要从 API 层到 WGSL 层全链路支持灵活的 hash_size 入参。

## What Changes

- **BREAKING**：`HashAlgorithm::target_size()` 不再硬编码固定值，改为根据 `hash_size` 参数动态计算
- **BREAKING**：`HashBits` 枚举替换为 `HashSize` 结构体，支持任意 2 的幂次网格尺寸（8/16/32/64）
- **BREAKING**：`PerceptualHashComputer` trait 增加 `hash_size()` 方法，`compute_sized` 参数类型变更
- **BREAKING**：`declare_phash_computer!` 宏签名变更，增加 `hash_size` 字段
- 6 个 WGSL 着色器中 `hash_u32s: array<u32, 8>` 改为动态计算上界（`array<u32, 128>` 覆盖最大 4096 bit）
- `block_hash.wgsl` 中 `blocks_x/blocks_y/block_means` 硬编码改为从 params 动态获取
- `PerceptualHasher` 构造函数增加 `hash_size` 参数，target_size 根据 hash_size 动态推导
- `compute_phash` / `compute_phash_from_gpu_buffer` 增加 `hash_size` 参数传播
- `impl_phash_computer_simple!` 宏中 `isqrt()` 推断改为使用显式 `hash_size` 推导 width/height
- `lib.rs` 公共 API 导出 `HashSize`

## Impact

- Affected specs: 感知哈希全部 6 种算法的 API 和 WGSL 着色器
- Affected code:
  - `src/tasks/hash_common.rs` — `HashBits` → `HashSize`，trait 变更，宏变更，`compute_phash*` 签名变更
  - `src/tasks/phasher.rs` — `PerceptualHasher` 构造函数、`compute()` 流程
  - `src/tasks/mean_hash.rs` 等 6 个薄包装 — 宏调用签名变更
  - `src/tasks/*.wgsl` 6 个着色器 — `hash_u32s` 数组扩容、block_hash 动态化
  - `src/lib.rs` — 导出 `HashSize`
  - `tests/*_test.rs` — 适配新 API
  - `benches/*_bench.rs` — 适配新 API

## ADDED Requirements

### Requirement: HashSize 可配置网格尺寸

系统 SHALL 提供 `HashSize` 类型替代 `HashBits`，支持 8/16/32/64 四种网格尺寸，对应哈希位宽 64/256/1024/4096 bit。

`HashSize` SHALL 存储网格边长（u32），提供 `bits()` / `u32s_per_image()` / `u64s_per_image()` / `target_size()` 等计算方法。

`HashSize` SHALL 提供 `Default` 实现（默认 8）。

#### Scenario: 用户选择 hash_size=16

- **WHEN** 用户创建 `PerceptualHasher::with_config(ctx, HashAlgorithm::Mean, false, HashSize::new(16))`
- **THEN** target_size 返回 (16, 16)，哈希输出 256 bit（4 个 u64）

#### Scenario: 用户选择 hash_size=32

- **WHEN** 用户创建 `PerceptualHasher::with_config(ctx, HashAlgorithm::Mean, false, HashSize::new(32))`
- **THEN** target_size 返回 (32, 32)，哈希输出 1024 bit（16 个 u64）

#### Scenario: 用户选择 hash_size=64

- **WHEN** 用户创建 `PerceptualHasher::with_config(ctx, HashAlgorithm::Mean, false, HashSize::new(64))`
- **THEN** target_size 返回 (64, 64)，哈希输出 4096 bit（64 个 u64）

### Requirement: 各算法根据 hash_size 动态推导 target_size

每种算法 SHALL 根据 `hash_size` 参数动态计算 target_size，而非硬编码固定值：

| 算法 | target_size 推导规则 |
|------|---------------------|
| Mean | (hash_size, hash_size) |
| Median | (hash_size, hash_size) |
| Gradient | (hash_size, hash_size + 1) |
| Block | (hash_size, hash_size) |
| VertGradient | (hash_size + 1, hash_size) |
| DoubleGradient | (hash_size + 1, hash_size + 1) |

#### Scenario: Gradient Hash hash_size=16

- **WHEN** 用户创建 Gradient Hash，hash_size=16
- **THEN** target_size = (16, 17)，有效哈希位 = 16 × 16 = 256 bit

### Requirement: WGSL 着色器支持可变 hash_size

所有 6 个 WGSL 着色器 SHALL 将 `var hash_u32s: array<u32, 8>` 替换为 `var hash_u32s: array<u32, 128>`，覆盖 hash_size=64 时的最大 4096 bit 输出。

`block_hash.wgsl` SHALL 将 `blocks_x/blocks_y` 从硬编码 8 改为从 params 动态传入，`block_means` 数组扩容为 `array<u32, 4096>`。

#### Scenario: Mean Hash hash_size=32 在 GPU 执行

- **WHEN** 用户使用 MeanHashComputer，hash_size=32，传入 32×32 图像
- **THEN** WGSL 着色器正确计算 1024 bit 哈希，无越界访问

### Requirement: PhashParams 扩展传递 hash_size

`PhashParams` uniform 结构体 SHALL 扩展为包含 `hash_size` 字段（网格边长），WGSL 着色器通过该字段动态计算 `pixels_per_image` 和 `total_bits`。

当前 `PhashParams` 为 `vec4<u32>` (image_count, width, height, hash_size_bits)。扩展为两个 vec4 或一个 struct，增加 `hash_size` 字段。

#### Scenario: PhashParams 传递 hash_size=16

- **WHEN** Rust 侧构造 PhashParams 时 hash_size=16
- **THEN** WGSL 着色器读取 hash_size=16，计算 pixels_per_image=256，total_bits=256

### Requirement: 向后兼容 HashBits

`HashBits` 枚举 SHALL 保留但标记为 `#[deprecated]`，并提供到 `HashSize` 的转换：

- `HashBits::B64` → `HashSize::new(8)`
- `HashBits::B128` → `HashSize::new(8)` (128 bit 无精确网格对应，取最近)
- `HashBits::B256` → `HashSize::new(16)`

#### Scenario: 旧代码使用 HashBits::B64

- **WHEN** 用户使用已废弃的 `HashBits::B64`
- **THEN** 编译产生 deprecation 警告，功能等价于 `HashSize::new(8)`

## MODIFIED Requirements

### Requirement: PerceptualHashComputer trait

`PerceptualHashComputer` trait 的 `compute_sized` 方法参数从 `hash_bits: HashBits` 变更为 `hash_size: HashSize`。

新增 `fn hash_size(&self) -> HashSize` 方法（替代 `fn hash_size_bits(&self) -> HashBits`）。

### Requirement: declare_phash_computer 宏

宏生成的 struct 增加 `hash_size: HashSize` 字段。`with_config` 签名变更为接受 `hash_size: HashSize` 参数。

### Requirement: PerceptualHasher 构造函数

`PerceptualHasher::new()` 默认 hash_size=8。`PerceptualHasher::with_config()` 签名变更为接受 `hash_size: HashSize`。`with_resize_mode()` 保留但默认 hash_size=8。

### Requirement: compute_phash / compute_phash_from_gpu_buffer

两个通用计算函数增加 `hash_size: HashSize` 参数，用于动态计算 target_width/target_height 和 output buffer 大小。

## REMOVED Requirements

### Requirement: HashAlgorithm::target_size() 硬编码

**Reason**: target_size 不再由算法类型唯一决定，而是由算法类型 + hash_size 共同推导。
**Migration**: 使用 `HashAlgorithm::target_size_for(hash_size: HashSize)` 替代。
