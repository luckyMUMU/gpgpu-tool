# yume-pdq 调研报告：对当前库改进的参考价值分析

> 调研日期：2026-05-24
> 库版本：yume-pdq v1.1.0
> 仓库地址：https://github.com/eternal-flame-AD/yume-pdq
> 许可证：Apache-2.0

---

## 一、yume-pdq 概览

**yume-pdq** 是由 Yumechi 开发的 Rust PDQ（Facebook/Meta Perceptual Hash）算法库，面向高吞吐量图像筛查场景，优先保证低延迟与高召回率。

### 关键技术指标

| 指标 | 数据 |
|------|------|
| 哈希算法 | PDQ（256-bit，基于 DCT 频域变换） |
| CPU 单哈希耗时 | <50μs (AVX2) |
| CPU 10M 向量匹配 | ~20ms (AVX512) |
| GPU 10M 向量匹配 | ~0.79ms (RTX 4070, Vulkan) |
| 召回率 | 100%（精确线性扫描） |
| no-std | ✅ 核心零依赖 |
| WASM | ✅ 支持 |
| FFI | ✅ C/Python 绑定 |
| Rust edition | 2024 |
| SLoC | ~9,000 |

### 依赖关系

```toml
[dependencies]
const-default = "1"
generic-array = { version = "1.2", features = ["const-default"] }
num-traits = { version = "0.2", default-features = false }
zeroize = { version = "1.8", default-features = false }

# 可选依赖
wgpu = { version = "25", optional = true, features = ["vulkan", "wgsl"] }  # 仅 Vulkan 匹配
pollster = { version = "0.4", optional = true }
rug = { version = "1.27", optional = true }          # 参考实现验证
clap = { version = "4.5", optional = true }           # CLI
core_affinity = { version = "0.8", optional = true }  # HPC 核心绑定

# 平台特定
[target.'cfg(target_arch = "x86_64")'.dependencies]
cpufeatures = "0.2"  # 运行时 CPU 特性检测
```

### Feature Flags

```toml
[features]
default = ["std"]
std = ["alloc"]
alloc = ["generic-array/alloc"]
avx512 = []                          # 编译期假定 AVX512
vulkan = ["dep:wgpu", "dep:pollster"] # GPU 匹配（非哈希计算）
portable-simd = []                    # Nightly portable-simd
portable-simd-fma = ["portable-simd"]
prefer-x86-intrinsics = []           # 优先使用 x86 intrinsics
cli = ["std", "dep:clap"]
hpc = ["std", "dep:core_affinity"]
ffi = ["std"]
```

---

## 二、yume-pdq 架构深度分析

### 2.1 分层设计（核心亮点）

```
┌─────────────────────────────────────────────┐
│  应用层: CLI / FFI (C/Python) / WASM        │  (可选)
├─────────────────────────────────────────────┤
│  匹配层: CPU SIMD Matcher / Vulkan GPU      │  (可选)
│         精确线性扫描，100% 召回率            │
├─────────────────────────────────────────────┤
│  哈希层: PDQ Kernel                         │  (核心，no-std)
│         标量 / AVX2 / AVX512 / portable-simd│
├─────────────────────────────────────────────┤
│  基础层: generic-array / num-traits          │  (极简依赖)
│         cpufeatures (x86_64 运行时检测)      │
└─────────────────────────────────────────────┘
```

**关键设计决策**：GPU 仅用于**匹配**（10M+ 规模向量数据库的 Hamming 距离批量计算），**哈希计算本身完全在 CPU 上完成**。这与当前项目用 GPU 做哈希计算的思路截然相反。

### 2.2 CPU SIMD 优化策略

yume-pdq 提供 4 级 CPU 内核，通过 feature flag 和运行时检测自动选择：

| 内核 | 指令集 | 数据宽度 | 吞吐量 | 适用场景 |
|------|--------|----------|--------|----------|
| 标量内核 | 无 SIMD | 1×f32 | 基线 | 任何平台，no-std |
| AVX2 内核 | f32x8 | 256-bit | ~8x 加速 | 消费级 CPU |
| AVX512 内核 | f32x16 | 512-bit | ~16x 加速 | 服务器级 CPU (Nightly) |
| portable-simd | 编译器自适应 | 平台最优 | 平台最优 | Nightly Rust |

**优化手段**：

1. **手写 `std::arch` intrinsics**：针对 PDQ 的 127×127 DCT 维度完全特化
2. **8-lane 并行比较**：PDQ 哈希的 8 种二面体变换天然适配 AVX512 的 8×64-bit lane 并行比较
3. **无数据依赖的跳转/索引**：安全 SIMD 模式，兼容 LLVM SafeStack+CFI
4. **运行时特性检测**：`cpufeatures` crate 在 x86_64 上自动选择最优内核
5. **DCT 维度调优**：从官方 64×64 增大到 127×127，更好利用现代 CPU 向量宽度

### 2.3 PDQ 算法核心流程

```
512×512 灰度图 → DCT-II 2D 变换 (127×127) → 频域系数提取
    → 中值量化 → 256-bit 哈希 → 质量评分
```

**与当前项目算法的本质区别**：

| 维度 | PDQ (yume-pdq) | 当前项目 6 种哈希 |
|------|----------------|-------------------|
| 变换域 | **频域** (DCT) | 空间域 (像素直接比较) |
| 信息量 | 256-bit，捕捉频率特征 | 64-bit，仅捕捉空间统计 |
| 鲁棒性 | 抗压缩/裁剪/缩放/色彩调整 | 仅抗轻微亮度变化 |
| 生产验证 | Meta/NCMEC 生产环境 | 教学演示级 |

### 2.4 匹配策略：为什么不用 BK-tree？

yume-pdq 的 TECHNICAL.md 给出了详细论证，**这对当前项目有直接参考价值**：

1. **PDQ 哈希的汉明权重几乎都是 128**：中值量化导致绝大多数哈希的汉明权重恰好为 128，BK-tree 无法通过汉明权重剪枝
2. **大半径搜索退化**：阈值 31 bit 下，BK-tree 需访问 62 bit 范围，接近全量扫描
3. **8 变体天然并行**：线性扫描可同时匹配 8 个二面体变体，BK-tree 需 8 次独立搜索
4. **实测数据**：
   - Faiss BinaryHNSW：90% 召回率需 ~10ms/查询
   - AVX512 精确扫描：100% 召回率仅需 ~20ms/10M 向量
5. **可测试性**：精确匹配有明确的正确答案，ANN 的召回率难以验证
6. **安全性**：线性扫描无数据依赖索引，无越界风险

### 2.5 二面体变换（Dihedral Transforms）

PDQ 哈希不是旋转/翻转不变的，因此 yume-pdq 自动生成 8 种变体：

```
original → rotate90 → rotate180 → rotate270
         → flip_x   → flip_y    → flip_plus1 → flip_minus1
```

这在实际图像匹配中至关重要（用户上传的图片可能被旋转/翻转），而当前项目**完全缺失**此能力。

### 2.6 精度与速度的权衡

yume-pdq 故意偏离 PDQ 参考实现（DCT 维度 127×127 vs 官方 64×64），换取：

- 更好利用现代 CPU 向量宽度
- 单哈希 <50μs (AVX2)
- DISC21 测试集最差 24 bit 偏差（匹配阈值 31 bit）

**但明确声明**：优化内核的哈希**不应提交到数据库**，仅用于匹配查询。

---

## 三、与当前项目的对比分析

### 3.1 算法维度对比

| 维度 | yume-pdq | 当前项目 (wgpu-tool) |
|------|----------|---------------------|
| **算法类型** | PDQ（DCT 频域变换） | 6 种空间域简单比较 |
| **算法深度** | 专业级（Meta 生产验证） | 入门级（教学/演示级） |
| **鲁棒性** | 抗压缩/裁剪/缩放/色彩调整 | 仅抗轻微亮度变化 |
| **哈希长度** | 256 bit 固定 | 64-1024 bit 可变 |
| **旋转不变性** | ✅ 8 种二面体变体 | ❌ 无 |
| **质量评估** | ✅ quality score | ❌ 无 |

### 3.2 实现架构对比

| 维度 | yume-pdq | 当前项目 |
|------|----------|----------|
| **哈希计算位置** | CPU (SIMD) | GPU (WGSL 着色器) |
| **匹配位置** | CPU SIMD / GPU Vulkan | BK-tree (CPU) |
| **CPU fallback** | ✅ 核心就是 CPU | ❌ 无（GPU 不可用则功能全失） |
| **GPU 用途** | 仅匹配加速（可选） | 哈希计算本身（必须） |
| **no-std** | ✅ | ❌ |
| **依赖量** | 极简（核心零依赖） | 重度依赖 wgpu |
| **运行环境** | 任何 CPU / WASM | 必须有 GPU |
| **数据传输** | 无（纯 CPU） | CPU→GPU 4 倍膨胀 (u8→u32) |

### 3.3 性能对比

| 场景 | yume-pdq (AVX2) | 当前项目 (GPU) |
|------|------------------|----------------|
| 单张哈希 | <50μs | GPU 调度开销 > 计算时间 |
| 批量哈希 (1000 张) | ~50ms | ~数 ms（GPU 优势场景） |
| 10M 向量匹配 | ~20ms (AVX512) | BK-tree 理论 O(log n) 但实际退化 |
| 无 GPU 环境 | 正常运行 | **完全不可用** |

### 3.4 当前项目的性能瓶颈（yume-pdq 视角下的诊断）

| 瓶颈 | 描述 | yume-pdq 的解决方式 |
|------|------|---------------------|
| **数据传输膨胀** | u8→u32 逐像素扩展，4 倍内存膨胀 | 纯 CPU，零传输 |
| **GPU 调度开销** | 8×8 图像仅 64 像素，GPU 调度远超计算 | CPU 直接计算，微秒级 |
| **Median 寄存器压力** | WGSL 256 桶直方图占 1KB 局部存储 | CPU SIMD histogram |
| **Block 重复计算** | 同一块均值被多次重复计算 | CPU 可缓存中间结果 |
| **同步阻塞** | `download_with_pool()` 阻塞等待 GPU | 无此问题 |
| **无 CPU fallback** | GPU 不可用时功能全失 | 核心就是 CPU |

---

## 四、可借鉴的改进建议

### ✅ 强烈建议借鉴

#### 1. 引入 PDQ/DCT 频域哈希算法

当前项目的 6 种空间域哈希在实际图像匹配场景中鲁棒性不足。PDQ 基于 DCT 频域变换，是 Meta 在生产环境中验证过的算法。

**具体方案**：
- 作为新的 `HashAlgorithm` 变体添加 PDQ
- 可参考 `pdqhash` crate（yume-pdq 的 dev-dependency，提供参考实现）
- 或直接依赖 `yume-pdq` 作为可选 feature（Apache-2.0 兼容）

```rust
pub enum HashAlgorithm {
    Mean,
    Median,
    Gradient,
    Block,
    VertGradient,
    DoubleGradient,
    Pdq,  // 新增：PDQ 频域哈希
}
```

#### 2. CPU 优先 + 可选 SIMD 加速架构

yume-pdq 的分层设计完美契合当前项目"纯 CPU 算法库"的目标：

```
当前:  用户 → PerceptualHasher → GpuContext → WGSL (GPU 必须)
建议:  用户 → PerceptualHasher → CPU 标量实现 (默认)
                                    ↓ (可选)
                                  SIMD 加速 (AVX2/512)
```

**具体方案**：
- 将现有 `tests/common/hash_reference.rs` 中的 CPU 参考实现升级为生产级实现
- 通过 feature flag (`simd`, `avx2`, `avx512`) 提供可选加速
- 使用 `cpufeatures` crate 运行时检测，自动选择最优内核
- 移除 `PerceptualHashComputer` trait 对 `GpuContext` 的依赖

#### 3. 二面体变换支持

当前项目完全缺失旋转/翻转不变性，在实际图像匹配中是致命缺陷。

```rust
/// 二面体变换哈希集合（8 种旋转/翻转变体）
pub struct DihedralHashes {
    pub original: u64,
    pub rotate90: u64,
    pub rotate180: u64,
    pub rotate270: u64,
    pub flip_h: u64,
    pub flip_v: u64,
    pub flip_diag: u64,
    pub flip_anti_diag: u64,
}

impl DihedralHashes {
    /// 从原始哈希位矩阵推导所有变体（无需重新计算图像）
    pub fn from_hash_matrix(bits: &[Vec<bool>], width: usize, height: usize) -> Self {
        // 旋转/翻转位矩阵，重新打包为 u64
    }
}
```

#### 4. 重新评估 BK-tree vs 精确线性扫描

yume-pdq 的论证表明，对于感知哈希匹配场景，BK-tree 的优势不明显甚至退化。建议：
- 保留 BK-tree 作为可选索引结构
- 新增 SIMD 加速的精确线性扫描匹配器
- 在 10M 规模以下优先使用线性扫描（100% 召回率保证）
- 匹配器 trait 化，支持策略选择：

```rust
pub trait HashMatcher {
    fn find_similar(&self, query: u64, threshold: u32) -> Vec<MatchResult>;
}

pub struct BkTreeMatcher { /* ... */ }
pub struct LinearScanMatcher { /* ... */ }  // 新增：SIMD 加速
```

#### 5. Feature Flag 重构

参考 yume-pdq 的 feature 设计，实现核心零依赖 + 可选加速：

```toml
[features]
default = ["std"]
std = ["alloc"]
alloc = []
simd = []          # 启用 SIMD 运行时检测
avx2 = ["simd"]    # 编译期假定 AVX2
avx512 = ["simd"]  # 编译期假定 AVX512 (nightly)
image = ["dep:image"]
pdq = ["dep:yume-pdq"]  # 可选 PDQ 算法
```

### ⚠️ 谨慎借鉴

#### 6. "近似哈希"策略

yume-pdq 故意偏离参考实现以换取速度，但明确声明优化内核的哈希不应提交到数据库。当前项目应：
- 保持精确实现作为默认
- 近似模式仅作为可选加速，并明确文档警告

#### 7. no-std 支持

yume-pdq 的 no-std 支持使其可运行于嵌入式和 WASM。当前项目由于 wgpu 依赖无法实现 no-std，但移除 GPU 依赖后可以考虑。优先级低于核心功能迁移。

### ❌ 不应借鉴

#### 8. Vulkan GPU 匹配

yume-pdq 的 Vulkan 匹配器违反当前项目"纯 CPU 算法库"的明确目标（AGENTS.md），不应引入。

---

## 五、改进优先级建议

| 优先级 | 改进项 | 预期收益 | 工作量 | 依赖关系 |
|--------|--------|----------|--------|----------|
| **P0** | 移除 GPU 依赖，CPU 实现作为核心 | 库可在任何环境运行 | 中 | 无（CPU 参考实现已存在） |
| **P1** | 引入 PDQ 算法 | 哈希质量质的飞跃 | 中 | P0 完成 |
| **P1** | 二面体变换支持 | 实际匹配场景必需 | 低 | P0 完成 |
| **P2** | SIMD 加速内核 | 性能 4-16x 提升 | 高 | P0 完成 |
| **P2** | 精确线性扫描匹配器 | 100% 召回率保证 | 低 | P0 完成 |
| **P3** | Feature flag 重构 | 架构清晰化 | 低 | P0 完成 |
| **P3** | no-std 支持 | 扩大适用场景 | 中 | P0+P2 完成 |

---

## 六、关键结论

1. **yume-pdq 对当前项目的改进有高度参考价值**，但参考方向需要"反转"：
   - yume-pdq 的 GPU 用于**匹配加速**，哈希计算在 CPU → 当前项目应**完全移除 GPU 依赖**，所有计算回归 CPU
   - yume-pdq 的核心价值在于 **CPU SIMD 优化**和**算法选择（PDQ/DCT）**，而非 Vulkan 加速

2. **yume-pdq 证明了纯 CPU 实现完全可以达到生产级性能**（AVX512 下 20ms/10M 向量），这为当前项目"纯 CPU 算法库"的目标提供了有力的实践佐证

3. **算法层面的差距是根本性的**：当前项目的 6 种空间域哈希与 PDQ 的 DCT 频域哈希不在同一量级，引入 PDQ 是提升哈希质量最有效的途径

4. **BK-tree 在感知哈希场景下的优势存疑**：yume-pdq 的论证表明精确线性扫描在 10M 规模下仍优于 BK-tree，当前项目应提供多种匹配策略供用户选择

5. **二面体变换是实际应用的硬需求**：没有旋转/翻转不变性的感知哈希在实际图像匹配中几乎不可用，这是当前项目最紧迫的功能缺口
