# RX 5700 GPU 并行计算与性能释放深度调研

## 1. 硬件架构

### 1.1 核心规格

| 参数 | 数值 |
|------|------|
| 架构 | RDNA 1.0 (Navi 10) |
| 制程 | TSMC 7nm |
| Compute Units (CU) | 36 |
| Dual Compute Units (WGP) | 18 |
| Stream Processors | 2304 |
| 纹理单元 | 144 |
| ROPs | 64 |
| Game Clock | 1625 MHz |
| Boost Clock | 1725 MHz |
| FP32 单精度 | 7.95 TFLOPS |
| FP16 半精度 | 15.9 TFLOPS |
| 显存 | 8 GB GDDR6 |
| 显存位宽 | 256-bit |
| 显存带宽 | 448 GB/s |
| 显存速率 | 14 Gbps |
| TDP | 180W |

### 1.2 CU 内部结构 (从 Vulkan Hardware DB 实测)

| 参数 | 数值 |
|------|------|
| SIMD per CU | 2 |
| VGPRs per SIMD | 1024 (32-bit) |
| VGPR 分配粒度 | 8 |
| Wavefront Slots per SIMD | 20 |
| Vulkan 报告 subgroupSize | 64 |
| 原生 Wave32 支持 | 是 (硬件原生，驱动可双发射) |

### 1.3 RDNA CU 与 GCN 关键差异

| 维度 | GCN | RDNA |
|------|-----|------|
| SIMD 配置 | 4×SIMD16 | 2×SIMD32 |
| Wave 执行 | Wave64 需 4 周期 | Wave64 1 周期 / Wave32×2 双发射 |
| 指令延迟 | 4 周期完成一条 Wave64 | 单周期完成 |
| ALU 数量 | 基准 | 2× GCN |
| 缓存带宽 | 基准 | 4× GCN |
| 解码器 | 1 路标量 + 1 路矢量 | 2 路标量 + 2 路矢量 |
| 操作数聚集 | 4 周期 | 1 周期 |

**核心结论**：RDNA 的 CU 虽然在数量上少于同代 Vega，但单 CU 吞吐量大幅提升——更宽的解码、双倍的 ALU、四倍的缓存带宽、单周期 Wave 指令完成，使得 36 CU 的 RX 5700 实际上在计算密度上远超 GCN 架构。

---

## 2. 内存层次与带宽

### 2.1 三级缓存体系

```
L0 Cache (per CU):   指令缓存(I$) + 标量数据缓存(K$) + 矢量数据缓存(V$)
                     V$ 拥有两倍载入带宽
L1 Cache (per WGP):  两个 CU 共享，RDNA 新增中间层级
L2 Cache:            4 MB，全局共享
Cache Line:          128 字节 (32 threads × 4 bytes = Wave32 单周期访问对齐)
```

### 2.2 内存类型性能对比

| 内存类型 | 延时 | 带宽 | 作用域 |
|----------|------|------|--------|
| VGPR (向量寄存器) | ~0 cycles | 极高 | 单线程 |
| SGPR (标量寄存器) | ~0 cycles | 极高 | 单线程(Wave 内广播) |
| LDS (Local Data Share) | ~20 cycles | ~2 TB/s (估算) | Workgroup 内 |
| L0 Cache | ~30-50 cycles | 高 | CU 内 |
| L1 Cache | ~50-100 cycles | 中 | WGP 内 (2 CU) |
| L2 Cache | ~150-300 cycles | 中 | 全局 |
| GDDR6 VRAM | ~400-800 cycles | 448 GB/s | 全局 |

### 2.3 有效带宽经验值

| 访问模式 | 利用率 |
|----------|--------|
| 完美合并 (Coalesced) | 70-90% |
| 有步长但对齐 | 40-70% |
| 随机访问 | 5-15% |

---

## 3. Occupancy 与资源占用

### 3.1 理论基础

RDNA1 每个 SIMD 有 20 个 Wavefront Slot，每个 CU 有 2 个 SIMD。在 Wave32 模式下，理论最大每 CU 驻留线程数为：

```
20 slots × 2 SIMDs × 32 threads = 1280 threads/CU (Wave32)
20 slots × 2 SIMDs × 64 threads = 2560 threads/CU (Wave64)
```

但实际 occupancy 受限于三种资源：

1. **VGPR 数量**：每个 SIMD 1024 个 32-bit VGPR。若 shader 使用 N 个 VGPR，每 SIMD 最多驻留 `floor(1024 / N)` 个 Wave。
2. **LDS 占用**：每 CU 有 64 KB LDS。Workgroup 的 shared memory 分配量直接限制同时驻留的 workgroup 数量。
3. **Workgroup 大小**：过小的 workgroup 浪费 slot，但过大的 workgroup 可能因 VGPR/LDS 瓶颈反而降低 occupancy。

### 3.2 RX 5700 关键资源表

| 资源 | 每 CU 总量 | 限制含义 |
|------|-----------|----------|
| VGPRs | 2048 (2×1024) | 决定每 CU 可驻留 Wave 数 |
| LDS | 64 KB | 决定每 CU 可驻留 Workgroup 数 |
| Wave Slots | 40 (2×20) | 硬上限 |
| 最大 Workgroup Size | 1024 threads | API 限制 |

### 3.3 Occupancy 速查表 (Wave32 模式，Workgroup = 64 threads)

| VGPRs/Thread | Waves/CU | Occupancy | 状态 |
|-------------|----------|-----------|------|
| ≤25 | 40 | 100% | 理想 |
| 26-32 | 32 | 80% | 良好 |
| 33-40 | 24 | 60% | 一般 |
| 41-51 | 20 | 50% | 警戒 |
| 52-64 | 16 | 40% | 偏低 |
| >64 | <16 | <40% | 严重不足 |

> 注：VGPR 分配粒度为 8，实际分配量上取整到 8 的倍数。

---

## 4. Compute Shader 优化策略

### 4.1 Workgroup 设计

| 策略 | 说明 |
|------|------|
| **大小为 64 的倍数** | 兼容 GCN Wave64 和 RDNA Wave32/Wave64 双模式，跨代最优 |
| **2D 推荐 8×8=64** | 天然匹配 256-byte 合并写入块，每 Wave 写一个 8×8 像素块 |
| **避免过小 (<32)** | 浪费 Wave Slot，无法充分隐藏延迟 |
| **避免过大 (>256)** | VGPR/LDS 压力增大，反而降低 occupancy |
| **分组活跃/非活跃线程** | 将活跃线程集中在连续的 32 个一组，RDNA 可整组跳过非活跃 Wave32 |

```
推荐: workgroup_size = (64, 1, 1) 或 (8, 8, 1)
备选: workgroup_size = (128, 1, 1) 或 (256, 1, 1)  (更高 occupancy 但需控制 VGPR)
```

### 4.2 内存合并访问

| 模式 | 推荐做法 |
|------|----------|
| **全局写入** | 每 Wave 写入 256 字节合并块 (32 threads × 8 bytes 或 64 threads × 4 bytes) |
| **图像处理** | 8×8 线程组写 8×8 像素块，使用 Morton Swizzle 线程重排 (+8% 性能) |
| **纹理采样** | 用 `Gather4` 替代 `Sample`（单通道采样时），减少纹理单元流量和 VGPR |
| **结构体访问** | 优先 Array of Structs → Struct of Arrays 避免跨步访问 |
| **对齐** | 大缓冲区对齐到 64 字节边界 |

#### Morton Swizzle 线程重排示例

```wgsl
// 标准行优先: thread(x,y) → pixel(x,y)
// Morton: thread(x,y) → pixel(morton(x,y))
// 使邻近线程访问邻近的 2×2 纹素块，匹配纹理硬件布局
fn morton_2d(x: u32, y: u32) -> u32 {
    // 交错 x, y 的比特位
    // 实现略，效果：cache 命中率提升 ~8%
}
```

### 4.3 LDS 优化

| 要点 | 细节 |
|------|------|
| **Bank 结构** | 32 个 Bank，每 Bank 32-bit (1 DWORD) |
| **Bank Conflict** | 同 Bank 同时访问序列化。用 `float4` 数组读 X 分量 → 8 路冲突；用 `float` 数组 → 2 路冲突 |
| **解决方案** | Struct of Arrays + Padding，例如 `float array[32 + PADDING]` |
| **经典场景** | 邻域滤波：先加载到 LDS，消除 9 次冗余全局读取 |
| **Multi-Pass** | 多趟算法中间结果存 LDS 而非写回全局内存，最后一次性写出 |

### 4.4 指令级优化

| 操作类型 | 吞吐量 | 建议 |
|----------|--------|------|
| FP32 FMA/MUL/ADD | Full Rate | 优先使用 |
| FP16 | 2× Rate (双倍速率) | 能降精度则降，同时减少 VGPR 占用 |
| INT32 | Full Rate | 自由使用 |
| 超越函数 (sin/cos/sqrt/log/rcp) | 1/4 Rate | 最小化使用，用近似替代 |
| 反三角函数 (atan/acos) | 100+ cycles | 绝对避免 |
| tan | 展开为 sin/cos (3 个超越函数) | 避免 |
| 除法 | 比乘法慢 | 尽量用倒数乘法替代 |

#### 快速近似替代

```
// sqrt → 使用 inversesqrt 再倒数
// sin/cos → FidelityFX 快速近似库
// rcp → native_recip (精度略降但快 4×)
```

### 4.5 Wave 内数据交换

| 操作 | 性能 | 适用范围 |
|------|------|----------|
| DPP8 (任意置换) | 极快 | 8 线程内 |
| DPP16 (预定义置换) | 极快 | 16 线程内 |
| Quad Operations | 极快 | 2×2 四元组内 |
| LDS Permute | 快 | 16-32 线程间 |
| 跨 Wave LDS | 需要 barrier | 任意范围 |

**建议**：优先 DPP8 → DPP16 → Quad Ops → LDS Permute，避免跨 32 线程以上的 shuffle（编译器会回退到慢路径）。

### 4.6 FidelityFX SPD 模式 —— 单 Pass 多级降采样

```wgsl
// 核心思想：单次 Dispatch 完成所有 Mip 级降采样
// 1. 输入纹理切为 64×64 的 tile
// 2. Workgroup = 256 threads，将 64×64 逐级降采样到 1×1
// 3. 无内部 barrier，仅一个全局同步点
// 4. Mip 间数据通过 LDS/DPP 传递，不写回 VRAM
// 5. 可与图形队列异步并行
//
// 效果：消除多 Pass 的 barrier 瓶颈，避免中间 mip 的 VRAM 读写
```

---

## 5. wgpu/Rust 特化建议

### 5.1 Workgroup Size 声明

```rust
// WGSL
@compute @workgroup_size(64)  // 推荐 64 或 128
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // ...
}

// 或 2D:
@compute @workgroup_size(8, 8)  // 8×8 适合图像处理
```

### 5.2 Dispatch 粒度计算

```
dispatch_count_x = ceil(total_items / workgroup_size_x)
```

建议 dispatch 量 = CU 数 × (4-8) 倍，即至少 144-288 个 workgroup，确保每个 CU 有足够 wave 隐藏延迟。

### 5.3 Storage Buffer 对齐

```rust
// Rust 侧：确保 buffer 大小对齐
let padded_size = (data_size + 255) & !255; // 256 字节对齐
```

### 5.4 使用 Push Constants 传递小参数

减少 Bind Group 切换开销，将 ≤32 bytes 的常变参数放入 push constants。

### 5.5 时间戳查询用于 Profiling

```rust
// wgpu 支持 timestamp-query 特性
// 在关键 Dispatch 前后插入，测量 GPU 执行时间
```

---

## 6. 感知哈希 GPU 化的 RX 5700 适配策略

结合 RDNA 特性和此前讨论的感知哈希方案：

| 优化点 | 具体策略 |
|--------|----------|
| **批量 DCT 计算** | 每 thread 处理 1 个 8×8 块，Workgroup=8×8=64，Dispatch=ceil(N_images×N_blocks/(64)) |
| **图像预处理** | 用 Morton Swizzle 线程重排读取源图像，提升 Cache 命中 |
| **中间数据** | DCT 系数暂存 LDS，避免跨 workgroup 的 VRAM 读写 |
| **FP16 精度** | DCT 和哈希计算可降为 FP16，吞吐翻倍、VGPR 减半 |
| **超越函数规避** | 哈希对比中的 sqrt → inversesqrt 优化 |
| **合并写入** | 最终哈希值 8 线程合并为一次 256-byte 写入 |

### 6.1 预估吞吐量

RX 5700 拥有 36 CU @ 1625 MHz Game Clock：

- FP16 理论峰值：15.9 TFLOPS
- 考虑内存带宽限制：448 GB/s
- 假设每像素 DCT 需要 ~200 FP16 FLOPS + 8 bytes 读写
- 内存瓶颈：448 GB/s ÷ 8 bytes/pixel = 56 G pixels/s
- 计算瓶颈：15.9 TFLOPS ÷ 200 FLOPS/pixel = 79.5 G pixels/s

以 256×256 图像为例，单张 65536 像素：
- 理论每秒处理：56G / 65536 ≈ 85 万张/秒 (内存瓶颈)
- 实际有效率 (~70%)：约 60 万张/秒

> 相比 CPU 单核 ~50-100 张/秒，GPU 加速比达到 6000-12000×。

---

## 7. 总结：RDNA 优化检查清单

| 类别 | 检查项 | 影响 |
|------|--------|------|
| Workgroup | 大小为 64 的倍数 | 跨代兼容 + 最优 occupancy |
| Workgroup | 2D 场景用 8×8 | 匹配 256-byte 合并写入 |
| VGPR | 控制在 32 以内 (≥80% occupancy) | 延迟隐藏能力 |
| VGPR | 用 FP16 缩减寄存器占用 | 翻倍 occupancy |
| LDS | Struct of Arrays + padding | 消除 Bank Conflict |
| LDS | 多趟算法中间结果留 LDS | 减少 VRAM 读写 |
| 内存 | 256-byte 合并写入 | 带宽利用率 70-90% |
| 内存 | Morton Swizzle 线程重排 | +8% Cache 命中 |
| 内存 | Gather4 替代 Sample | 减少纹理单元流量 |
| 指令 | 避免超越函数 (sin/cos/sqrt) | 4× 吞吐 |
| 指令 | 避免反三角函数 (atan/acos) | 100+ cycles |
| Wave | DPP8/DPP16 优于 LDS permute | 减少延迟 |
| Wave | 活跃线程连续分组 | RDNA 可跳过非活跃 Wave32 |
| Dispatch | 总量 ≥ CU×4 | 充分填充 GPU |
| API | Push Constants (≤32B) | 减少 Bind Group 切换 |