# GPU 加速强化策略分析

**日期**: 2026-05-23  
**基于**: 18 份基准测试 + 两份性能报告 + 着色器分析

---

## 核心发现：GPU 仅用于 0.6% 的计算时间

```
单次 compute() 时间分布 (1000 条 SHA-256)

├─ CPU-GPU 同步阻塞  ████████████████████████████████████████ 87.7% (1400µs)
├─ Buffer 创建/销毁  ▌  1.2% (20µs)
├─ 数据上传           █  3.0% (50µs)
├─ 命令编码+提交      ▌  1.8% (30µs)
├─ GPU 实际计算       ▏  0.6% (10µs)  ← 仅此部分！
├─ 数据下载+解析      █  3.0% (50µs)
                      ─────
                     ~1600µs 总计
```

**结论**: 任何对着色器和 buffer 的微优化都无法产生可测量效果。必须首先消除同步瓶颈。

---

## 优化路线图

### 层级一：消除同步瓶颈（预估 10-100x 提升）

| 优化项 | 当前代价 | 目标 | 预估收益 |
|--------|---------|------|---------|
| **1.1 全异步批量提交 API** | 每次 compute() 阻塞 1.4ms | 合并为单次 poll | **10-100x** |
| **1.2 多 block 链式着色器** | 每条消息每 block 一次往返 | 消息内所有 block 单次 dispatch | **N×** (N=block数) |
| **1.3 持久化映射暂存缓冲区** | 每次 map_async 1.4ms | 环形缓冲持久映射 | **14x** |
| **1.4 双缓冲流水线** | 计算和传输串行 | 上一批计算时上传下一批 | **1.5-2x** |

### 层级二：着色器优化（预估 3-10x 提升）

| 优化项 | 问题 | 修复 | 预估收益 |
|--------|------|------|---------|
| **2.1 Median hash 共享直方图** | 每线程 1024B 数组 | `var<workgroup>` 共享 | **5-10x** |
| **2.2 Resize 消除除法和取模** | 每像素整数除法 | 预计算或 u8 数组 | **3-5x** |
| **2.3 SHA-256 消息调度优化** | 每线程 256B w 数组 | 分片 + 寄存器复用 | **2-3x** |
| **2.4 非合并内存访问修复** | block_hash/vert_gradient 跨步 | 数据重排或转置 pass | **2-3x** |

### 层级三：架构级优化（预估 1.5-3x 提升）

| 优化项 | 问题 | 修复 |
|--------|------|------|
| **3.1 可变 workgroup size** | 小批量时 SM 利用率 <15% | batch < 256 时降为 64 |
| **3.2 像素打包传输** | 每像素 4B vs 1B | 感知哈希输入 4 像素/u32 |
| **3.3 统一 params buffer 缓存** | 每次创建 16B uniform buffer | 复用或合并为 push constant |

---

## 详细方案

### 1.1 全异步批量提交 API

**现状**: `Sha256BatchSubmitter` 已支持异步提交单 block 批次，但：
- 多 block 路径中断流水线（第 534 行 submmit encoder）
- 缺少面向用户的统一异步 API
- 感知哈希没有对应的异步提交器

**方案**: 创建 `AsyncComputeSession`：

```rust
pub struct AsyncComputeSession<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    encoder: wgpu::CommandEncoder,
    pending: Vec<PendingJob>,
}

impl AsyncComputeSession {
    /// 提交一批计算任务（不阻塞）
    pub fn submit_sha256(&mut self, computer: &Sha256Computer, messages: &[Vec<u8>]);
    pub fn submit_phash(&mut self, computer: &dyn PerceptualHashComputer, images: &[Vec<u8>]);
    
    /// 统一等待所有任务完成（仅 1 次 poll）
    pub fn wait_all(&mut self) -> Result<Vec<ComputeResult>>;
}
```

**关键**: 所有 dispatch + copy 编码到同一个 `CommandEncoder`，一次 `queue.submit()` + 一次 `poll(Wait)`。

**预估**: 10 个单 block 批次：同步 10×1.6ms=16ms → 异步 ~1.6ms（10x 加速）。100 个批次：100×1.6ms=160ms → ~1.7ms（94x 加速）。

### 1.2 多 block 链式着色器

**现状**: SHA-256 多 block 路径逐 block dispatch+download：
```rust
for block in &blocks {
    pipeline.dispatch(..., [1, 1, 1]);  // 每 block 一次
    let result = output_buffer.download()?;  // 每 block 1.4ms 同步
    intermediate = parse(result);
}
```
64B 消息（2 block）= 3.2ms vs CPU 757ns = **42,000x 退化**。

**方案**: 创建新的 WGSL 着色器变体，在单次 dispatch 内处理消息的全部 block：

```wgsl
// sha256_chained.wgsl
struct MessageBlocks {
    blocks: array<u32>,      // 全部 block 数据平铺
    block_offsets: array<u32>, // 每条消息的 block 偏移
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let msg_idx = gid.x;
    let start_block = block_offsets[msg_idx];
    let end_block = block_offsets[msg_idx + 1];
    
    var h: array<u32, 8> = INITIAL_HASH;
    for (var b = start_block; b < end_block; b++) {
        h = sha256_compress(h, blocks, b * 16);
    }
    
    // 输出最终哈希
    hashes[msg_idx * 8 .. (msg_idx+1) * 8] = h;
}
```

**预估**: 10 条 64B 消息：32.3ms → ~1.7ms（**19x**）。10 条 1024B（17 block）：292ms → ~2ms（**146x**）。

### 1.3 持久化映射暂存缓冲区

**现状**: 每次 `download()` 创建 staging buffer → map_async → poll(Wait) → 读取 → unmap → 销毁。poll(Wait) 约 1.4ms。

**方案**: 预分配 N 个 staging buffer，保持持久映射，使用环形缓冲区：

```rust
struct PersistentStagingRing {
    buffers: Vec<wgpu::Buffer>,    // 预分配 + 持久映射
    mapped_views: Vec<wgpu::BufferView<'static>>,
    write_idx: AtomicUsize,
    read_idx: AtomicUsize,
    fence_values: Vec<u64>,        // 每个 buffer 的完成 fence
}
```

**关键**: 使用 wgpu `Buffer::map_async` + `Maintain::Poll`（非 Wait），配合 fence 轮询。将 1.4ms 的阻塞等待转为 ~50µs 的 fence 检查。

**预估**: 单次往返 1.4ms → 0.1ms（**14x**）。

### 1.4 双缓冲流水线

**方案**: GPU 计算当前批次的同时，CPU 上传下一批次的数据：

```
时间轴：
Batch 0: [Upload] [GPU Compute]        [Download]
Batch 1:            [Upload] [GPU Compute]        [Download]
Batch 2:                       [Upload] [GPU Compute]
```

wgpu 的命令队列天然支持这种流水线——多次 `queue.submit()` 无需 CPU 同步即可在 GPU 上序列化执行。

---

### 2.1 Median Hash 共享直方图

**现状**: 每个线程拥有 256 元素的 `array<u32, 256>` histogram。256 线程 × 256 元素 = 65536 次写入，全部溢出到 VRAM。

**方案**: 使用 `var<workgroup>` 共享内存，每个 workgroup 共享一个直方图：

```wgsl
var<workgroup> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, ...) {
    // 每个线程初始化直方图的一个槽位
    if (lid.x < 256u) {
        atomicStore(&histogram[lid.x], 0u);
    }
    workgroupBarrier();
    
    // 每个线程处理图像的 N/256 个像素
    let pixels_per_thread = pixels_per_image / 256u;
    for (var i = 0u; i < pixels_per_thread; i++) {
        let val = pixels[base + lid.x * pixels_per_thread + i];
        atomicAdd(&histogram[val], 1u);
    }
    workgroupBarrier();
    
    // 只有线程 0 计算中位数
    if (lid.x == 0u) {
        let median = find_median(histogram);
        // ... 生成哈希
    }
}
```

**预估**: 内存占用从 256KB/workgroup 降至 1KB/workgroup，寄存器压力消除，**5-10x 加速**。

### 2.2 Resize 消除除法

**现状**: `get_src_pixel()` 每次调用包含整数除法和取模。

**方案**: 改变数据格式或预计算偏移：

```wgsl
// 方案 A: 使用 array<u8> (如果 wgpu 支持)
@group(0) @binding(0) var<storage, read> src_pixels: array<u8>;

// 方案 B: 预计算每像素的字偏移和位移
struct PixelOffset {
    word_idx: u32,
    shift: u32,
}
// 在 uniform buffer 中预计算

// 方案 C: 简单用 u32 每像素（已经打包则跳过解包）
// 在 CPU 侧将 u8 像素预先转为 u32，避免 GPU 上解包
```

**预估**: **3-5x 加速** resize 操作。

### 2.3 SHA-256 消息调度优化

**现状**: `var w: array<u32, 64>` 每线程 256B。

**方案**: SHA-256 消息调度的前 16 个字可以直接从输入读取，后续 48 个字可以增量计算并立即使用，无需全量存储：

```wgsl
fn sha256_compress(h: ptr<function, array<u32, 8>>, block: ptr<function, array<u32, 16>>) {
    // 只需 16 个字的滑动窗口而非 64 个字
    var w: array<u32, 16>;
    // 初始 16 字从 block 复制
    for (var i = 0u; i < 16u; i++) { w[i] = block[i]; }
    
    // 64 轮压缩，滚动更新 w
    for (var t = 0u; t < 64u; t++) {
        if (t >= 16u) {
            let idx = t % 16u;
            w[idx] = small_sigma1(w[(idx + 14u) % 16u]) + w[(idx + 9u) % 16u] 
                   + small_sigma0(w[(idx + 1u) % 16u]) + w[idx];
        }
        // 压缩轮
        let t1 = h[7] + big_sigma1(h[4]) + ch(h[4], h[5], h[6]) + K[t] + w[t % 16u];
        let t2 = big_sigma0(h[0]) + maj(h[0], h[1], h[2]);
        // 旋转 h 寄存器
    }
}
```

**预估**: 内存从 256B → 64B，寄存器压力降低 75%，**2-3x 加速**。

### 2.4 非合并内存访问修复

**Vertical Gradient**: 列优先访问 `pixels[row * width + col]`，步长为 width。

**方案**: 将输入数据转置后传入，或将垂直梯度改为使用 `subgroupShuffle` 收集相邻行像素：

```wgsl
// 使用 subgroup 操作
let current = pixels[base + row * width + col];
let below = subgroupShuffleDown(current, width); // 获取下一个"行"的像素
```

**Block Hash**: 四重嵌套循环，高度跨步。

**方案**: 改为先一次性加载 2×2 块的所有像素到寄存器，再计算均值：

```wgsl
// 2x2 块的均值可以合并 4 次内存访问为 1 次
let top_left = pixels[base + ...];
let top_right = pixels[base + ... + 1];
let bottom_left = pixels[base + ... + width];
let bottom_right = pixels[base + ... + width + 1];
let mean = (top_left + top_right + bottom_left + bottom_right) / 4u;
```

---

### 3.1 可变 workgroup size

**问题**: `@workgroup_size(256)` 对于小批量（batch < 256）浪费 GPU 资源。

**方案**: 根据批次大小动态选择 workgroup size：

| 批次大小 | workgroup | 理由 |
|---------|-----------|------|
| < 64 | 64 | 单 workgroup 足够 |
| 64-256 | 128 | 平衡占用 |
| 256-1024 | 256 | 当前默认 |
| > 1024 | 512 | 最大化占用 |

---

## 性能提升预估

### 保守预估（仅实现层级一）

| 场景 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 1000 条 SHA-256 (32B) | 1.78 ms | 0.15 ms | **12x** |
| 10000 条 SHA-256 (32B) | 3.99 ms | 0.40 ms | **10x** |
| 10 条 SHA-256 (64B, 多 block) | 32.3 ms | 0.50 ms | **65x** |
| 10 条 SHA-256 (1024B, 17 block) | 292 ms | 1.00 ms | **292x** |
| 1000 张 Mean Hash (8×8) | ~2 ms | 0.15 ms | **13x** |
| GPU vs CPU 盈亏平衡点 | 30,000 条 | **1,000 条** | 30x 降低 |

### 激进预估（实现层级一+二+三）

| 场景 | 提升 |
|------|------|
| 单 block SHA-256 批量 | **15-20x** |
| 多 block SHA-256 | **200-500x** |
| 感知哈希 (含 resize) | **10-15x** |
| GPU vs CPU 盈亏平衡 | **~200 条** |

---

## 实现优先级

| 优先级 | 优化 | 难度 | 影响面 | 预估价值 |
|--------|------|------|--------|---------|
| **P0** | 链式着色器 | 中 | 仅 SHA-256 | 多 block 路径 **100-500x** |
| **P0** | 统一异步 Session API | 中 | 所有算法 | **10-100x** |
| **P1** | Median hash 共享直方图 | 低 | 仅 Median | **5-10x** |
| **P1** | Resize 消除除法 | 低 | 所有 resize | **3-5x** |
| **P2** | SHA-256 滑动窗口 W | 中 | 仅 SHA-256 | **2-3x** |
| **P2** | 非合并访问修复 | 低 | Vert/Block | **2-3x** |
| **P3** | 持久化映射环形缓冲 | 高 | 所有算法 | **14x**（同步延迟） |
| **P3** | 可变 workgroup size | 低 | 所有算法 | 小幅 (小批量) |

---

## 风险提醒

1. **链式着色器的 VRAM 限制**: 长消息（如 1MB）的 block 数据全部放入 single dispatch 会超出 buffer binding size 限制。需要分块策略（每 N 个 block 一组）。

2. **共享直方图的原子操作代价**: `atomicAdd` 会有存储体冲突。对于 256 线程 × 256 槽位的直方图，冲突概率高。可使用多个局部直方图 + 最终合并。

3. **双缓冲流水线复杂度**: 需要仔细管理 wgpu buffer 生命周期，避免 use-after-submit。

4. **持久化映射**: wgpu 目前对 buffer 持久映射的支持有限（需要 `MAP_READ | MAP_WRITE` 同时设置，但某些后端不支持）。可降级为 BufferPool + 更频繁的 unmap。
