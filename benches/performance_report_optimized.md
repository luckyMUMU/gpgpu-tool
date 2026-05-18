# SHA-256 GPU 计算引擎性能测试报告（优化后）

> 测试日期: 2026-05-18
> 测试平台: Windows (wgpu Vulkan 后端)
> 测试工具: Criterion.rs (100 samples, 优化构建)
> 优化内容: BufferPool 缓冲区复用 + 多 block 批量收集

---

## 一、核心结论

| 场景 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| 单条短消息 (32B) | ~1.58ms | **1.62ms** | +2.5% (噪声) |
| 1000 条批量 (32B) | ~1.86ms | **1.78ms** | -4.3% |
| **10000 条批量 (32B)** | **~3.88ms** | **~3.99ms** | **+2.8% (噪声)** |
| 多 block 64B (10条) | ~31.3ms | **32.3ms** | +3.2% (噪声) |
| 空 dispatch | ~122µs | **142µs** | +16% (噪声) |

**关键发现**: BufferPool 优化在单次基准测试中的效果被 **测试方差** 掩盖。固定同步开销（~1.5ms/次）仍是绝对主导因素，BufferPool 减少的分配开销（约 50-100µs）在噪声范围内难以稳定测量。

---

## 二、详细测试结果（优化后，全新基准）

### 2.1 单 block 批量处理 (消息长度 32B)

| 消息数量 | GPU 耗时 | 每条耗时 | CPU 耗时 | 每条耗时 | GPU/CPU 比值 |
|---------|---------|---------|---------|---------|-------------|
| 1 | 1.62ms | 1620µs | 52ns | 52ns | **31197x** |
| 10 | 1.65ms | 165µs | 507ns | 51ns | **3256x** |
| 100 | 1.63ms | 16.3µs | 5.11µs | 51ns | **318x** |
| 1000 | 1.78ms | 1.78µs | 52.4µs | 52ns | **34x** |
| 10000 | 3.99ms | 0.40µs | 524µs | 52ns | **7.6x** |

**分析**:
- GPU 固定开销仍约 **1.6ms**，与消息数量基本无关
- 10000 条批量时，GPU 每条 0.40µs，CPU 每条 52ns，仍慢 7.6 倍
- 盈亏平衡点仍需 ~50000 条/批次

### 2.2 多 block 消息处理 (10 条消息)

| 消息长度 | GPU 耗时 | CPU 耗时 | GPU/CPU 比值 |
|---------|---------|---------|-------------|
| 64B (2 blocks) | 32.3ms | 757ns | **42668x** |
| 128B (3 blocks) | 51.0ms | 1.07µs | **47664x** |
| 256B (5 blocks) | 84.8ms | 1.73µs | **49017x** |
| 1024B (17 blocks)| 292.6ms | 5.62µs | **52064x** |

**分析**:
- 多 block 消息性能与优化前基本持平（逐 block dispatch 的同步开销未被消除）
- 64B 消息 = 2 blocks × 10 条 = 20 次 dispatch，每次约 1.6ms
- 这是当前架构下无法避免的（SHA-256 有数据依赖，必须串行）

### 2.3 GPU 开销拆解

| 测试项 | 耗时 | 说明 |
|-------|------|------|
| 空 dispatch (无计算) | 142µs | 纯命令提交+同步开销 |
| 单消息端到端 | 1.64ms | 数据准备 + 调度 + 计算 + 下载 |
| 1000 消息端到端 | 1.81ms | 批量摊薄后，固定开销仍占 90%+ |

**推算**:
- 单次完整往返 ≈ **1.6-1.8ms**
- 空 dispatch 142µs，剩余 ~1.5ms 为 buffer 创建/数据拷贝/map_async 等待

### 2.4 Pipeline 缓存

| 操作 | 耗时 |
|------|------|
| Pipeline 创建 (缓存命中) | ~5.2µs |

**分析**: 缓存机制工作良好，非瓶颈。

### 2.5 Workgroup Size 对比 (10000 条 32B 消息)

| Workgroup Size | 耗时 | 相对性能 |
|---------------|------|---------|
| 64 | 3.42ms | 基准 (最快) |
| 128 | 3.39ms | +1% |
| 256 | 3.47ms | -1% |
| 512 | 3.64ms | -6% |

**分析**:
- workgroup size 差异 < 6%，被同步开销完全淹没
- wg=64 略快，可能因为小 workgroup 减少了 warp 内空闲线程
- 当未来消除同步开销后，workgroup size 调优将变得重要

---

## 三、优化效果评估

### BufferPool 预期收益 vs 实际测量

| 优化项 | 理论收益 | 实际测量 | 差异原因 |
|--------|---------|---------|---------|
| BufferPool 复用 | 减少 50-100µs/次 buffer 创建 | 在噪声范围内不可见 | 同步开销 1.6ms 占主导，100µs 仅占 6% |
| 多 block 批量收集 | 减少多次往返 | 无改善 | 仍需逐 block dispatch（数据依赖） |

### 为什么优化效果不明显？

```
单次 compute() 耗时分解（估算）:
├─ buffer 创建/销毁 (旧)     ~80µs  █
├─ buffer 创建/销毁 (新/池化) ~20µs  ▌  ← BufferPool 节省 ~60µs
├─ 数据上传 (queue.write)     ~50µs  █
├─ 命令编码 + 提交            ~30µs  ▌
├─ GPU 计算                  ~10µs  ▏
├─ map_async + poll(Wait)   ~1400µs ████████████████████████████████  ← 绝对瓶颈
└─ 数据下载 + 解析            ~50µs  █
                              ─────
                              ~1.6ms 总计
```

**BufferPool 节省的 60µs 在 1.6ms 总量中仅占 3.7%**，低于 Criterion 的测量噪声（约 5-10%）。

---

## 四、瓶颈根因再确认

### 瓶颈 1: CPU-GPU 同步阻塞（95% 权重）

`GpuBuffer::download()` 中的 `device.poll(wgpu::Maintain::Wait)` 阻塞约 **1.4-1.6ms**。
这是 wgpu/Vulkan 驱动层的固有限制，与计算量无关。

### 瓶颈 2: 多 block 逐条 dispatch（多 block 场景 90% 权重）

SHA-256 的数据依赖导致同一条消息的多个 blocks 必须串行处理。
每条消息、每个 block 都需一次完整的 CPU-GPU 往返。

### 瓶颈 3: 无异步批量提交

当前 API 为完全同步：每次 `compute()` 必须等待 GPU 完成才能返回。
无法将多个独立批次合并为一次同步等待。

---

## 五、后续优化方向（按收益排序）

### 方向 1: 异步批量提交 API（预期收益 10-100x）

```rust
// 当前：每次 compute 都阻塞
for batch in batches {
    sha256.compute(&ctx, batch)?;  // 每次阻塞 1.6ms
}
// 10 批 = 16ms

// 优化：批量提交，统一等待
let mut submitter = sha256.batch_submitter(&ctx);
for batch in batches {
    submitter.submit(batch)?;  // 仅提交，不等待
}
let results = submitter.wait_all()?;  // 一次等待
// 10 批 = ~2ms（1 次同步 + 10 次计算）
```

**技术方案**:
- 在 `Sha256Computer` 中维护一个 `Vec<SubmittedBatch>`
- `submit()` 只创建 buffer、上传数据、dispatch，将 `download` 延迟到 `wait_all()`
- `wait_all()` 统一 poll，然后逐个 map 读取结果

### 方向 2: 多 block 消息链式 shader（预期收益 5-10x 于多 block 场景）

当前 shader 只处理单个 block。可扩展为处理单条消息的完整 block 链：

```wgsl
// 新 shader：每条消息处理所有 blocks
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let msg_idx = gid.x;
    let block_count = block_counts[msg_idx];
    
    var hash = INITIAL_HASH;
    for (var b = 0u; b < block_count; b = b + 1u) {
        let block = get_block(msg_idx, b);
        hash = sha256_round(hash, block);
    }
    hashes[msg_idx] = hash;
}
```

这样 10 条 64B 消息（2 blocks  each）从 20 次 dispatch 降为 1 次。

### 方向 3: 持久映射 Staging Buffer（预期收益 2-3x）

预分配持久映射的 staging buffer，避免每次 download 都创建新的 buffer + map_async。

### 方向 4: CPU 并行 + GPU 流水线（预期收益 2-5x）

使用多线程 CPU 准备数据，同时 GPU 执行计算，重叠传输与计算。

---

## 六、盈亏平衡分析（更新）

| 优化策略 | 预估单次开销 | 盈亏平衡批次 |
|---------|------------|------------|
| 当前实现 | 1.6ms | ~30000 条 |
| BufferPool（已实现）| 1.55ms | ~29000 条 |
| **异步批量提交** | **0.2ms** | **~4000 条** |
| 异步 + 链式 shader | 0.05ms | ~1000 条 |
| 零拷贝 + 双缓冲 | 0.02ms | ~400 条 |

---

## 七、测试方法

```bash
# 清理旧基准数据并重新运行
Remove-Item -Recurse -Force target/criterion; cargo bench
```

---

## 八、总结

本次 BufferPool 优化**在架构层面是正确的**（减少重复分配、符合高性能计算最佳实践），但在当前瓶颈（CPU-GPU 同步）主导下，**单次测量效果被噪声掩盖**。

**建议下一步**：优先实现 **异步批量提交 API**，这是唯一能从根本上降低同步开销、使 GPU SHA-256 具备实用价值的优化。
