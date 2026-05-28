# SHA-256 GPU 计算引擎性能测试报告

> 测试日期: 2026-05-18
> 测试平台: Windows (wgpu Vulkan 后端)
> 测试工具: Criterion.rs (100 samples, 优化构建)

---

## 一、核心结论

| 场景 | GPU 表现 | 与 CPU 对比 | 结论 |
|------|---------|------------|------|
| 单条短消息 (32B) | ~1.51ms | CPU 52ns，**慢 29000x** | GPU 不适合单条处理 |
| 1000 条批量 (32B) | ~1.69ms | CPU 53µs，**慢 32x** | 批量仍不足抵消开销 |
| 10000 条批量 (32B) | ~3.88ms | CPU 523µs，**慢 7.4x** | 大规模批量开始显现优势 |
| 多 block 消息 (64B) | ~31.3ms/10条 | CPU 788ns/条，**慢 40000x** | 逐条 dispatch 是致命瓶颈 |
| 空 dispatch | ~122µs | - | 单次调度纯开销 |

**关键发现**: GPU 计算本身极快，但 **CPU-GPU 同步开销（~1.5ms/次）** 是压倒性瓶颈。只有当单次提交处理 >=50000 条消息时，GPU 才可能超越 CPU。

---

## 二、详细测试结果

### 2.1 单 block 批量处理 (消息长度 32B)

| 消息数量 | GPU 耗时 | 每条耗时 | CPU 耗时 | 每条耗时 | GPU/CPU 比值 |
|---------|---------|---------|---------|---------|-------------|
| 1 | 1.58ms | 1580µs | 52ns | 52ns | **30384x** |
| 10 | 1.58ms | 158µs | 511ns | 51ns | **3092x** |
| 100 | 1.71ms | 17.1µs | 5.14µs | 51ns | **333x** |
| 1000 | 1.86ms | 1.86µs | 53.3µs | 53ns | **35x** |
| 10000 | 3.88ms | 0.39µs | 523µs | 52ns | **7.4x** |

**分析**:
- GPU 固定开销约 **1.5ms**（与消息数量无关）
- 当 N=1 时，99.9% 时间花在调度+同步+下载
- 当 N=10000 时，计算部分开始摊薄固定开销，但仍慢 7.4 倍
- 盈亏平衡点估算: 需要 ~50000 条/批次 GPU 才能超越 CPU

### 2.2 多 block 消息处理 (10 条消息)

| 消息长度 | GPU 耗时 | CPU 耗时 | GPU/CPU 比值 |
|---------|---------|---------|-------------|
| 64B (2 blocks) | 31.3ms | 7.88µs | **3972x** |
| 128B (3 blocks) | 55.3ms | 10.4µs | **5317x** |
| 256B (5 blocks) | 81.2ms | 16.6µs | **4892x** |
| 1024B (17 blocks)| 275ms | 55.6µs | **4946x** |

**分析**:
- 多 block 消息采用 **逐条、逐 block dispatch**，每次都要承受完整的 CPU-GPU 往返开销
- 64B 消息需 2 次 dispatch，每次约 15ms，说明多 block 路径存在额外问题
- 这是当前实现的最大性能陷阱

### 2.3 GPU 开销拆解

| 测试项 | 耗时 | 说明 |
|-------|------|------|
| 空 dispatch (无计算) | 122µs | 纯命令提交+同步开销 |
| 单消息端到端 | 1.51ms | 数据准备 + 调度 + 计算 + 下载 |
| 1000 消息端到端 | 1.69ms | 批量摊薄后，固定开销仍占 90%+ |

**推算**:
- 单次完整往返（含数据上传、调度、下载）≈ **1.5ms**
- 其中空 dispatch 约 122µs，剩余 ~1.38ms 花在 buffer 创建、数据拷贝、map_async 等待

### 2.4 Pipeline 缓存

| 操作 | 耗时 |
|------|------|
| 首次创建 (缓存未命中) | ~5.2µs |
| 后续创建 (缓存命中) | ~5.2µs |

**分析**:
- 由于 `Sha256Computer::new` 在基准测试中被内层循环调用，实际测试的是缓存命中路径
- 5.2µs 的耗时说明缓存机制工作良好，Arc 克隆开销极低
- 这不是性能瓶颈

### 2.5 Workgroup Size 对比 (10000 条 32B 消息)

| Workgroup Size | 耗时 | 相对性能 |
|---------------|------|---------|
| 64 | 3.61ms | 基准 |
| 128 | 3.88ms | -7% |
| 256 | 3.84ms | -6% |
| 512 | 3.64ms | -1% |

**分析**:
- 在当前瓶颈（CPU-GPU 同步）主导下，workgroup size 对整体性能影响 **< 10%**
- 差异被固定开销淹没，无法体现 GPU 内部并行效率差异
- 当未来优化掉同步开销后，workgroup size 调优将变得重要

---

## 三、性能瓶颈根因分析

### 瓶颈 1: CPU-GPU 同步阻塞 (权重: 90%)

当前 `GpuBuffer::download()` 使用同步等待:
```rust
staging.slice(..).map_async(...);
device.poll(wgpu::Maintain::Wait);  // 阻塞 CPU 直到 GPU 完成
```

每次 `compute()` 调用都要经历:
1. 创建 input/output/staging 缓冲区
2. 上传数据到 GPU
3. 提交命令队列
4. **阻塞等待 GPU 完成**
5. 映射 staging buffer 读取结果

这个往返在 Vulkan 驱动层约需 **1-2ms**，与计算量无关。

### 瓶颈 2: 多 block 逐条处理 (权重: 80% 于多 block 场景)

```rust
for block in blocks.iter() {
    // 每条消息、每个 block 都创建新 buffer + 新 dispatch + 新下载
    let input_buffer = GpuBuffer::from_data(...);
    self.pipeline.dispatch(...);
    let result = output_buffer.download(...)?;
}
```

64B 消息 = 2 blocks = 2 次完整往返 ≈ 3ms × 10 条 = 30ms+

### 瓶颈 3: Buffer 重复分配 (权重: 20%)

每次调用都创建新的 `GpuBuffer`，没有复用机制:
- 小批量时分配开销占比高
- 大批量时分配 + 释放压力增大

---

## 四、优化建议

### 高优先级 (预期收益 10-100x)

#### 1. 异步 API 设计 (收益: 100x+ 吞吐)

将同步 `compute()` 改为异步 `compute_async()`，允许 CPU 批量提交后统一等待:

```rust
// 当前: 同步阻塞
let hashes = sha256.compute(&ctx, &messages)?;  // 阻塞 1.5ms

// 优化: 异步批量提交
let mut batch = sha256.begin_batch(&ctx);
for chunk in messages.chunks(10000) {
    batch.submit(chunk)?;  // 仅提交，不等待
}
let all_hashes = batch.wait_all()?;  // 一次等待所有结果
```

这样 10 次提交只需 1 次同步等待，10 条消息从 15ms 降至 ~2ms。

#### 2. 多 block 批量打包 (收益: 100x+ 于多 block 场景)

将多条消息的所有 blocks 打包到一个大 buffer，单次 dispatch:

```rust
// 当前: 每条消息每个 block 一次 dispatch
for msg in messages {
    for block in msg.blocks() {
        dispatch(1, 1, 1);  // 致命
    }
}

// 优化: 所有消息的所有 blocks 打包成一次 dispatch
let total_blocks = messages.iter().map(|m| m.block_count()).sum();
let packed = pack_all_blocks(&messages);
dispatch(total_blocks, 1, 1);  // 单次！
```

#### 3. Buffer 池化复用 (收益: 2-5x 小批量)

预分配固定大小的 buffer 池，避免每次创建/销毁:

```rust
struct BufferPool {
    input_pools: Vec<GpuBuffer>,
    output_pools: Vec<GpuBuffer>,
    staging_pools: Vec<GpuBuffer>,
}
```

### 中优先级 (预期收益 2-5x)

#### 4. 持久映射缓冲区 (Persistent Mapped Buffer)

使用 `mapped_at_creation: true` 或 `BufferUsages::MAP_WRITE` 减少上传开销。

#### 5. 双缓冲流水线

CPU 准备第 N+1 批数据的同时，GPU 计算第 N 批，重叠计算与传输。

### 低优先级 (预期收益 < 2x)

#### 6. Workgroup Size 微调

当前瓶颈不在 GPU 内部，workgroup size 优化效果不明显。待异步优化完成后再评估。

#### 7. Shader 内联优化

WGSL 中 SHA-256 轮函数可进一步内联展开，减少寄存器压力。

---

## 五、与 CPU 的盈亏平衡分析

假设单次同步开销为 **1.5ms**，CPU 单核吞吐为 **~50ns/条**:

| 优化策略 | 预估单次开销 | 盈亏平衡批次大小 |
|---------|------------|----------------|
| 当前实现 | 1.5ms | ~30000 条 |
| Buffer 池化 | 0.8ms | ~16000 条 |
| 异步批量提交 | 0.15ms | ~3000 条 |
| 异步 + 池化 | 0.05ms | ~1000 条 |
| 零拷贝持久映射 | 0.02ms | ~400 条 |

**结论**: 只有在实现异步批量提交后，GPU SHA-256 才能在千级批量下具备实用价值。

---

## 六、测试方法说明

```bash
cargo bench
```

测试覆盖:
- `sha256_single_block_batch`: 单 block 消息批量 GPU vs CPU
- `sha256_multi_block`: 多 block 消息 GPU vs CPU
- `sha256_pipeline_cache`: 管线创建缓存效率
- `sha256_per_message_overhead`: 不同消息长度的 1000x 批量
- `sha256_workgroup_size`: workgroup size 64/128/256/512 对比
- `sha256_gpu_overhead`: 空 dispatch / 单消息 / 1000 消息端到端

---

## 七、下一步行动

1. **实现异步批量 API** (`compute_async` + `BatchHandle`)
2. **实现多 block 消息打包 dispatch**
3. **添加 BufferPool 减少分配开销**
4. **重新跑基准测试验证优化效果**
