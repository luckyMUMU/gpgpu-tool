# wgpu-compute-engine 二次审查报告

**日期**: 2026-05-23  
**范围**: 修复后全量代码库（hash_size 扩展 + 链式着色器 + resize 优化）

---

## 执行摘要

修复质量高，主线功能全部正确。发现 **2 个高危问题**（竞态条件 + 基准测试算法不一致）和若干中低优先级的测试覆盖缺口。

| 严重程度 | 数量 | 关键项 |
|----------|------|--------|
| 🔴 高 | 2 | BufferPool 竞态条件、bench 参考实现算法错误 |
| 🟠 中 | 3 | batch.rs 静默映射失败、缺少 hash_size>64 测试、链式着色器测试不足 |
| 🟡 低 | 3 | 未使用参数、BK-Tree 64bit 限制、重复 CPU 参考实现 |

---

## 🔴 高危问题

### 1. BufferPool 早期释放竞态条件

**位置**: `src/tasks/sha256.rs` — `submit_single_block_batch:493` 和 `submit_multi_block_message:550`

**问题**: input buffer 在共享 encoder **提交之前**就被释放回 pool。

```rust
// submit_single_block_batch 第 500 行
self.computer.buffer_pool.release(input_buffer.into_raw(), BufferUsage::Storage);
// ... encoder 尚未提交！在 wait_all() 才会 queue.submit()
```

**后果**: 如果 encoder 提交之前，后续 `submit()` 调用从 pool 获取**同一 buffer** 并 `write_buffer` 覆盖数据，则 GPU 将读取到错误数据。

**修复方案**: 将 input buffer 释放延迟到 `wait_all()`：
```rust
struct PendingBatch {
    input_buffer: Option<GpuBuffer>,  // 新增字段
    // ... 现有字段
}
```

### 2. benches 参考实现与 GPU WGSL 算法不一致

**位置**: `benches/common/hash_reference.rs`

| 函数 | benches 版本 | GPU WGSL | 影响 |
|------|-------------|----------|------|
| `mean_hash` | `u32` 均值 + `>` 比较 | `f32` 均值 + `>=` 比较 | 基准 CPU 数据完全不可比 |
| `median_hash` | 排序求中值 | 直方图求中值 | 同上 |

**后果**: 所有 6 个感知哈希 benchmark 的 CPU 对比数据均基于错误算法，GPU vs CPU 加速比报告不可信。

**修复**: 将 `benches/common/hash_reference.rs` 与 `tests/common/hash_reference.rs` 统一，或直接让 benchmarks 引用 tests 版本。

---

## 🟠 中等问题

### 3. `GpuBatchSubmitter::wait_all` 静默映射失败

**位置**: `src/batch.rs:130-138`

```rust
.map_async(wgpu::MapMode::Read, |result| {
    if let Err(e) = result {
        log::error!("批量 staging buffer 映射失败: {}", e);
    }
});
// 如果映射失败，下面 get_mapped_range() 会 panic
```

**修复**: 与 `Sha256BatchSubmitter` 对齐，使用 channel 验证映射成功后再 `get_mapped_range()`。

### 4. 缺少 hash_size > 64 的测试

- 所有测试默认使用 `HashBits::B64`，无 B128/B256 测试
- CPU 参考实现 `hash_reference.rs` 硬编码返回 `u64`，无法验证 >64 位输出
- WGSL 端已完整支持，Rust 端缺少验证

### 5. 链式着色器缺少显式测试

- `sha256_chained.wgsl` 仅在 64 字节测试中隐式命中
- 缺少：55/56 边界、128B、1024B 等消息的显式链式着色器验证

---

## 🟡 低优问题

### 6. 未使用参数 `_input_u32_count`

`src/tasks/hash_common.rs:145` — `compute_phash_from_gpu_buffer` 的 `_input_u32_count: usize` 完全未使用。

### 7. BK-Tree 硬编码 64-bit

`BkTree<u64>` / `hamming_distance(u64, u64)` — 若未来启用 >64-bit 哈希需升级。

### 8. 重复 CPU 参考实现

`real_image_hash_test.rs` 内嵌独立的 CPU 实现（cpu_mean_hash 等），与 `hash_reference.rs` 存在维护分歧风险。

---

## 已确认正确的部分

| 项目 | 状态 |
|------|------|
| `download()` panic 修复 | ✅ 正确的 channel 模式 |
| `from_raw/into_raw/raw` pub(crate) | ✅ 正确封装 |
| `release()` usage 参数修复 | ✅ 所有调用点一致 |
| `_instance` 注释 | ✅ 清晰准确 |
| 链式着色器算法 | ✅ Merkle-Damgard 正确，K 常量已验证 |
| WGSL hash_size_bits 一致性 | ✅ 6 个 shader 完全统一 |
| 输出缓冲区大小计算 | ✅ B64/B128/B256 均正确 |
| hash_u32s[8] 边界 | ✅ 无溢出风险 |
| 宏向后兼容性 | ✅ 默认 64-bit，无破坏性变更 |
| Resize u32-per-pixel | ✅ 转换正确，无溢出 |
| 测试通过率 | ✅ 全部 32 个测试通过，clippy 零警告 |

---

## 修复优先级

### 立即修复 (P0)
1. PendingBatch 添加 input_buffer 字段，延迟释放
2. 统一 benches 参考实现与 tests 版本

### 短期 (P1)
3. 修复 batch.rs wait_all 映射错误处理
4. 新增 B128 测试 + CPU 参考实现升级
5. 新增链式着色器边界测试

### 中期 (P2)
6. 移除 `_input_u32_count` 或添加校验
7. 统一 real_image_hash_test 的 CPU 实现

---

## 修复记录 (2026-05-23 17:00)

| 优先级 | 问题 | 修复 | 文件 |
|--------|------|------|------|
| 🔴 P0 | BufferPool 早期释放竞态 | `PendingBatch` 添加 `input_buffers_to_release`，延迟到 `wait_all` 释放 | `sha256.rs` |
| 🔴 P0 | bench 参考实现算法错误 | `mean_hash`: `u32>/` → `f32/>`=`; `median_hash`: sort → histogram | `benches/common/hash_reference.rs` |
| 🟠 P1 | batch.rs 静默映射失败 | 改用 channel 验证映射成功后再 `get_mapped_range()` | `batch.rs` |

**验证**: `cargo check` ✅, `cargo clippy` ✅ 零警告, SHA-256 8/8 测试通过 ✅
