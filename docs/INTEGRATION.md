# 集成指南

> **版本**: 0.3.0 | **更新日期**: 2026-06-16

本文档说明如何将 `gpgpu-tool` 集成到第三方 Rust 项目中。

---

## 1. 集成方式

### 1.1 从 Git 仓库集成（推荐）

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", tag = "v0.3.0" }
```

或指定分支：

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", branch = "master2" }
```

### 1.2 从本地路径集成

如果已克隆仓库到本地：

```toml
[dependencies]
gpgpu-tool = { path = "../wgpu-tool" }
```

### 1.3 从 .crate 文件集成

如果已有打包好的 `gpgpu-tool-0.3.0.crate` 文件：

```bash
# 解压到本地目录
tar -xzf gpgpu-tool-0.3.0.crate -C ./vendor/

# 在 Cargo.toml 中引用
```

```toml
[dependencies]
gpgpu-tool = { path = "./vendor/gpgpu-tool-0.3.0" }
```

### 1.4 从 crates.io 集成（未来）

当发布到 crates.io 后：

```toml
[dependencies]
gpgpu-tool = "0.3"
```

---

## 2. Feature Flags

| Feature | 默认 | 说明 |
|---------|------|------|
| `cpu-fallback` | ✅ 开启 | CPU 降级实现（依赖 `sha2` crate） |
| `image` | ❌ 关闭 | `image` crate 集成，支持从文件加载图像 |
| `pdq` | ❌ 关闭 | PDQ 感知哈希（DCT 频域，256-bit） |

### 2.1 最小依赖（仅 GPU 核心）

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", tag = "v0.3.0", default-features = false }
```

### 2.2 图像处理支持

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", tag = "v0.3.0", features = ["image"] }
```

### 2.3 PDQ 哈希支持

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", tag = "v0.3.0", features = ["pdq"] }
```

### 2.4 全功能

```toml
[dependencies]
gpgpu-tool = { git = "https://github.com/solo-king/wgpu-tool", tag = "v0.3.0", features = ["image", "pdq"] }
```

---

## 3. 快速入门示例

### 3.1 SHA-256 并行哈希

```rust
use gpgpu_tool::{GpuContext, Sha256Computer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建 GPU 上下文（一次性，复用）
    let mut ctx = GpuContext::new_sync()?;

    // 创建 SHA-256 计算器
    let sha256 = Sha256Computer::new(&mut ctx)?;

    // 批量计算哈希（GPU 并行）
    let messages: Vec<Vec<u8>> = vec![
        b"hello".to_vec(),
        b"world".to_vec(),
        b"gpu".to_vec(),
    ];
    let hashes: Vec<[u8; 32]> = sha256.compute(&ctx, &messages)?;

    for (msg, hash) in messages.iter().zip(hashes.iter()) {
        println!("{:?} -> {}", msg, hex::encode(hash));
    }

    Ok(())
}
```

### 3.2 感知图像哈希

```rust
use gpgpu_tool::{
    GpuContext,
    tasks::phasher::{PerceptualHasher, HashAlgorithm},
    tasks::hash_common::HashSize,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ctx = GpuContext::new_sync()?;

    // 创建感知哈希计算器（Gradient 算法，hash_size=32）
    let hasher = PerceptualHasher::with_hash_size(
        &mut ctx,
        HashAlgorithm::Gradient,
        HashSize::new(32),
    )?;

    // 输入：灰度像素数组 + 尺寸
    let images: Vec<Vec<u8>> = vec![vec![128u8; 256 * 256]];
    let dimensions: Vec<(u32, u32)> = vec![(256, 256)];

    // 计算 1024-bit 哈希
    let hashes: Vec<u64> = hasher.compute(&ctx, &images, &dimensions)?;

    println!("Hash: {:016x}", hashes[0]);

    Ok(())
}
```

### 3.3 GPU 汉明距离匹配

```rust
use gpgpu_tool::{GpuContext, GpuHashMatcherBytes, HashBytes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ctx = GpuContext::new_sync()?;
    let matcher = GpuHashMatcherBytes::new(&mut ctx)?;

    // 准备 1024-bit 哈希数据（每条 128 字节）
    let hashes: Vec<HashBytes> = vec![
        HashBytes::from_bytes(vec![0u8; 128]),
        HashBytes::from_bytes(vec![1u8; 128]),
        // ... 更多哈希
    ];

    // GPU 计算最近邻（阈值=20）
    let threshold = 20u32;
    let nearest = matcher.find_nearest_neighbors(&ctx, &hashes, &hashes, threshold)?;

    for (idx, (nn_idx, dist)) in nearest.iter().enumerate() {
        if *dist <= threshold {
            println!("Hash {} nearest: idx={}, dist={}", idx, nn_idx, dist);
        }
    }

    Ok(())
}
```

### 3.4 BK-tree 相似度搜索

```rust
use gpgpu_tool::tasks::bktree_bytes::BkTreeBytes;
use gpgpu_tool::HashBytes;

fn main() {
    // 构建 BK-tree
    let hashes: Vec<HashBytes> = vec![
        HashBytes::from_bytes(vec![0u8; 128]),
        HashBytes::from_bytes(vec![1u8; 128]),
        // ... 更多哈希
    ];
    let tree = BkTreeBytes::from_hashes(hashes.clone());

    // 查询相似哈希（Hamming 距离 ≤ 5）
    let query = HashBytes::from_bytes(vec![0u8; 128]);
    let similar = tree.find(&query, 5);

    for (hash, dist) in similar {
        println!("Found similar hash at distance {}", dist);
    }
}
```

---

## 4. 性能建议

### 4.1 GPU 批量处理

| 场景 | 建议批量大小 | 原因 |
|------|-------------|------|
| SHA-256 | ≥ 1000 | GPU dispatch 开销 ~1.6ms |
| 感知哈希 | ≥ 100 | 充分利用 GPU 并行 |
| 汉明距离匹配 | ≥ 1000 | GPU 最近邻比 CPU 快 778x |

### 4.2 GpuContext 复用

```rust
// ✅ 正确：创建一次，复用
let mut ctx = GpuContext::new_sync()?;
let sha256 = Sha256Computer::new(&mut ctx)?;
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean)?;
// 所有计算器共享同一个 ctx

// ❌ 错误：每次操作创建新 ctx
for msg in messages {
    let ctx = GpuContext::new_sync()?; // 每次创建开销大
    let sha256 = Sha256Computer::new(&mut ctx)?;
    // ...
}
```

### 4.3 GPU vs CPU 选择

| 任务规模 | 推荐方式 |
|---------|---------|
| SHA-256 < 100 条 | CPU（`sha2` crate） |
| SHA-256 ≥ 1000 条 | GPU `Sha256Computer` |
| 感知哈希 < 10 张 | CPU（`phasher_cpu`） |
| 感知哈希 ≥ 100 张 | GPU `PerceptualHasher` |
| 匹配 < 1000 条 | CPU BK-tree |
| 匹配 ≥ 10000 条 | GPU `GpuHashMatcherBytes` |

---

## 5. 错误处理

### 5.1 GPU 不可用

```rust
use gpgpu_tool::{GpuContext, GpuError};

match GpuContext::new_sync() {
    Ok(mut ctx) => {
        // GPU 可用，使用 GPU 实现
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean)?;
        // ...
    }
    Err(GpuError::NoAdapter) => {
        // GPU 不可用，降级到 CPU
        eprintln!("GPU not available, using CPU fallback");
        let hasher_cpu = gpgpu_tool::tasks::phasher_cpu::PHasherCpu::new();
        // ...
    }
    Err(e) => return Err(e.into()),
}
```

### 5.2 Feature Flag 检查

```rust
// 检查 cpu-fallback feature 是否启用
#[cfg(feature = "cpu-fallback")]
{
    let cpu_hasher = gpgpu_tool::tasks::phasher_cpu::PHasherCpu::new();
}

#[cfg(not(feature = "cpu-fallback"))]
{
    eprintln!("CPU fallback not available, GPU required");
}
```

---

## 6. 依赖项

### 6.1 核心依赖（始终包含）

| Crate | 版本 | 说明 |
|-------|------|------|
| `wgpu` | 24 | GPU 计算后端 |
| `pollster` | 0.4 | 异步执行器 |
| `bytemuck` | 1 | 内存布局转换 |
| `thiserror` | 2 | 错误类型定义 |
| `log` | 0.4 | 日志接口 |
| `smallvec` | 1 | 小数组优化 |

### 6.2 可选依赖

| Crate | Feature | 说明 |
|-------|---------|------|
| `image` | `image` | 图像加载/处理 |
| `sha2` | `cpu-fallback` | SHA-256 CPU 实现 |

---

## 7. 平台支持

| 平台 | GPU 后端 | 状态 |
|------|---------|------|
| Windows | Vulkan / DX12 | ✅ 支持 |
| macOS | Metal | ✅ 支持 |
| Linux | Vulkan | ✅ 支持 |
| Web | WebGPU | ⚠️ 实验性 |

---

## 8. 打包文件位置

```
target/package/gpgpu-tool-0.3.0.crate  # 426KB（183 个文件）
```

---

## 9. 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| 0.3.0 | 2026-06-16 | GPU Hash Matcher + 变长哈希 + 4096-bit 支持 |
| 0.2.0 | 2026-05-24 | 感知哈希 6 算法 + GPU resize + BK-tree |
| 0.1.0 | 2026-05-20 | SHA-256 GPU 并行哈希 |