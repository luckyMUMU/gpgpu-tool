# wgpu Rust 并行计算与哈希计算实现指南

> 整理自 wgpu_examples 文档（2026-05-17）
> 来源：`D:\Code\AI-note\20-Knowledge\01-Programming\wgpu_examples_docs\`

---

## 1. wgpu 并行计算基础

### 1.1 Workgroup 概念

wgpu 的并行计算基于 **Workgroup（工作组）** 模型：

- `@workgroup_size(size_x, size_y, size_z)` 定义每个 workgroup 的线程数
- 总调用次数 = `workgroup_count * workgroup_size`
- 每个线程通过 `global_invocation_id` 获取唯一 ID

相关示例：`[[hello_workgroups]]`

### 1.2 Storage Buffer

GPU 并行计算的核心数据结构：

```rust
// 创建 storage buffer（可读写）
let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("Storage Buffer"),
    size: (data.len() * std::mem::size_of::<u32>()) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});
```

相关示例：`[[big_compute_buffers]]`（大 buffer 分块处理）

### 1.3 Compute Shader 基本结构

WGSL 计算着色器模板：

```wgsl
struct Data {
    values: array<u32>,
};

@group(0) @binding(0)
var<storage, read_write> data: Data;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    // 并行计算逻辑
    data.values[idx] = compute_hash(idx);
}
```

---

## 2. 相关示例模式分析

### 2.1 hello_workgroups — Workgroup 调度

**关键点**：
- 演示 `@workgroup_size(x, y, z)` 的含义
- 如何计算全局 invocation ID
- Workgroup 数量与 size 的关系

**适用场景**：需要精细控制并行粒度的计算

### 2.2 repeated_compute — 多 Pass 计算

**关键点**：
- 多次 dispatch 同一个 compute shader
- 中间结果写回 storage buffer
- `WgpuContext` 结构体封装 device/queue/buffer 管理

**适用场景**：需要多次迭代的计算（如哈希链式计算）

相关结构：`[[repeated_compute#struct.WgpuContext|WgpuContext]]`

### 2.3 big_compute_buffers — 大 Buffer 分块

**关键点**：
- 单个 Buffer 大小有限制（`MAX_BUFFER_SIZE`）
- 需要分块（chunks）处理大数组
- `calculate_chunks()` 计算需要的 dispatch 次数
- `execute_gpu_inner()` 分批执行

**常量定义**：
- `MAX_BUFFER_SIZE`: 单个 buffer 最大字节数
- `MAX_DISPATCH_SIZE`: 单次 dispatch 最大工作数

相关函数：
- `[[big_compute_buffers#fn.calculate_chunks|calculate_chunks]]`
- `[[big_compute_buffers#fn.execute_gpu|execute_gpu]]`

**适用场景**：大规模并行哈希计算（> 128MB 数据）

### 2.4 cooperative_matrix — 协同矩阵计算

**关键点**：
- 利用 GPU 的 Tensor Core（NVIDIA）或 AMX（Intel）
- 硬件加速的矩阵运算
- `adapter.cooperative_matrix_properties()` 查询硬件支持

**适用场景**：哈希计算中的矩阵运算（如 Merkle Tree 构建）

相关结构：`[[cooperative_matrix#struct.Dimensions|Dimensions]]`

### 2.5 render_with_compute — Compute Shader 渲染

**关键点**：
- 用 compute shader 直接写入 texture
- 不需要顶点着色器
- 适合通用并行计算范式

**适用场景**：哈希结果可视化、通用计算

---

## 3. 并行哈希计算实现思路

### 3.1 适用哈希算法

**GPU 友好哈希算法**（适合并行计算）：

| 算法 | 特点 | 适用场景 |
|------|------|-----------|
| **Blake3** | 高度并行化，原生支持 GPU | 大文件哈希、Merkle Tree |
| **SHA-256** | 标准算法，可并行（Merkle Tree） | 区块链、验证场景 |
| **XXHash** | 极快，适合 GPU 实现 |  checksum、去重 |
| **CityHash** | Google 出品，64 位优化 | 大数据哈希表 |

### 3.2 WGSL 实现示例（Blake3 风格）

```wgsl
// blake3_hash.wgsl
struct HashData {
    input: array<u8>,
    output: array<u32>,
};

@group(0) @binding(0)
var<storage, read_write> data: HashData;

// Blake3 混合函数（简化版）
fn blake3_mix(a: u32, b: u32) -> (u32, u32) {
    var va = a;
    var vb = b;
    // 简化轮函数
    va = va + vb;
    vb = va ^ vb;
    va = va + vb;
    return (va, vb);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let chunk_size = 64u;  // Blake3 块大小
    let start = idx * chunk_size;
    
    // 每个线程处理一个 64 字节块
    var h0: u32 = 0x6A09E667;
    var h1: u32 = 0xBB67AE85;
    
    // 简化哈希计算
    for (var i = 0u; i < chunk_size; i += 4u) {
        let word = load_u32(start + i);
        (h0, h1) = blake3_mix(h0, word);
    }
    
    data.output[idx * 8] = h0;
    data.output[idx * 8 + 1] = h1;
}
```

### 3.3 Rust 端调用代码

```rust
// main.rs - 并行哈希计算主流程
use wgpu;

struct HashComputePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    staging_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
}

impl HashComputePipeline {
    fn new(device: &wgpu::Device, data_size: u64) -> Self {
        // 1. 加载 WGSL shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hash Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blake3_hash.wgsl").into()),
        });
        
        // 2. 创建 storage buffer
        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hash Storage Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_DST 
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // 3. 创建 bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Hash Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // 4. 创建 pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Hash Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Hash Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        // 5. 创建 bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Hash Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage_buffer.as_entire_binding(),
                },
            ],
        });
        
        // 6. 创建 staging buffer（用于读回结果）
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: data_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bind_group,
            staging_buffer,
            storage_buffer,
        }
    }
    
    fn compute_hash(&self, input_data: &[u8]) -> Vec<u8> {
        // 1. 写入输入数据
        self.queue.write_buffer(&self.storage_buffer, 0, input_data);
        
        // 2. 创建 command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Hash Compute Encoder"),
        });
        
        // 3. Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hash Compute Pass"),
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            let workgroup_size = 256;
            let dispatch_count = (input_data.len() as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(dispatch_count, 1, 1);
        }
        
        // 4. 拷贝结果到 staging buffer
        encoder.copy_buffer_to_buffer(
            &self.storage_buffer,
            0,
            &self.staging_buffer,
            0,
            input_data.len() as u64,
        );
        
        // 5. 提交命令
        self.queue.submit(Some(encoder.finish()));
        
        // 6. 读取结果
        let buffer_slice = self.staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        
        let data = buffer_slice.get_mapped_range();
        let result = data.to_vec();
        drop(data);
        self.staging_buffer.unmap();
        
        result
    }
}
```

### 3.4 大文件哈希（分块处理）

参考 `[[big_compute_buffers]]` 模式：

```rust
fn hash_large_file(device: &wgpu::Device, queue: &wgpu::Queue, file_path: &str) -> Vec<u8> {
    let data = std::fs::read(file_path).expect("Failed to read file");
    let chunk_size = 128 * 1024 * 1024;  // 128MB per chunk
    
    let mut hasher = HashComputePipeline::new(device, chunk_size as u64);
    let mut final_hash = Vec::new();
    
    for chunk in data.chunks(chunk_size) {
        let chunk_hash = hasher.compute_hash(chunk);
        final_hash.extend_from_slice(&chunk_hash);
    }
    
    // 最终合并哈希（可选：再跑一次 hash 合并）
    final_hash
}
```

---

## 4. 性能优化建议

### 4.1 Workgroup 大小选择

- **NVIDIA GPU**: 推荐 `@workgroup_size(256)`
- **AMD GPU**: 推荐 `@workgroup_size(64)` 或 `128`
- **Intel GPU**: 推荐 `@workgroup_size(32)` 或 `64`

查询硬件属性：
```rust
let workgroup_size = device.limits().max_compute_workgroup_size;
```

### 4.2 Memory Access Pattern

- **Coalesced Access**: 相邻线程访问相邻内存地址
- **避免 Bank Conflict**: WGSL 中使用 `array<vec4<u32>>` 代替 `array<u32>`
- **Shared Memory**: 使用 `var<workgroup> shared_mem: array<u32, 256>` 减少全局内存访问

### 4.3 批量处理

参考 `[[repeated_compute]]` 模式：
```rust
// 一次 dispatch 处理多个哈希块
for _ in 0..num_passes {
    compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
}
```

### 4.4 异步流水线

```rust
// 使用 wgpu::PollType::Wait 避免阻塞
device.poll(wgpu::Maintain::Wait);
```

---

## 5. 完整示例：并行 SHA-256

```wgsl
// sha256_parallel.wgsl
struct Sha256Data {
    input: array<u8>,
    hash: array<u32, 8>,
};

@group(0) @binding(0)
var<storage, read_write> data: Sha256Data;

// SHA-256 常量（前 64 个质数立方根小数部分前 32 位）
let K: array<u32, 64> = array(
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, ...
);

fn sha256_compress(hash: ptr<function, array<u32, 8>>, block: ptr<function, array<u32, 16>>) {
    // SHA-256 压缩函数（简化）
    var a = (*hash)[0];
    var b = (*hash)[1];
    // ... 64 轮运算
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    var hash = array(
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    );
    
    // 每个线程处理一个 512-bit block
    var block: array<u32, 16>;
    for (var i = 0u; i < 16u; i++) {
        block[i] = load_u32(idx * 64 + i * 4);
    }
    
    sha256_compress(&hash, &block);
    
    // 写回结果
    for (var i = 0u; i < 8u; i++) {
        data.hash[idx * 8 + i] = hash[i];
    }
}
```

---

## 6. 参考资源

### wgpu_examples 文档

| 示例 | 说明 | 链接 |
|------|------|------|
| `[[hello_workgroups]]` | Workgroup 调度基础 | `hello_workgroups.md` |
| `[[repeated_compute]]` | 多 Pass 计算 | `repeated_compute.md` |
| `[[big_compute_buffers]]` | 大 Buffer 分块 | `big_compute_buffers.md` |
| `[[cooperative_matrix]]` | 矩阵并行计算 | `cooperative_matrix.md` |
| `[[render_with_compute]]` | Compute Shader 渲染 | `render_with_compute.md` |
| `[[storage_texture]]` | Storage Texture 使用 | `storage_texture.md` |

### 外部资源

- **Blake3 官方实现**: https://github.com/BLAKE3-team/BLAKE3
- **wgpu 文档**: https://wgpu.rs/
- **WGSL 规范**: https://www.w3.org/TR/WGSL/
- **Compute Shader 优化指南**: https://developer.nvidia.com/gpugems

---

## 7. 总结

1. **wgpu 并行计算核心**：Workgroup + Storage Buffer + Compute Shader
2. **哈希计算适用算法**：Blake3（原生并行）> SHA-256（Merkle Tree）> XXHash（快速 checksum）
3. **大文件处理**：参考 `big_compute_buffers` 分块处理
4. **性能优化**：合理选择 workgroup 大小、优化内存访问模式、使用 shared memory
5. **完整流程**：Rust 端准备数据 → 写入 Storage Buffer → Dispatch Compute Shader → 读回结果

**下一步**：
- 基于 `hello_workgroups` 理解 workgroup 调度
- 基于 `repeated_compute` 理解多 pass 计算
- 实现完整的 Blake3 或 SHA-256 并行版本

---

*报告生成时间：2026-05-18*
*基于 wgpu_examples 29.0.0 文档整理*
