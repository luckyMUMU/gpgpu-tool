实现一个基于 wgpu 的 Rust GPGPU 计算库时，良好的架构设计是兼顾易用与性能的关键。参考 `hash-shader`[https://github.com/RustyBamboo/hash-shader] 这类已有的 SHA256 实现，我们可以把它的核心模式提炼出来，形成一个可供复用的计算框架。

### 🏗️ 整体架构设计：能力层与业务层分离

为了兼顾易用性与可扩展性，架构可以清晰地分为两个层次：

*   **`gpgpu-core` (能力层)**：负责封装所有与 `wgpu` 交互的底层细节，对外提供一套稳定、安全的 GPGPU 抽象 API。
*   **`gpgpu-algos` (业务实现层)**：构建于能力层之上，用于实现具体的算法（如 SHA256），由核心库的消费者按需选择或实现。

项目的 Rust 模块划分如下：
```rust
// gpgpu-core/src/lib.rs
mod device;     // GPU 设备管理（初始化、适配器、队列）
mod buffer;     // 缓冲区管理（分配、上传、下载、映射）
mod shader;     // 着色器管理（编译、模块创建）
mod pipeline;   // 计算管线管理
mod kernel;     // 内核抽象（统一调度接口）
mod command;    // 命令调度与同步（编码、提交）

pub use device::GpuDevice;
pub use buffer::GpuBuffer;
pub use kernel::{Kernel, DispatchConfig};
// ... 其他公开导出
```
```rust
// gpgpu-algos/src/sha256.rs
use gpgpu_core::{GpuDevice, GpuBuffer, Kernel};

pub struct Sha256Kernel { /* ... */ }
impl Sha256Kernel {
    pub fn new(device: &mut GpuDevice) -> Result<Self, GpuError> { /* ... */ }
    pub fn hash(&self, input: &[u8]) -> Result<Vec<u8>, GpuError> { /* ... */ }
}
```

接下来，我们深入各层的具体设计。

### ⚙️ 能力层 (`gpgpu-core`) 的核心模块设计

能力层的设计目标是隐藏 `wgpu` 的复杂性，提供一套高层次的 GPGPU 编程抽象。

#### 1. 核心计算上下文 (`GpuContext`)
一个核心 `GpuContext` 结构体将作为所有 GPU 操作的入口，封装 `wgpu` 中关键的四个顶层对象：
*   `instance`: `wgpu::Instance`
*   `adapter`: `wgpu::Adapter`
*   `device`: `wgpu::Device`
*   `queue`: `wgpu::Queue`

它还需要管理一个**错误通道**，因为 `device` 创建时可能会产生运行时错误。能力层应提供一个类似 `poll_device_errors` 的方法，由用户在合适的时机（例如每帧或每次调用后）主动轮询，以保持 API 的安全性。

```rust
impl GpuContext {
    pub fn new() -> Result<Self, GpuError> { /* ... */ }
    pub fn poll_device_errors(&self) -> Result<(), GpuError> { /* ... */ }
}
```

#### 2. 缓冲区抽象 (`GpuBuffer<T>`)
这是一个关键的性能抽象。为了提高效率，需要对缓冲区进行精细化管理，区分其用途：
*   **Device-Local Buffer**: 驻留在 GPU 显存，读写速度最快，主要用于计算。
*   **Staging Buffer**: 位于 CPU 可访问的系统内存，专门用于 CPU 与 GPU 之间的数据传输。

利用 Rust 的生命周期和所有权，可以优雅地实现 `GpuBuffer<T>` 类型：
```rust
pub struct GpuBuffer<T> {
    // 当类型 T 可以是任意类型时，我们需要存储其长度和原始字节缓冲区
    buffer: wgpu::Buffer,
    len: usize,      // 元素个数
    _marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    // 创建一个仅 GPU 可见的缓冲区
    pub fn new(device: &wgpu::Device, usage: wgpu::BufferUsages, data: &[T]) -> Self { /* ... */ }

    // 从 CPU 读取数据，这需要一个 staging buffer 或调用 mapping
    pub async fn read(&self, queue: &wgpu::Queue, device: &wgpu::Device) -> Result<Vec<T>, GpuError> { /* ... */ }

    // 由 GpuKernel 在内部调用，用于设置为顶点/索引/存储缓冲区
    pub(crate) fn raw(&self) -> &wgpu::Buffer { &self.buffer }
}
```

**高级管理**：能力层还可以提供 `BufferPool`，用以复用同大小和用法的缓冲区，避免频繁分配和释放，这在高负载计算中能显著提升性能。

#### 3. 内核封装 (`Kernel` trait 与 `ComputePipelineManager`)
`Kernel` trait 是让业务层“可执行”的核心。能力层可以提供一个 `ComputePipelineManager` 来管理所有内核的管线，避免重复编译：

```rust
pub trait Kernel {
    // 定义着色器源代码
    fn source(&self) -> Cow<'static, str>;
    // 定义工作组的维度
    fn workgroup_size(&self) -> (u32, u32, u32);
    // 定义所需的缓冲区布局 (用于自动生成 BindGroupLayout)
    fn bind_group_layouts(&self) -> Vec<wgpu::BindGroupLayoutEntry>;
}

pub struct ComputePipelineManager {
    pipelines: HashMap<TypeId, wgpu::ComputePipeline>,
}

impl ComputePipelineManager {
    pub fn get_or_create<K: Kernel + 'static>(&mut self, device: &wgpu::Device, kernel: &K) -> &wgpu::ComputePipeline {
        // 使用 TypeId 作为 key，如果 pipeline 已存在则返回引用，否则创建新的
    }
}
```

这种设计允许业务层只需实现 `Kernel` trait，能力层就能处理管线创建、绑定组布局自动生成等繁琐的工作，减少样板代码。

#### 4. 命令调度与同步 (`GpuCommandBuffer`)
能力层应提供一个高层 API 来封装 wgpu 的命令记录和提交。设计模式可以是“构建器模式”（Builder Pattern）或“命令收集器”，允许用户记录一系列计算操作，最后统一提交，同时自动处理资源屏障。

```rust
let mut cmd_buf = GpuCommandBuffer::new(&device);
cmd_buf.begin();
cmd_buf.bind_kernel(&sha256_kernel);
cmd_buf.dispatch(workgroups);
cmd_buf.submit(&queue);
```

#### 5. 可选的运行时抽象 (`Runtime`)
对于需要支持多后端（如 CUDA, Metal, Vulkan）的场景，可以在能力层之上设计一个 `Runtime` trait，为上层业务提供统一的接口，实现“一次编写，多后端运行”的能力。`wgpu` 后端是其一，未来可以通过条件编译或特性标志支持 CUDA。

```rust
pub trait Runtime {
    type Buffer<T>;
    fn allocate<T: Clone>(&self, data: &[T]) -> Self::Buffer<T>;
    fn execute(&self, kernel: &dyn Kernel, buffers: &[&dyn AnyBuffer]) -> Result<(), RuntimeError>;
}

pub struct WgpuRuntime;
impl Runtime for WgpuRuntime {
    // 实现细节
}

pub struct CudaRuntime; // 特性标志开启时可用
```

### 🚀 业务实现层：以 SHA256 为例

能力层构建好后，在其之上实现具体算法会非常方便。例如 `Sha256Kernel`：

```rust
use gpgpu_core::{GpuDevice, GpuBuffer, Kernel, GpuError};
use std::borrow::Cow;

pub struct Sha256Kernel {
    // 可以缓存一些预编译的绑定组等
}

impl Kernel for Sha256Kernel {
    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(include_str!("sha256.wgsl"))
    }

    fn workgroup_size(&self) -> (u32, u32, u32) {
        (256, 1, 1) // 定义工作组大小，影响并行粒度和共享内存使用
    }

    fn bind_group_layouts(&self) -> Vec<wgpu::BindGroupLayoutEntry> {
        // 1. 输入缓冲区: { binding: 0, visibility: Compute, ty: Buffer { type: ReadOnlyStorage } }
        // 2. 输出缓冲区: { binding: 1, visibility: Compute, ty: Buffer { type: Storage } }
    }
}

impl Sha256Kernel {
    pub fn new(device: &mut GpuDevice) -> Result<Self, GpuError> {
        // 预创建 kernel 所需的管线状态
        device.create_compute_pipeline_from_kernel(self)
    }

    pub fn hash(&self, device: &mut GpuDevice, input: &[u8]) -> Result<Vec<u8>, GpuError> {
        let input_buffer = GpuBuffer::new(device, input);
        let output_buffer = GpuBuffer::new_uninit(device, 32); // SHA256 输出固定 32 字节

        let mut cmd_buf = device.create_command_buffer();
        cmd_buf.begin();
        cmd_buf.bind_kernel(self);
        cmd_buf.bind_buffers(&[&input_buffer, &output_buffer]);
        let workgroups = calculate_workgroups(input.len(), self.workgroup_size().0);
        cmd_buf.dispatch(workgroups);
        cmd_buf.submit_and_wait();

        output_buffer.read()
    }
}
```

### ⚖️ 关键设计权衡与最佳实践

1.  **安全性**：利用 Rust 的所有权系统，可以创建一种机制，确保缓冲区在使用期间不会被意外释放或修改。例如，`GpuKernel` 可以“借用” `GpuBuffer` 作为其执行的一部分，编译时就能检查资源冲突。
2.  **同步策略**：`GpuCommandBuffer` 的 `submit_and_wait` 使用同步等待，保证数据正确性，但会阻塞调用线程。对于不需要立即结果的场景，可以设计 `submit_async` 方法返回一个 `Future`，允许并发执行多个计算任务。
3.  **内存占用**：`BufferPool` 必须设定大小上限，并实现 LRU 淘汰策略，避免积累过多未用缓冲区。
4.  **前端依赖**：除了 `wgpu = "0.20"` 之外，核心能力库还应依赖 `bytemuck` 和 `encase` 来处理数据对齐和转换，提升易用性。
5.  **团队协作扩展**：为了未来多人协作开发，可以在能力层之上构建一个独立的 CLI 工具（如 `gpgpu-cli`），用于测试和调试核心库的功能。
6.  **着色器语言路线选择**：
    *   **纯 WGSL**：推荐优先选择。这是 wgpu 的官方着色语言，跨平台支持最好，且工具链成熟。
    *   **Rust-gpu**：备选，可实现在 Rust 中编写着色器并编译为 SPIR-V，但需要 Nightly Rust 和额外编译步骤，对新手不友好。

这个设计方案可以将 `wgpu` 的复杂性隐藏在能力层之下，为上层业务算法（如 SHA256）提供简洁、安全的 API。通过精心设计的缓冲区管理和内核抽象，可以在性能和易用性之间取得平衡，并确保未来能够顺利扩展更多功能。
