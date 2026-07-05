use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use wgpu::{
    BindGroupLayout, BufferUsages, CommandEncoder,
    ComputePipeline as WgpuComputePipeline, Device, Queue,
};

use crate::buffer::{BufferUsage, GpuBuffer};
use crate::error::GpuError;

/// Push Constant 最大允许大小（字节）。
///
/// WebGPU 规范要求 Push Constant 不超过 128 字节，
/// 超过此限制会导致 GPU 驱动错误。
const PUSH_CONSTANT_MAX_SIZE: u32 = 128;

/// 绑定类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingType {
    StorageReadOnly,
    StorageReadWrite,
    Uniform,
}

impl BindingType {
    /// 返回此绑定类型对应的 wgpu BufferBindingType。
    pub fn buffer_binding_type(&self) -> wgpu::BufferBindingType {
        match self {
            BindingType::StorageReadOnly => wgpu::BufferBindingType::Storage { read_only: true },
            BindingType::StorageReadWrite => wgpu::BufferBindingType::Storage { read_only: false },
            BindingType::Uniform => wgpu::BufferBindingType::Uniform,
        }
    }

    /// 返回此绑定类型对应的 BufferUsages。
    ///
    /// 委托到 [`BufferUsage::to_wgpu_usage`]，消除重复的标志位映射逻辑。
    pub fn buffer_usages(&self) -> BufferUsages {
        let (usage, include_copy_src) = match self {
            BindingType::StorageReadOnly => (BufferUsage::Storage, false),
            BindingType::StorageReadWrite => (BufferUsage::Storage, true),
            BindingType::Uniform => (BufferUsage::Uniform, false),
        };
        usage.to_wgpu_usage(include_copy_src)
    }
}

/// 管线描述符，定义绑定布局、入口点和 Push Constant 配置。
#[derive(Debug, Clone)]
pub struct PipelineDescriptor {
    pub bindings: Vec<BindingType>,
    pub wgsl: String,
    pub workgroup_size: [u32; 3],
    /// WGSL 着色器入口函数名，默认 `"main"`。
    pub entry_point: &'static str,
    /// Push Constant 数据大小（字节数），最大 128 字节。
    ///
    /// 设为 `None`（默认）时不使用 Push Constant；设为 `Some(size)` 时
    /// 在管线布局中声明对应大小的 Push Constant Range。
    pub push_constant_size: Option<u32>,
}

impl PipelineDescriptor {
    /// 默认 3-binding 布局（StorageReadOnly, StorageReadWrite, Uniform）。
    pub fn default_3_binding(wgsl: impl Into<String>, workgroup_size: [u32; 3]) -> Self {
        Self {
            bindings: vec![
                BindingType::StorageReadOnly,
                BindingType::StorageReadWrite,
                BindingType::Uniform,
            ],
            wgsl: wgsl.into(),
            workgroup_size,
            entry_point: "main",
            push_constant_size: None,
        }
    }

    /// 2-binding + Push Constant 布局（StorageReadOnly, StorageReadWrite）。
    ///
    /// 参数通过 Push Constant 传递，无需 Uniform buffer。
    /// `push_constant_size` 为 Push Constant 数据的字节数，必须为 4 的倍数。
    pub fn push_constant_2_binding(
        wgsl: impl Into<String>,
        workgroup_size: [u32; 3],
        push_constant_size: u32,
    ) -> Self {
        assert!(
            push_constant_size <= PUSH_CONSTANT_MAX_SIZE,
            "Push constant 大小不能超过 {} 字节，当前为 {}",
            PUSH_CONSTANT_MAX_SIZE,
            push_constant_size
        );
        assert!(
            push_constant_size.is_multiple_of(4),
            "Push constant 大小必须是 4 字节对齐，当前为 {}",
            push_constant_size
        );
        Self {
            bindings: vec![
                BindingType::StorageReadOnly,
                BindingType::StorageReadWrite,
            ],
            wgsl: wgsl.into(),
            workgroup_size,
            entry_point: "main",
            push_constant_size: Some(push_constant_size),
        }
    }
}

/// BindGroup 缓存上限。
const BIND_GROUP_CACHE_MAX_ENTRIES: usize = 64;

/// 缓存键类型：典型绑定数 2-3，SmallVec 内联避免堆分配。
type CacheKey = SmallVec<[usize; 8]>;

/// 双向链表空指针哨兵。
const NONE: u32 = u32::MAX;

/// LRU 双向链表节点，存储在 arena 中。
struct LruNode {
    key: CacheKey,
    value: Arc<wgpu::BindGroup>,
    prev: u32,
    next: u32,
}

/// O(1) LRU BindGroup 缓存。
///
/// 使用 `HashMap<CacheKey, u32>` + arena 分配的双向链表实现全 O(1) 操作：
/// - 查找：O(1) HashMap 查询
/// - 触摸：O(1) 链表节点移动到 MRU 端
/// - 淘汰：O(1) 移除 LRU 端尾节点
///
/// 相比旧方案（generation counter + O(n) 扫描），消除了线性扫描开销。
///
/// # 缓存有效性
///
/// 缓存 key 基于缓冲区对象的内存地址。当同一组 `GpuBuffer` 对象被多次
/// dispatch 时，缓存命中；当缓冲区被释放后重新分配，旧条目自然失效
/// （不会命中，最终被淘汰策略清除）。
struct BindGroupCache {
    /// key → arena slot index
    map: RefCell<HashMap<CacheKey, u32>>,
    /// 节点 arena（slot 复用）
    nodes: RefCell<Vec<Option<LruNode>>>,
    /// 空闲 slot 回收列表
    free: RefCell<Vec<u32>>,
    /// MRU 端（最近使用）链表头
    head: RefCell<u32>,
    /// LRU 端（最久未使用）链表尾
    tail: RefCell<u32>,
    /// 当前缓存条目数
    len: RefCell<usize>,
}

impl BindGroupCache {
    fn new() -> Self {
        Self {
            map: RefCell::new(HashMap::new()),
            nodes: RefCell::new(Vec::new()),
            free: RefCell::new(Vec::new()),
            head: RefCell::new(NONE),
            tail: RefCell::new(NONE),
            len: RefCell::new(0),
        }
    }

    /// 从 arena 分配一个 slot，优先复用空闲 slot。O(1) 操作。
    fn alloc_slot(nodes: &mut Vec<Option<LruNode>>, free: &mut Vec<u32>, node: LruNode) -> u32 {
        if let Some(idx) = free.pop() {
            nodes[idx as usize] = Some(node);
            idx
        } else {
            let idx = nodes.len() as u32;
            nodes.push(Some(node));
            idx
        }
    }

    /// 将 slot 对应节点移动到链表头（MRU 端）。O(1) 操作。
    fn move_to_head(
        nodes: &mut Vec<Option<LruNode>>,
        head: &mut u32,
        tail: &mut u32,
        slot: u32,
    ) {
        if *head == slot {
            return; // 已经是 MRU，无需移动
        }

        let (prev, next) = {
            let node = nodes[slot as usize].as_ref().unwrap();
            (node.prev, node.next)
        };

        // 从当前位置摘除（slot 非 head，故 prev 必有效）
        nodes[prev as usize].as_mut().unwrap().next = next;
        if next != NONE {
            nodes[next as usize].as_mut().unwrap().prev = prev;
        } else {
            *tail = prev; // slot 原为尾节点，更新 tail
        }

        // 插入链表头
        let old_head = *head;
        {
            let node = nodes[slot as usize].as_mut().unwrap();
            node.prev = NONE;
            node.next = old_head;
        }
        if old_head != NONE {
            nodes[old_head as usize].as_mut().unwrap().prev = slot;
        }
        *head = slot;
    }

    /// 获取或创建 BindGroup。缓存命中时 O(1) 更新 LRU 顺序并返回。
    fn get_or_create(
        &self,
        device: &Device,
        layout: &BindGroupLayout,
        buffers: &[&GpuBuffer],
    ) -> Arc<wgpu::BindGroup> {
        let key: CacheKey = buffers
            .iter()
            .map(|b| b.raw() as *const _ as usize)
            .collect();

        // 缓存命中：移动到 MRU 端并返回
        let hit = { self.map.borrow().get(&key).copied() };
        if let Some(slot) = hit {
            log::debug!("BindGroup 缓存命中: {} 个缓冲区", key.len());
            let mut nodes = self.nodes.borrow_mut();
            let mut head = self.head.borrow_mut();
            let mut tail = self.tail.borrow_mut();
            Self::move_to_head(&mut nodes, &mut head, &mut tail, slot);
            return Arc::clone(&nodes[slot as usize].as_ref().unwrap().value);
        }

        // 缓存未命中：创建新 BindGroup
        log::debug!("BindGroup 缓存未命中，创建新 BindGroup");
        let bg = Self::create_bind_group(device, layout, buffers);
        let arc = Arc::new(bg);

        {
            let mut map = self.map.borrow_mut();
            let mut nodes = self.nodes.borrow_mut();
            let mut free_list = self.free.borrow_mut();
            let mut head = self.head.borrow_mut();
            let mut tail = self.tail.borrow_mut();
            let mut len = self.len.borrow_mut();

            // LRU 淘汰：O(1) 移除尾节点直到低于上限
            while *len >= BIND_GROUP_CACHE_MAX_ENTRIES {
                let lru = *tail;
                if lru == NONE {
                    break;
                }

                let lru_key = nodes[lru as usize].as_ref().unwrap().key.clone();
                let lru_prev = nodes[lru as usize].as_ref().unwrap().prev;

                if lru_prev != NONE {
                    nodes[lru_prev as usize].as_mut().unwrap().next = NONE;
                } else {
                    *head = NONE; // 仅剩一个元素
                }
                *tail = lru_prev;

                map.remove(&lru_key);
                nodes[lru as usize] = None;
                free_list.push(lru);
                *len -= 1;

                log::debug!("LRU 淘汰 BindGroup 缓存条目");
            }

            // 在链表头插入新节点
            let new_node = LruNode {
                key: key.clone(),
                value: Arc::clone(&arc),
                prev: NONE,
                next: *head,
            };
            let slot = Self::alloc_slot(&mut nodes, &mut free_list, new_node);

            if *head != NONE {
                nodes[*head as usize].as_mut().unwrap().prev = slot;
            } else {
                *tail = slot; // 首个元素
            }
            *head = slot;

            map.insert(key, slot);
            *len += 1;
        }

        arc
    }

    fn create_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        buffers: &[&GpuBuffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.raw().as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cached_compute_bind_group"),
            layout,
            entries: &entries,
        })
    }

    fn clear(&self) {
        self.map.borrow_mut().clear();
        self.nodes.borrow_mut().clear();
        self.free.borrow_mut().clear();
        *self.head.borrow_mut() = NONE;
        *self.tail.borrow_mut() = NONE;
        *self.len.borrow_mut() = 0;
    }

    fn len(&self) -> usize {
        *self.len.borrow()
    }
}

/// 计算管线，封装 WGSL 着色器的编译、绑定组创建和 dispatch 调度。
///
/// 内部使用 `PipelineCache` 实现着色器级缓存，同一 WGSL 源码 + workgroup size
/// 组合只编译一次。同时内置 `BindGroupCache`，同一管线 + 缓冲区组合的
/// BindGroup 只创建一次，避免 GPU 驱动重复验证和编译。
///
/// # 使用方式
///
/// 通常不直接构造，而是通过 `GpuContext::get_or_create_pipeline()` 获取缓存实例：
///
/// ```ignore
/// use gpgpu_tool::{GpuContext, PipelineDescriptor};
///
/// let mut ctx = GpuContext::new_sync().unwrap();
/// let pipeline = ctx.get_or_create_pipeline(
///     &PipelineDescriptor::default_3_binding(
///         include_str!("path/to/shader.wgsl"),
///         [8, 8, 1],
///     ),
/// ).unwrap();
/// ```
pub struct ComputePipeline {
    pipeline: WgpuComputePipeline,
    bind_group_layout: BindGroupLayout,
    bindings: Vec<BindingType>,
    workgroup_size: [u32; 3],
    entry_point: &'static str,
    push_constant_size: Option<u32>,
    bind_group_cache: BindGroupCache,
    /// 可复用 Uniform 缓冲区，避免非 Push Constant 路径每次创建新缓冲区
    reusable_uniform: RefCell<Option<GpuBuffer>>,
}

impl ComputePipeline {
    /// 从管线描述符编译计算管线。
    ///
    /// `descriptor` 指定绑定布局、WGSL 源码和 workgroup 尺寸。
    pub fn create(device: &Device, descriptor: &PipelineDescriptor) -> Result<Self, GpuError> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(descriptor.wgsl.as_str().into()),
        });
        // 注意：wgpu 的 create_shader_module 在 WGSL 编译失败时 panic 而非返回 Result。
        // 如果未来 wgpu 版本提供 Result 接口，此处应返回 GpuError::ShaderCompile。
        // 当前通过 descriptor.wgsl 的编译期有效性保证安全。

        let entries: Vec<wgpu::BindGroupLayoutEntry> = descriptor
            .bindings
            .iter()
            .enumerate()
            .map(|(i, bt)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: bt.buffer_binding_type(),
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &entries,
        });

        let push_constant_ranges: Vec<wgpu::PushConstantRange> = descriptor
            .push_constant_size
            .map(|size| vec![wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..size,
            }])
            .unwrap_or_default();

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &push_constant_ranges,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(descriptor.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            bindings: descriptor.bindings.clone(),
            workgroup_size: descriptor.workgroup_size,
            entry_point: descriptor.entry_point,
            push_constant_size: descriptor.push_constant_size,
            bind_group_cache: BindGroupCache::new(),
            reusable_uniform: RefCell::new(None),
        })
    }

    /// 执行一次 GPU dispatch，内部创建 encoder、绑定资源和提交。
    ///
    /// 适用于独立调用的场景。对于批量提交场景使用
    /// [`encode_dispatch_into`](ComputePipeline::encode_dispatch_into)。
    ///
    /// BindGroup 通过内部缓存复用，同一管线 + 缓冲区组合只创建一次。
    ///
    /// `compute_units` 为 GPU 计算单元数量（可通过 [`GpuContext::compute_units`] 获取），
    /// 传入 0 表示跳过 occupancy 诊断。当总线程数远小于 CU 数量时输出低 occupancy 警告。
    pub fn dispatch(
        &self,
        device: &Device,
        queue: &Queue,
        buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
        push_constants: Option<&[u8]>,
        compute_units: u32,
    ) {
        check_occupancy(compute_units, self.workgroup_size, dispatch_count);

        let bind_group = self.bind_group_cache.get_or_create(
            device,
            &self.bind_group_layout,
            buffers,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &*bind_group, &[]);
            if let Some(data) = push_constants {
                let expected = self.push_constant_size.unwrap() as usize;
                assert_eq!(
                    data.len(),
                    expected,
                    "Push constant 数据长度 ({}) 与声明大小 ({}) 不匹配",
                    data.len(),
                    expected
                );
                pass.set_push_constants(0, data);
            }
            pass.dispatch_workgroups(dispatch_count[0], dispatch_count[1], dispatch_count[2]);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    #[doc(hidden)]
    pub fn encode_dispatch_into(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
        push_constants: Option<&[u8]>,
        compute_units: u32,
    ) {
        check_occupancy(compute_units, self.workgroup_size, dispatch_count);

        let bind_group = self.bind_group_cache.get_or_create(
            device,
            &self.bind_group_layout,
            buffers,
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &*bind_group, &[]);
            if let Some(data) = push_constants {
                let expected = self.push_constant_size.unwrap() as usize;
                assert_eq!(
                    data.len(),
                    expected,
                    "Push constant 数据长度 ({}) 与声明大小 ({}) 不匹配",
                    data.len(),
                    expected
                );
                pass.set_push_constants(0, data);
            }
            pass.dispatch_workgroups(dispatch_count[0], dispatch_count[1], dispatch_count[2]);
        }
    }

    /// 执行一次 GPU dispatch，自动根据管线配置选择 Push Constant 或 Uniform buffer 传递参数。
    ///
    /// 当管线使用 Push Constant 时，`params_bytes` 通过 `set_push_constants` 传递，
    /// `storage_buffers` 直接作为绑定组；当管线使用 Uniform buffer 时，自动创建
    /// Uniform 缓冲区并追加到绑定组末尾（匹配 `default_3_binding` 布局顺序）。
    ///
    /// 此方法封装了 `push_constant_size().is_some()` 的分支判断，消除调用方重复代码。
    pub fn dispatch_with_params(
        &self,
        device: &Device,
        queue: &Queue,
        params_bytes: &[u8],
        storage_buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
        compute_units: u32,
    ) {
        if self.push_constant_size.is_some() {
            self.dispatch(
                device,
                queue,
                storage_buffers,
                dispatch_count,
                Some(params_bytes),
                compute_units,
            );
        } else {
            let params_buffer = self.get_or_create_uniform_buffer(device, queue, params_bytes);
            let mut buffers: Vec<&GpuBuffer> = Vec::with_capacity(storage_buffers.len() + 1);
            buffers.extend(storage_buffers);
            buffers.push(&params_buffer);
            self.dispatch(device, queue, &buffers, dispatch_count, None, compute_units);
        }
    }

    /// 将带参数的 GPU dispatch 编码到共享 encoder 中（不提交）。
    ///
    /// 与 [`dispatch_with_params`](Self::dispatch_with_params) 功能相同，
    /// 但将命令编码到提供的 encoder 而非创建新 encoder 并立即提交。
    /// 适用于多阶段管线需要合并为单次 `queue.submit()` 的场景。
    pub fn encode_dispatch_with_params_into(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        params_bytes: &[u8],
        storage_buffers: &[&GpuBuffer],
        dispatch_count: [u32; 3],
        compute_units: u32,
    ) {
        if self.push_constant_size.is_some() {
            self.encode_dispatch_into(
                device,
                encoder,
                storage_buffers,
                dispatch_count,
                Some(params_bytes),
                compute_units,
            );
        } else {
            // encode 路径无 queue 参数，无法 write_buffer 复用，仍创建新缓冲区
            let params_buffer = GpuBuffer::from_bytes(device, params_bytes, BufferUsage::Uniform);
            let mut buffers: Vec<&GpuBuffer> = Vec::with_capacity(storage_buffers.len() + 1);
            buffers.extend(storage_buffers);
            buffers.push(&params_buffer);
            self.encode_dispatch_into(
                device,
                encoder,
                &buffers,
                dispatch_count,
                None,
                compute_units,
            );
        }
    }

    /// 获取或创建可复用 Uniform 缓冲区，并写入参数数据。
    ///
    /// 首次调用时创建缓冲区，后续调用复用已有缓冲区（通过 `queue.write_buffer` 更新内容）。
    /// 如果参数数据大于已有缓冲区，自动创建更大的缓冲区替换。
    fn get_or_create_uniform_buffer(&self, device: &Device, queue: &Queue, params_bytes: &[u8]) -> GpuBuffer {
        let required_size = params_bytes.len() as u64;
        let mut reusable = self.reusable_uniform.borrow_mut();

        match reusable.as_ref() {
            Some(existing) if existing.size() >= required_size => {
                // 复用已有缓冲区，仅更新内容
                queue.write_buffer(existing.raw(), 0, params_bytes);
                existing.clone()
            }
            _ => {
                // 创建新缓冲区（首次或参数更大时）
                let new_buffer = GpuBuffer::from_bytes(device, params_bytes, BufferUsage::Uniform);
                *reusable = Some(new_buffer.clone());
                new_buffer
            }
        }
    }

    /// 返回此管线的 workgroup 尺寸。
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }

    /// 返回此管线的 Push Constant 大小（字节数）。
    ///
    /// `None` 表示此管线不使用 Push Constant。
    pub fn push_constant_size(&self) -> Option<u32> {
        self.push_constant_size
    }

    /// 返回此管线的绑定类型列表。
    pub fn bindings(&self) -> &[BindingType] {
        &self.bindings
    }

    /// 返回此管线的 WGSL 入口函数名。
    pub fn entry_point(&self) -> &'static str {
        self.entry_point
    }

    /// 清空此管线的 BindGroup 缓存。
    ///
    /// 当缓冲区被释放或回收后，应调用此方法清除可能失效的缓存条目。
    /// 正常使用中无需手动调用——LRU 淘汰策略会自动清理最久未访问的条目。
    pub fn clear_bind_group_cache(&self) {
        self.bind_group_cache.clear();
    }

    /// 返回此管线当前 BindGroup 缓存条目数。
    pub fn bind_group_cache_len(&self) -> usize {
        self.bind_group_cache.len()
    }
}

/// 将 Push Constant 版 WGSL 转换为 Uniform buffer 回退版。
///
/// 替换 `var<push_constant>` 为 `@group(0) @binding(2) var<uniform>`，
/// 适用于不支持 Push Constant 的设备回退到 Uniform buffer 传递参数。
///
/// # 示例
///
/// ```text
/// // 输入: var<push_constant> params: Params;
/// // 输出: @group(0) @binding(2) var<uniform> params: Params;
/// ```
pub fn wgsl_push_constant_to_uniform(wgsl: &str) -> String {
    wgsl.replace("var<push_constant>", "@group(0) @binding(2) var<uniform>")
}

/// 检查 dispatch occupancy，当总线程数远小于 CU 数量时输出低 occupancy 警告。
///
/// 阈值：`total_threads < compute_units * 64`。
/// 仅在 `compute_units > 0` 时执行检查。
fn check_occupancy(compute_units: u32, workgroup_size: [u32; 3], dispatch_count: [u32; 3]) {
    if compute_units == 0 {
        return;
    }
    let threads_per_wg = workgroup_size[0] as u64
        * workgroup_size[1] as u64
        * workgroup_size[2] as u64;
    let total_workgroups = dispatch_count[0] as u64
        * dispatch_count[1] as u64
        * dispatch_count[2] as u64;
    let total_threads = threads_per_wg * total_workgroups;
    let threshold = compute_units as u64 * 64;
    if total_threads < threshold {
        log::warn!(
            "低 occupancy 警告: 总线程数 {} 远小于 CU×64={} (CU={}, wg_size={:?}, dispatch={:?})",
            total_threads,
            threshold,
            compute_units,
            workgroup_size,
            dispatch_count,
        );
    }
}
