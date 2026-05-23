---
name: "wgpu-compute-engine"
description: "GPU parallel compute engine using wgpu. Invoke when user needs GPU-accelerated SHA-256 hashing, perceptual image hashing, BK-tree similarity search, or wants to build custom GPU compute pipelines on top of wgpu."
---

# wgpu-compute-engine — Agent Usage Guide

Cross-platform GPU compute engine built on wgpu. Provides out-of-the-box GPU acceleration for CPU-intensive tasks without requiring WGSL shader authoring.

## When to Invoke This Skill

- User asks about GPU-accelerated hashing (SHA-256, perceptual image hashing)
- User needs image similarity detection or approximate nearest neighbor search
- User wants to build custom GPU compute pipelines using wgpu
- User mentions BK-tree, Hamming distance, phash, or GPU compute
- User asks about batch GPU processing or zero-copy data pipelines
- User needs to process large-scale image datasets (10K+ images) with GPU

## Architecture Overview

```
┌─── Business Layer (Tasks) ────────────────────────────┐
│  Sha256Computer    PerceptualHasher      BkTree       │
│  (SHA-256 hash)    (6 phash algorithms)  (ANN search) │
│                   + GpuResize (zero-copy)             │
└────────────┬──────────────┬──────────────┬────────────┘
             │              │              │
┌────────────▼──────────────▼──────────────▼────────────┐
│              Capability Layer (Core)                    │
│  GpuContext  GpuBuffer  BufferPool  ComputePipeline    │
│  GpuBatchSubmitter  GpuError                           │
└───────────────────────────────────────────────────────┘
```

**Rule**: Business layer depends on capability layer. No cross-dependencies between algorithms.

## Quick Reference: Public API

### Capability Layer (use directly)

| Type | Module | Purpose |
|------|--------|---------|
| `GpuContext` | root | GPU context: device, queue, pipeline cache. Entry point for ALL GPU ops |
| `GpuBuffer` | root | CPU↔GPU data transfer buffer |
| `BufferUsage` | root | Buffer type enum: `Storage` or `Uniform` |
| `BufferPool` | root | Size-tiered buffer reuse pool |
| `ComputePipeline` | root | Compute pipeline: shader compile + dispatch |
| `GpuBatchSubmitter` | root | Async batch submitter: merge multiple dispatches into one submit |
| `BatchJob` | root | Batch job descriptor |
| `GpuError` | root | Unified error type (9 variants, thiserror) |

### Business Layer (via `tasks::` submodule)

| Type | Module | Purpose |
|------|--------|---------|
| `Sha256Computer` | `tasks::sha256` | GPU parallel SHA-256 hashing |
| `PerceptualHasher` | `tasks::phasher` | Perceptual image hashing (6 algorithms) |
| `HashAlgorithm` | `tasks::phasher` | Algorithm selector enum |
| `BkTree` | `tasks::bktree` | BK-tree approximate nearest neighbor search |
| `hamming_distance` | `tasks::bktree` | Hamming distance between two u64 hashes |

### Internal Modules (#[doc(hidden)], avoid direct use)

| Module | Why hidden | Use instead |
|--------|-----------|-------------|
| `tasks::hash_common` | Internal trait + macros | `PerceptualHasher` |
| `tasks::gpu_resize` | Internal GPU resize impl | `PerceptualHasher::with_resize_mode(true)` |
| `tasks::mean_hash` etc. | Thin wrappers via macros | `PerceptualHasher::new(algo)` |

## Usage Patterns

### Pattern 1: SHA-256 Parallel Hashing

```rust
use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};

// Step 1: Create GPU context (do this ONCE, reuse everywhere)
let mut ctx = GpuContext::new_sync()?;

// Step 2: Create hasher (pipeline compiled once, cached)
let sha256 = Sha256Computer::new(&mut ctx)?;

// Step 3: Batch compute (send ALL messages at once for best GPU utilization)
let messages: Vec<Vec<u8>> = vec![b"hello".to_vec(), b"world".to_vec()];
let hashes: Vec<[u8; 32]> = sha256.compute(&ctx, &messages)?;
```

**Key rules**:
- Messages ≤ 55 bytes → single-block mode (fully parallel)
- Messages > 55 bytes → multi-block mode (sequential chain per message)
- Batch size ≥ 1000 for GPU to outperform CPU (dispatch overhead ~1.6ms)
- Use `Sha256Computer::batch_submit()` for async multi-batch workflows

### Pattern 2: Perceptual Image Hashing

```rust
use wgpu_compute_engine::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};

let mut ctx = GpuContext::new_sync()?;

// CPU resize path (default, good for small batches)
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean)?;

// Input: grayscale pixel arrays + dimensions (all same size for batch)
let images: Vec<Vec<u8>> = vec![vec![128u8; 256 * 256]];
let dimensions: Vec<(u32, u32)> = vec![(256, 256)];
let hashes: Vec<u64> = hasher.compute(&ctx, &images, &dimensions)?;
```

**Algorithm selection guide**:

| Algorithm | Size | Best for | Speed |
|-----------|------|----------|-------|
| `Mean` | 8×8 | General similarity, fast baseline | Fastest |
| `Median` | 8×8 | Robust to outliers (bright spots) | Fast |
| `Gradient` | 8×9 | Edge-sensitive, rotation-aware | Fast |
| `Block` | 16×16 | Local feature preservation | Slower (more pixels) |
| `VertGradient` | 9×8 | Vertical edge detection | Fast |
| `DoubleGradient` | 9×9 | Bidirectional edge detection | Fast |

### Pattern 3: GPU Zero-Copy Pipeline (Large Images)

```rust
// GPU resize path (for large images, 512×512+)
let hasher = PerceptualHasher::with_resize_mode(
    &mut ctx, HashAlgorithm::Mean, true  // true = GPU resize
)?;

// Same API, internally: GPU resize → GPU hash (no CPU intermediate)
let hashes = hasher.compute(&ctx, &images, &dimensions)?;
```

**When to use GPU resize**:
- Image size ≥ 512×512 AND batch ≥ 100 → GPU resize faster
- Image size < 256×256 → CPU resize faster (transfer overhead dominates)
- Mixed sizes → falls back to CPU resize per-image

### Pattern 4: BK-tree Similarity Search

```rust
use wgpu_compute_engine::tasks::bktree::{BkTree, hamming_distance};

// Step 1: Build tree from hash values
let tree = BkTree::from_hashes(hashes.iter().copied());

// Step 2: Find similar images (Hamming distance ≤ threshold)
let similar: Vec<(u64, u32)> = tree.find(query_hash, 5);

// Step 3: Find nearest neighbor
let nearest: Option<(u64, u32)> = tree.find_nearest(query_hash);
```

**Performance** (100K hashes):
- `find_nearest`: ~50ns (vs brute-force ~312µs → **6240x faster**)
- `find(threshold=5)`: ~150ns (vs brute-force ~312µs → **2080x faster**)
- Build: ~97ms for 100K entries

### Pattern 5: Custom GPU Compute Pipeline

```rust
use wgpu_compute_engine::{GpuContext, GpuBuffer, BufferUsage, ComputePipeline};

let mut ctx = GpuContext::new_sync()?;

// Step 1: Get or create pipeline (cached by WGSL hash + workgroup size)
let pipeline = ctx.get_or_create_pipeline(
    include_str!("my_shader.wgsl"),
    [256, 1, 1],
)?;

// Step 2: Create buffers
let input = GpuBuffer::from_data(ctx.device(), &my_data, BufferUsage::Storage);
let output = GpuBuffer::empty(ctx.device(), output_size, BufferUsage::Storage);
let params = GpuBuffer::from_data(ctx.device(), &[my_params], BufferUsage::Uniform);

// Step 3: Dispatch
let dispatch_x = (item_count as u32).div_ceil(256);
pipeline.dispatch(
    ctx.device(), ctx.queue(),
    &input, &output, &params,
    [dispatch_x, 1, 1],
);

// Step 4: Read results
let result = output.download(ctx.device(), ctx.queue())?;
```

**WGSL shader contract** (must follow this layout):
```wgsl
@group(0) @binding(0) var<storage, read> input: array<T>;
@group(0) @binding(1) var<storage, read_write> output: array<T>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(N)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
```

### Pattern 6: Async Batch Submission

```rust
use wgpu_compute_engine::{GpuContext, GpuBuffer, BufferUsage, BatchJob};
use std::sync::Arc;

let mut ctx = GpuContext::new_sync()?;
let pipeline = ctx.get_or_create_pipeline(shader, [256, 1, 1])?;

let mut submitter = wgpu_compute_engine::GpuBatchSubmitter::new();

for batch_data in batches {
    let input = GpuBuffer::from_data(ctx.device(), &batch_data, BufferUsage::Storage);
    let output = GpuBuffer::empty(ctx.device(), output_size, BufferUsage::Storage);
    let params = GpuBuffer::from_data(ctx.device(), &[params_data], BufferUsage::Uniform);

    submitter.submit(BatchJob {
        input,
        output,
        params,
        pipeline: Arc::clone(&pipeline),
        dispatch: [dispatch_x, 1, 1],
    });
}

// One GPU submit for all jobs → 4-6x faster than individual dispatches
let results: Vec<Vec<u8>> = submitter.wait_all()?;
```

## Critical Rules for Agents

### MUST DO

1. **Create `GpuContext` ONCE and reuse it** — context creation is expensive (adapter selection, device initialization)
2. **Batch inputs** — GPU dispatch has ~1.6ms fixed overhead; single-item batches waste GPU resources
3. **Use `BufferPool` for frequent allocations** — reduces GPU buffer creation/destruction overhead
4. **Handle `GpuError::NoAdapter`** — GPU may not be available; always provide CPU fallback
5. **Use `with_resize_mode(true)` for large images** — enables zero-copy GPU pipeline
6. **Match image dimensions for batch** — `PerceptualHasher::compute()` requires all images in a batch to have the same dimensions

### MUST NOT DO

1. **Do NOT use `tasks::hash_common` directly** — it's `#[doc(hidden)]` internal infrastructure; use `PerceptualHasher`
2. **Do NOT use `tasks::gpu_resize` directly** — use `PerceptualHasher::with_resize_mode()`
3. **Do NOT use `GpuBuffer::from_raw/into_raw/raw`** — these are internal API for pipeline integration
4. **Do NOT use `ComputePipeline::encode_dispatch_into`** — internal API; use `dispatch()` or `GpuBatchSubmitter`
5. **Do NOT create `GpuContext` per operation** — reuse across all GPU operations
6. **Do NOT inline WGSL as strings** — use `include_str!("shader.wgsl")` with separate `.wgsl` files
7. **Do NOT add `main.rs`** — this is a library crate only; use `cargo run --example` for demos

### Performance Decision Matrix

| Scenario | Recommended Path | Why |
|----------|-----------------|-----|
| SHA-256, batch < 100 | CPU (e.g., `sha2` crate) | GPU dispatch overhead dominates |
| SHA-256, batch ≥ 1000 | GPU `Sha256Computer` | 7-34x faster per message |
| Image hash, small images (< 256px) | CPU resize path | Transfer overhead < compute savings |
| Image hash, large images (≥ 512px) | GPU resize (`with_resize_mode(true)`) | Zero-copy pipeline avoids CPU intermediate |
| Image hash, batch ≥ 1000 | GPU resize + batch | Best GPU utilization |
| Similarity search, < 1K hashes | Brute force `hamming_distance` loop | BK-tree overhead not worth it |
| Similarity search, ≥ 10K hashes | `BkTree` | O(log N) vs O(N), 2000-6000x faster |
| Multiple GPU dispatches | `GpuBatchSubmitter` | 4-6x faster than individual dispatches |

## Feature Flags

| Flag | Effect | Required for |
|------|--------|-------------|
| `image` | Enables `image` crate integration | Loading images from files in tests/examples |
| (default) | No extra deps | All core functionality works |

## File Locations

| What | Where | Notes |
|------|-------|-------|
| GPU context | `src/context.rs` | `GpuContext`: sync/async creation |
| Buffer management | `src/buffer.rs`, `src/buffer_pool.rs` | Pool tiers by size |
| Pipeline caching | `src/context.rs` → `PipelineCache` | fxhash-based, auto-compile |
| Batch submit | `src/batch.rs` | Async submit, wait_all pattern |
| SHA-256 GPU | `src/tasks/sha256.rs` + `.wgsl` | Reference implementation (654 lines) |
| Image hashing | `src/tasks/phasher.rs` | Orchestrator + GPU resize integration |
| Hash common | `src/tasks/hash_common.rs` | Shared trait + macros + zero-copy entry |
| GPU resize | `src/tasks/gpu_resize.rs` + `resize.wgsl` | Box filter, zero-copy output |
| BK-tree | `src/tasks/bktree.rs` | Pure algorithm, no GPU dependency |
| Integration tests | `tests/*_test.rs` | One per algorithm |
| Benchmarks | `benches/*_bench.rs` | Criterion, no harness |

## Error Handling Pattern

```rust
match GpuContext::new_sync() {
    Ok(mut ctx) => {
        // GPU available, use GPU path
        let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean)?;
        let hashes = hasher.compute(&ctx, &images, &dimensions)?;
    }
    Err(GpuError::NoAdapter) => {
        // No GPU available, fall back to CPU implementation
        eprintln!("GPU not available, using CPU fallback");
    }
    Err(e) => return Err(e.into()),
}
```

## Adding New Algorithms

### Mode A: Full Algorithm (like SHA-256)

1. Create `src/tasks/new_algo.rs` + `src/tasks/new_algo.wgsl`
2. Implement struct holding `Arc<ComputePipeline>`
3. Implement `compute()`: pack → buffer → dispatch → download
4. Optional: implement `batch_submit()` for async batch
5. Export in `src/tasks/mod.rs`

### Mode B: Perceptual Hash Algorithm

1. Create `src/tasks/new_hash.rs` + `src/tasks/new_hash.wgsl`
2. Use macros: `declare_phash_computer!` + `impl_phash_computer_simple!`
3. Add variant to `HashAlgorithm` enum in `phasher.rs`
4. Add match arm in `PerceptualHasher::new()`
5. Export in `src/tasks/mod.rs` (with `#[doc(hidden)]`)

### WGSL Shader Requirements

- Must use binding layout: `@binding(0)` input, `@binding(1)` output, `@binding(2)` uniform params
- Entry point must be `fn main(@builtin(global_invocation_id) gid: vec3<u32>)`
- Must be a separate `.wgsl` file (NOT inline string)
- Use `include_str!("shader.wgsl")` to load

## Build & Test Commands

```bash
cargo build                              # Build (no image feature)
cargo build --features image             # Build with image support
cargo test --features image              # Run all tests
cargo test --test gpu_resize_test        # Run specific test
cargo bench --bench large_scale_bench    # Run benchmark
cargo clippy --features image            # Lint check
cargo doc --features image --no-deps     # Generate docs
cargo run --example demo                 # Run demo
```