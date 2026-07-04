---
name: "gpgpu_tool"
version: "0.4.0"
updated: "2026-07-04"
description: "GPU parallel compute engine using wgpu. Invoke when user needs GPU-accelerated SHA-256 hashing, perceptual image hashing, BK-tree similarity search, GPU Hamming distance matching, variable-length hash matching, czkawka GPU integration, or wants to build custom GPU compute pipelines on top of wgpu."
---

> **Version**: 0.4.0 | **Rust Edition**: 2021 | **wgpu**: v24 | **Updated**: 2026-07-04

# gpgpu_tool — Agent Usage Guide

Cross-platform GPU compute engine built on wgpu. Provides out-of-the-box GPU acceleration for CPU-intensive tasks without requiring WGSL shader authoring. Supports 64-bit to 4096-bit hash matching with GPU-accelerated Hamming distance computation.

## When to Invoke This Skill

- User asks about GPU-accelerated hashing (SHA-256, perceptual image hashing)
- User needs image similarity detection or approximate nearest neighbor search
- User wants to build custom GPU compute pipelines using wgpu
- User mentions BK-tree, Hamming distance, phash, or GPU compute
- User asks about batch GPU processing or zero-copy data pipelines
- User needs to process large-scale image datasets (10K+ images) with GPU
- User needs GPU-accelerated hash matching (64-bit to 4096-bit)
- User mentions variable-length hashes, HashBytes, or GPU Hamming distance
- User needs end-to-end GPU image matching (image → hash → match)
- User wants to integrate GPU acceleration into czkawka or similar image dedup tools
- User mentions czkawka, CzkawkaGpuAccelerator, or GPU-accelerated similar image detection

## Architecture Overview

```
┌─── Business Layer (Tasks) ──────────────────────────────────────────────────────┐
│  Sha256Computer    PerceptualHasher      BkTree / BkTreeBytes                   │
│  (SHA-256 hash)    (6 phash algorithms)  (ANN search, 64-bit + variable-length) │
│                   + GpuResize (zero-copy)                                        │
│                                                                                   │
│  GpuHashMatcher / GpuHashMatcherBytes   GpuImageMatcher                          │
│  (GPU Hamming distance, 3 pipelines)    (end-to-end: image → hash → match)       │
│                                                                                   │
│  HashMatcherFacade / HashMatcherFacadeBytes                                       │
│  (CPU matching: strategy + chain + facade, 64-bit + variable-length)              │
│                                                                                   │
│  DihedralHashes64/256/1024/4096                                                   │
│  (D4 group 8 rotations/flips, rotation-invariant matching)                        │
│                                                                                   │
│  CzkawkaGpuAccelerator (feature: czkawka-compat)                                  │
│  (One-stop: shared context + RGBA→hash + similar pairs + GPU threshold filter)    │
└────────────┬──────────────┬──────────────┬──────────────┬────────────────────────┘
             │              │              │              │
┌────────────▼──────────────▼──────────────▼──────────────▼────────────────────────┐
│              Capability Layer (Core)                                               │
│  GpuContext  GpuBuffer  BufferPool  ComputePipeline  GpuPipelineBuilder          │
│  GpuBatchSubmitter  GpuError  BackendDispatcher  DoubleBufferStaging             │
└──────────────────────────────────────────────────────────────────────────────────┘
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
| `BufferPoolConfig` | root | Buffer pool configuration parameters |
| `ComputePipeline` | root | Compute pipeline: shader compile + dispatch |
| `BindingType` | root | Binding type for pipeline descriptor |
| `PipelineDescriptor` | root | Pipeline descriptor (bindings, WGSL source, workgroup size) |
| `GpuBatchSubmitter` | root | Async batch submitter: merge multiple dispatches into one submit |
| `BatchJob` | root | Batch job descriptor |
| `GpuError` | root | Unified error type (9 variants, thiserror) |
| `ComputeBackend` | root | GPU/CPU backend selector enum |
| `BackendDispatcher` | root | GPU/CPU backend dispatch trait |
| `DefaultBackendDispatcher` | root | Default backend dispatcher implementation |
| `GpuPipelineBuilder` | root | Declarative pipeline builder (chain GPU processing steps) |
| `GpuPipelineStep` | root | Pipeline step enum |
| `DoubleBufferStaging` | root | Double-buffer staging manager (ping-pong overlap) |
| `download_batch` | root | Batch download multiple GPU buffers to CPU |
| `download_batch_with_pool` | root | Batch download with BufferPool reuse |

### Business Layer — Hashing (via `tasks::` submodule)

| Type | Module | Purpose |
|------|--------|---------|
| `Sha256Computer` | `tasks::sha256` | GPU parallel SHA-256 hashing |
| `PerceptualHasher` | `tasks::phasher` | Perceptual image hashing (6 algorithms) |
| `HashAlgorithm` | `tasks::phasher` | Algorithm selector enum |
| `HashSize` | `tasks::hash_common` | Hash grid size configuration (8/16/32/64) |
| `GpuConvolution` | `tasks::convolution` | GPU 2D convolution (Full2D / Separable) |
| `BorderMode` | `tasks::convolution` | Convolution border mode (Zero/Clamp/Reflect) |
| `ConvMode` | `tasks::convolution` | Convolution mode (Full2D/Separable) |
| `GpuGaussianBlur` | `tasks::gaussian_blur` | GPU Gaussian blur (based on separable convolution) |

### Business Layer — Matching (64-bit)

| Type | Module | Purpose |
|------|--------|---------|
| `BkTree` | `tasks::bktree` | BK-tree approximate nearest neighbor search (64-bit) |
| `hamming_distance` | `tasks::bktree` | Hamming distance between two u64 hashes |
| `HashMatcher` | `tasks::matcher` | 64-bit hash matching strategy trait |
| `MatchResult` | `tasks::matcher` | 64-bit match result (hash + distance) |
| `LinearScanMatcher` | `tasks::matcher` | Exact linear scan matcher, 100% recall |
| `BkTreeMatcher` | `tasks::matcher` | BK-tree adapter matcher |
| `ChainedMatcher` | `tasks::matcher` | Chain of responsibility matcher |
| `ChainStrategy` | `tasks::matcher` | Chain strategy (Union/FirstHit) |
| `HashMatcherFacade` | `tasks::matcher` | Unified facade (strategy + dihedral enhancement) |

### Business Layer — Matching (Variable-Length HashBytes)

| Type | Module | Purpose |
|------|--------|---------|
| `HashBytes` | `tasks::hash_bytes` | Variable-length hash type (Vec\<u8\> wrapper, 64-bit to 4096-bit) |
| `BkTreeBytes` | `tasks::bktree_bytes` | BK-tree for variable-length hashes |
| `HashMatcherBytes` | `tasks::matcher_bytes` | Variable-length hash matching strategy trait |
| `MatchResultBytes` | `tasks::matcher_bytes` | Variable-length match result |
| `LinearScanMatcherBytes` | `tasks::matcher_bytes` | Variable-length linear scan matcher |
| `BkTreeMatcherBytes` | `tasks::matcher_bytes` | Variable-length BK-tree matcher |
| `ChainedMatcherBytes` | `tasks::matcher_bytes` | Variable-length chain matcher |
| `ChainStrategyBytes` | `tasks::matcher_bytes` | Variable-length chain strategy |
| `HashMatcherFacadeBytes` | `tasks::matcher_bytes` | Variable-length unified facade |

### Business Layer — GPU Matching

| Type | Module | Purpose |
|------|--------|---------|
| `GpuHashMatcher` | `tasks::gpu_matcher` | GPU Hamming distance matcher (64-bit, 3 pipelines) |
| `GpuHashMatcherFacade` | `tasks::gpu_matcher` | 64-bit GPU matcher facade (adapts to HashMatcher trait) |
| `GpuHashMatcherBytes` | `tasks::gpu_matcher` | Variable-length GPU Hamming distance matcher |
| `GpuHashMatcherFacadeBytes` | `tasks::gpu_matcher` | Variable-length GPU matcher facade |
| `GpuImageMatcher` | `tasks::gpu_image_matcher` | End-to-end GPU image matcher (image → hash → match) |

### Business Layer — Dihedral Transforms

| Type | Module | Purpose |
|------|--------|---------|
| `DihedralHashes64` | `tasks::dihedral` | 64-bit D4 group 8 transforms |
| `DihedralHashes256` | `tasks::dihedral` | 256-bit D4 group 8 transforms |
| `DihedralHashes1024` | `tasks::dihedral` | 1024-bit D4 group 8 transforms (32×32 bit matrix) |
| `DihedralHashes4096` | `tasks::dihedral` | 4096-bit D4 group 8 transforms (64×64 bit matrix) |
| `DihedralTransform` | `tasks::dihedral` | Unified D4 transform trait |

### Business Layer — CPU Fallback (feature: `cpu-fallback`)

| Type | Module | Purpose |
|------|--------|---------|
| `Sha256Cpu` | `tasks::sha256_cpu` | CPU SHA-256 fallback |
| `PHasherCpu` | `tasks::phasher_cpu` | CPU perceptual hash fallback |

### Business Layer — PDQ Hash (feature: `pdq`)

| Type | Module | Purpose |
|------|--------|---------|
| `PdqHashGpu` | `tasks::pdq_hash` | PDQ GPU hasher (DCT-based 256-bit) |
| `PdqHashCpu` | `tasks::pdq_hash` | PDQ CPU hasher |
| `PdqHashResult` | `tasks::pdq_hash` | PDQ hash result (256-bit hash + quality) |

### Integration Layer — czkawka Compat (feature: `czkawka-compat`)

| Type | Module | Purpose |
|------|--------|---------|
| `CzkawkaGpuAccelerator` | `czkawka_compat` | One-stop GPU accelerator: shared context + hash + match |
| `GpuHashMatcherBytes::compute_similar_pairs` | `tasks::gpu_matcher` | Symmetric similar pairs (czkawka `gpu_compare_hashes_auto` format) |
| `GpuHashMatcherBytes::compute_similar_pairs_asymmetric` | `tasks::gpu_matcher` | Asymmetric pairs (czkawka `gpu_compare_hashes_asymmetric` format) |
| `GpuHashMatcherBytes::compute_similar_pairs_gpu_filtered` | `tasks::gpu_matcher` | GPU-side threshold filtering (avoids full matrix download) |
| `PerceptualHasher::compute_from_rgba` | `tasks::phasher` | RGBA→grayscale→hash (czkawka formula `(R*77+G*150+B*29)>>8`) |
| `HashAlgorithm::from_czkawka` | `tasks::phasher` | Convert czkawka algorithm name to HashAlgorithm |
| `HashSize::from_czkawka` | `tasks::hash_common` | Convert czkawka hash_size (8/16/32/64) to HashSize |

### Internal Modules (#[doc(hidden)], avoid direct use)

| Module | Why hidden | Use instead |
|--------|-----------|-------------|
| `tasks::hash_common` | Internal trait + macros + HashSize | `PerceptualHasher` or individual `*HashComputer` |
| `tasks::gpu_resize` | Internal GPU resize impl | `PerceptualHasher::with_resize_mode(true)` |
| `tasks::mean_hash` etc. | Thin wrappers via macros | `PerceptualHasher::new(algo)` |
| `tasks::convolution` | Internal GPU convolution impl | `GpuConvolution` or `GpuGaussianBlur` |
| `tasks::gaussian_blur` | Internal GPU blur impl | `GpuGaussianBlur` |

## Usage Patterns

### Pattern 1: SHA-256 Parallel Hashing

```rust
use gpgpu_tool::{GpuContext, tasks::sha256::Sha256Computer};

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
use gpgpu_tool::{GpuContext, tasks::phasher::{PerceptualHasher, HashAlgorithm}};
use gpgpu_tool::tasks::hash_common::HashSize;

let mut ctx = GpuContext::new_sync()?;

// Default hash_size=8 (64-bit hash)
let hasher = PerceptualHasher::new(&mut ctx, HashAlgorithm::Mean)?;

// Custom hash_size=16 (256-bit hash)
let hasher = PerceptualHasher::with_hash_size(
    &mut ctx, HashAlgorithm::Mean, HashSize::new(16)
)?;

// Full config: hash_size + GPU resize + workgroup_size
let hasher = PerceptualHasher::with_full_config(
    &mut ctx, HashAlgorithm::Mean, true, HashSize::new(16), [256, 1, 1]
)?;

// Input: grayscale pixel arrays + dimensions
let images: Vec<Vec<u8>> = vec![vec![128u8; 256 * 256]];
let dimensions: Vec<(u32, u32)> = vec![(256, 256)];
let hashes: Vec<u64> = hasher.compute(&ctx, &images, &dimensions)?;
```

**Algorithm selection guide**:

| Algorithm | Default Size | hash_size=16 Size | Best for | Speed |
|-----------|-------------|-------------------|----------|-------|
| `Mean` | 8×8 | 16×16 | General similarity, fast baseline | Fastest |
| `Median` | 8×8 | 16×16 | Robust to outliers (bright spots) | Fast |
| `Gradient` | 8×9 | 16×17 | Edge-sensitive, rotation-aware | Fast |
| `Block` | 8×8 | 16×16 | Local feature preservation | Slower |
| `VertGradient` | 9×8 | 17×16 | Vertical edge detection | Fast |
| `DoubleGradient` | 9×9 | 17×17 | Bidirectional edge detection | Fast |

**Mean/Median Hash optimization**: CPU pre-computes threshold (mean/median), passes via `f32::to_bits()` + shader `bitcast<f32>()`, eliminating O(N²) loop in GPU shader.

### Pattern 3: GPU Zero-Copy Pipeline (Large Images)

```rust
// GPU resize path with custom hash_size
let hasher = PerceptualHasher::with_full_config(
    &mut ctx, HashAlgorithm::Mean, true, HashSize::new(16), [256, 1, 1]
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
use gpgpu_tool::tasks::bktree::{BkTree, hamming_distance};

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
use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage, ComputePipeline};

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
use gpgpu_tool::{GpuContext, GpuBuffer, BufferUsage, BatchJob};
use std::sync::Arc;

let mut ctx = GpuContext::new_sync()?;
let pipeline = ctx.get_or_create_pipeline(shader, [256, 1, 1])?;

let mut submitter = gpggpu_tool::GpuBatchSubmitter::new();

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

### Pattern 7: GPU Hash Matching

GPU-accelerated Hamming distance matching with 3-pipeline architecture:
- **Distance matrix** (`compute_hamming_matrix`): N×M Hamming distance matrix
- **Nearest neighbor** (`find_nearest_neighbor`): Standard pipeline, ≤1024-bit, workgroup_size=256
- **Large hash nearest neighbor** (`find_nearest_neighbor_large`): ≤4096-bit, workgroup_size=32

**Pipeline selection**: `u32_per_hash > 32` → large-hash pipeline, else standard pipeline.

#### 64-bit GPU Matching

```rust
use gpgpu_tool::{GpuContext, tasks::gpu_matcher::GpuHashMatcherFacade};

let mut ctx = GpuContext::new_sync()?;

// Create GPU matcher facade (adapts to HashMatcher trait)
let matcher = GpuHashMatcherFacade::new(&mut ctx)?;

// Database of hashes
let db_hashes: Vec<u64> = vec![0xA1B2C3D4, 0x12345678, 0x87654321];

// Find all matches within threshold
let results: Vec<MatchResult> = matcher.find_similar(&db_hashes, query_hash, 5)?;

// Distance matrix (N queries × M database)
let gpu_matcher = GpuHashMatcher::new(&mut ctx)?;
let distances: Vec<u32> = gpu_matcher.compute_distance_matrix(
    &ctx, &query_hashes, &db_hashes
)?;
```

#### Variable-Length GPU Matching

```rust
use gpgpu_tool::{GpuContext, HashBytes, tasks::gpu_matcher::GpuHashMatcherFacadeBytes};

let mut ctx = GpuContext::new_sync()?;

// Create variable-length GPU matcher
let matcher = GpuHashMatcherFacadeBytes::new(&mut ctx, 32)?; // 32 u32 per hash = 1024-bit

// Database of HashBytes (e.g., 1024-bit hashes from Block hash with hash_size=32)
let db_hashes: Vec<HashBytes> = vec![...];

// Find matches within threshold
let results: Vec<MatchResultBytes> = matcher.find_similar(&db_hashes, &query_hash, 20)?;
```

**Performance** (Czkawka cache data, 20000 × 1024-bit hashes, threshold=20):

| Method | Scale | Time | Matches |
|--------|-------|------|---------|
| CPU linear scan (200 × 20000) | 4M comparisons | 17.1s | 200 |
| CPU estimated full (20000 × 20000) | 400M comparisons | ~1711s | — |
| **GPU nearest neighbor (20000 × 20000)** | **400M comparisons** | **2.2s** | 20000 |
| GPU distance matrix (500 × 500) | 250K comparisons | 14.6ms | 532 |

**GPU acceleration: 778x** vs CPU for full-scale nearest-neighbor search.

### Pattern 8: Variable-Length Hash Matching

For hashes beyond 64-bit (256/1024/4096-bit), use `HashBytes` and `*Bytes` suffixed APIs.

#### Building and Querying with HashBytes

```rust
use gpgpu_tool::{HashBytes, BkTreeBytes, tasks::matcher_bytes::HashMatcherFacadeBytes};

// HashBytes wraps Vec<u8> — supports any bit width
let hash_256: HashBytes = HashBytes::from(vec![0u8; 32]);  // 256-bit
let hash_1024: HashBytes = HashBytes::from(vec![0u8; 128]); // 1024-bit
let hash_4096: HashBytes = HashBytes::from(vec![0u8; 512]); // 4096-bit

// BK-tree for variable-length hashes
let tree = BkTreeBytes::from_hashes(db_hashes);
let similar: Vec<(HashBytes, u32)> = tree.find(&query_hash, 10);

// Facade with strategy selection
let facade = HashMatcherFacadeBytes::linear_scan(db_hashes);
let results: Vec<MatchResultBytes> = facade.find_similar(&query_hash, 10)?;
```

#### Dihedral Transform Enhancement

```rust
use gpgpu_tool::{DihedralHashes1024, DihedralTransform};

// Compute 8 D4 group variants for rotation-invariant matching
let dihedral = DihedralHashes1024::from_hash_bytes(&hash_1024);
let all_variants: [&HashBytes; 8] = [
    &dihedral.identity,
    &dihedral.rotate_90,
    &dihedral.rotate_180,
    &dihedral.rotate_270,
    &dihedral.flip_h,
    &dihedral.flip_v,
    &dihedral.flip_diag,
    &dihedral.flip_anti_diag,
];

// Match against all variants for rotation-invariant search
for variant in &all_variants {
    let results = facade.find_similar(variant, threshold)?;
    if !results.is_empty() {
        break; // First hit strategy
    }
}
```

**HashBytes → u32 alignment**: `hash_bytes_to_u32()` automatically pads HashBytes to u32-aligned arrays for GPU upload. Empty or zero-length hashes return empty result (defensive check).

### Pattern 9: End-to-End GPU Image Matching

```rust
use gpgpu_tool::{GpuContext, tasks::gpu_image_matcher::GpuImageMatcher};
use gpgpu_tool::tasks::phasher::HashAlgorithm;

let mut ctx = GpuContext::new_sync()?;

// Create end-to-end matcher (GPU hash + GPU match)
let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Gradient)?;

// Query images and database images
let query_images = vec![vec![128u8; 64 * 64]];
let query_dims = vec![(64u32, 64u32)];
let db_images = vec![vec![200u8; 64 * 64]];
let db_dims = vec![(64u32, 64u32)];

// One call: GPU resize → GPU hash → GPU Hamming distance matching
let results = matcher.find_similar(
    &ctx, &query_images, &query_dims,
    &db_images, &db_dims, 10  // threshold=10
)?;
```

### Pattern 10: czkawka GPU Integration (feature: `czkawka-compat`)

The `CzkawkaGpuAccelerator` provides a one-stop API for integrating GPU acceleration into czkawka's similar image detection pipeline. It shares a single `GpuContext` across both hash computation and distance matching stages, avoiding redundant GPU initialization.

#### Quick Start: Two-Stage Pipeline

```rust
use gpgpu_tool::czkawka_compat::CzkawkaGpuAccelerator;
use gpgpu_tool::tasks::phasher::HashAlgorithm;

// Create accelerator (hash_size=8 → 64-bit, Gradient algorithm)
// Returns GpuError::GpuUnavailable if GPU not available — caller should fallback to CPU
let accelerator = CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient)?;

// Stage 1: Batch compute perceptual hashes from RGBA image data
// Uses czkawka-compatible grayscale formula: (R*77 + G*150 + B*29) >> 8
let rgba_images: Vec<Vec<u8>> = vec![vec![128u8; 64 * 64 * 4]]; // RGBA8888
let dims: Vec<(u32, u32)> = vec![(64, 64)];
let hashes = accelerator.compute_hashes(&rgba_images, &dims)?;

// Stage 2: Find similar pairs (symmetric mode, tolerance=10)
// Returns Vec<(parent_idx, child_idx, distance)> sorted by distance
let pairs = accelerator.find_similar_pairs(&hashes, 10)?;
for (parent, child, distance) in &pairs {
    println!("Similar: {} <-> {} (distance={})", parent, child, distance);
}
```

#### Asymmetric Mode (Reference vs Normal folders)

```rust
// Find similar pairs between two different hash sets
let pairs = accelerator.find_similar_pairs_asymmetric(
    &ref_hashes,   // Reference folder hashes
    &normal_hashes, // Normal folder hashes
    10,             // tolerance
)?;
// Returns Vec<(ref_idx, normal_idx, distance)>
```

#### GPU-Side Threshold Filtering (Large Datasets)

For large hash collections (N ≥ 2000), the accelerator automatically switches to GPU-side threshold filtering using `hamming_pairs.wgsl`, which:
- Computes all N×N Hamming distances on GPU
- Filters by threshold directly on GPU using atomic counters
- Downloads only matching pairs (typically far fewer than N²)
- Avoids downloading the full distance matrix (which would be 1.6GB for N=20000)

This is transparent — no API change needed. The accelerator auto-selects the optimal path:
- **N < 2000**: Distance matrix + CPU filter (lower overhead for small N)
- **N ≥ 2000**: GPU-side `hamming_pairs.wgsl` filter (avoids massive matrix download)

#### Error Handling & CPU Fallback

```rust
match CzkawkaGpuAccelerator::new(8, HashAlgorithm::Gradient) {
    Ok(accelerator) => {
        // GPU path — use accelerator.compute_hashes() + find_similar_pairs()
        let hashes = accelerator.compute_hashes(&images, &dims)?;
        let pairs = accelerator.find_similar_pairs(&hashes, tolerance)?;
    }
    Err(GpuError::GpuUnavailable) => {
        // GPU not available — fallback to czkawka's native CPU path
        // No action needed; czkawka continues with its own hasher
    }
    Err(e) => return Err(e.into()),
}
```

#### Algorithm & Hash Size Mapping

| czkawka `HashAlg` | `HashAlgorithm` | czkawka `hash_size` | `HashSize` | Bit width |
|---------------------|-----------------|---------------------|------------|-----------|
| `Mean` | `Mean` | 8 | `HashSize::new(8)` | 64-bit |
| `Median` | `Median` | 16 | `HashSize::new(16)` | 256-bit |
| `Gradient` | `Gradient` | 32 | `HashSize::new(32)` | 1024-bit |
| `Blockhash` | `Block` | 64 | `HashSize::new(64)` | 4096-bit |
| `VertGradient` | `VertGradient` | | | |
| `DoubleGradient` | `DoubleGradient` | | | |

Use `HashAlgorithm::from_czkawka("Blockhash")` and `HashSize::from_czkawka(8)` for automatic conversion.

#### Direct API Access (Without Accelerator)

For advanced use cases, access the underlying APIs directly:

```rust
use gpgpu_tool::{GpuContext, HashBytes, tasks::gpu_matcher::GpuHashMatcherBytes};
use gpgpu_tool::tasks::phasher::{PerceptualHasher, HashAlgorithm};

let mut ctx = GpuContext::new_sync()?;
let hasher = PerceptualHasher::with_hash_size(&mut ctx, HashAlgorithm::Gradient, HashSize::new(8))?;
let matcher = GpuHashMatcherBytes::new(&mut ctx)?;

// RGBA → hash
let hashes = hasher.compute_from_rgba_to_hash_bytes(&ctx, &rgba_images, &dims)?;

// GPU-side filtered pairs (explicit call)
let pairs = matcher.compute_similar_pairs_gpu_filtered(&ctx, &hashes, 10)?;

// Or asymmetric mode
let pairs = matcher.compute_similar_pairs_asymmetric_gpu_filtered(
    &ctx, &ref_hashes, &normal_hashes, 10
)?;
```

#### Integration into czkawka (Upstream PR Guide)

1. Add `gpgpu-tool` as optional dependency in `czkawka_core/Cargo.toml`:
   ```toml
   [dependencies]
   gpgpu-tool = { version = "0.4", features = ["czkawka-compat"], optional = true }
   
   [features]
   gpu-accel = ["dep:gpgpu-tool"]
   ```

2. Replace hash computation in `collect_image_file_entry`:
   ```rust
   #[cfg(feature = "gpu-accel")]
   let hashes = accelerator.compute_hashes(&rgba_images, &dims)?;
   ```

3. Replace distance comparison in `compare_hashes_with_non_zero_tolerance`:
   ```rust
   #[cfg(feature = "gpu-accel")]
   let pairs = accelerator.find_similar_pairs(&hashes, tolerance)?;
   ```

4. Handle GPU unavailable: catch `GpuError::GpuUnavailable` and fallback to czkawka CPU path.

## Critical Rules for Agents

### MUST DO

1. **Create `GpuContext` ONCE and reuse it** — context creation is expensive (adapter selection, device initialization)
2. **Batch inputs** — GPU dispatch has ~1.6ms fixed overhead; single-item batches waste GPU resources
3. **Use `BufferPool` for frequent allocations** — reduces GPU buffer creation/destruction overhead
4. **Handle `GpuError::NoAdapter`** — GPU may not be available; always provide CPU fallback
5. **Use `with_resize_mode(true)` for large images** — enables zero-copy GPU pipeline
6. **Match image dimensions for batch** — `PerceptualHasher::compute()` requires all images in a batch to have the same dimensions
7. **Use `HashSize::new(N)` for larger hashes** — default is 8 (64-bit); use 16/32/64 for higher precision
8. **Choose correct GPU matcher pipeline** — `u32_per_hash > 32` → large-hash pipeline (workgroup_size=32), else standard (workgroup_size=256)
9. **Use `HashBytes` for > 64-bit hashes** — u64 matchers only support 64-bit; use `*Bytes` variants for 256/1024/4096-bit
10. **Use `GpuHashMatcher` for large-scale matching** — GPU 778x faster than CPU for 20K×20K nearest-neighbor search

### MUST NOT DO

1. **Do NOT use `tasks::hash_common` directly** — it's `#[doc(hidden)]` internal infrastructure; use `PerceptualHasher`
2. **Do NOT use `tasks::gpu_resize` directly** — use `PerceptualHasher::with_resize_mode()`
3. **Do NOT use `GpuBuffer::from_raw/into_raw/raw`** — these are internal API for pipeline integration
4. **Do NOT use `ComputePipeline::encode_dispatch_into`** — internal API; use `dispatch()` or `GpuBatchSubmitter`
5. **Do NOT create `GpuContext` per operation** — reuse across all GPU operations
6. **Do NOT inline WGSL as strings** — use `include_str!("shader.wgsl")` with separate `.wgsl` files
7. **Do NOT add `main.rs`** — this is a library crate only; use `cargo run --example` for demos
8. **Do NOT mix 64-bit and HashBytes matchers** — use `HashMatcher`/`BkTree` for u64, `HashMatcherBytes`/`BkTreeBytes` for `HashBytes`
9. **Do NOT pass empty HashBytes to GPU matcher** — `hash_bytes_to_u32` returns empty for zero-length hashes (defensive check)

### Performance Decision Matrix

| Scenario | Recommended Path | Why |
|----------|-----------------|-----|
| SHA-256, batch < 100 | CPU (e.g., `sha2` crate) | GPU dispatch overhead dominates |
| SHA-256, batch ≥ 1000 | GPU `Sha256Computer` | 7-34x faster per message |
| Image hash, small images (< 256px) | CPU resize path | Transfer overhead < compute savings |
| Image hash, large images (≥ 512px) | GPU resize (`with_resize_mode(true)`) | Zero-copy pipeline avoids CPU intermediate |
| Image hash, batch ≥ 1000 | GPU resize + batch | Best GPU utilization |
| Similarity search, < 1K hashes (64-bit) | Brute force `hamming_distance` loop | BK-tree overhead not worth it |
| Similarity search, ≥ 10K hashes (64-bit) | `BkTree` | O(log N) vs O(N), 2000-6000x faster |
| Hash matching, ≥ 1K hashes (any bit width) | `GpuHashMatcher` / `GpuHashMatcherBytes` | GPU 778x faster for large-scale nearest-neighbor |
| Hash matching, < 1K hashes (any bit width) | CPU `LinearScanMatcher` / `LinearScanMatcherBytes` | GPU dispatch overhead not worth it |
| End-to-end image matching | `GpuImageMatcher` | Single call: GPU hash + GPU match |
| Multiple GPU dispatches | `GpuBatchSubmitter` | 4-6x faster than individual dispatches |
| Rotation-invariant matching | `DihedralHashes*` + `HashMatcherFacade` | 8 D4 variants, CPU-only transforms |

## Feature Flags

| Flag | Default | Effect | Required for |
|------|---------|--------|-------------|
| `cpu-fallback` | ❌ | Enables CPU implementations (`Sha256Cpu`, `PHasherCpu`), depends on `sha2` crate | CPU fallback when GPU unavailable |
| `image` | ❌ | Enables `image` crate integration | Loading images from files in tests/examples |
| `pdq` | ❌ | Enables PDQ perceptual hash (DCT-based 256-bit) | PDQ hash algorithm |
| `gpu-accel` | ❌ | Enables image crate integration (alias for `image`) | GPU-accelerated image processing |
| `czkawka-compat` | ❌ | Enables `CzkawkaGpuAccelerator` + RGBA→hash + similar pairs API | czkawka GPU integration |
| `dual-encoder` | ❌ | Enables `DualEncoderSubmitter` for large batches (>4096) | Hiding latency with dual-encoder交替提交 |

> **⚠️ Breaking Change (v0.4.0)**: `default` feature changed from `["cpu-fallback"]` to `[]`. Enable `cpu-fallback` explicitly if CPU fallback is needed.

## File Locations

| What | Where | Notes |
|------|-------|-------|
| GPU context | `src/context.rs` | `GpuContext`: sync/async creation, pipeline cache |
| Buffer management | `src/buffer.rs`, `src/buffer_pool.rs` | Pool tiers by size, DoubleBufferStaging |
| Pipeline caching | `src/context.rs` → `PipelineCache` | fxhash-based, auto-compile |
| Pipeline builder | `src/pipeline_builder.rs` | Declarative chain GPU processing steps |
| Backend dispatch | `src/backend_dispatcher.rs` | GPU/CPU backend selection trait |
| Batch submit | `src/batch.rs` | Async submit, wait_all pattern |
| Pixel pack/unpack | `src/pixel_pack.rs` | u8↔u32 conversion for GPU shaders |
| SHA-256 GPU | `src/tasks/sha256.rs` + `.wgsl` | Reference implementation |
| SHA-256 CPU | `src/tasks/sha256_cpu.rs` | CPU fallback (feature: `cpu-fallback`) |
| Image hashing | `src/tasks/phasher.rs` | Orchestrator + GPU resize integration |
| Hash common | `src/tasks/hash_common.rs` | Shared trait + macros + zero-copy entry |
| GPU resize | `src/tasks/gpu_resize.rs` + `resize.wgsl` | Bilinear interpolation, zero-copy output |
| Convolution | `src/tasks/convolution.rs` + `convolution.wgsl` | 2D/Separable, Storage binding for kernel |
| Gaussian blur | `src/tasks/gaussian_blur.rs` | Based on GpuConvolution (separable) |
| Dihedral transforms | `src/tasks/dihedral.rs` | D4 group 8 transforms, 64/256/1024/4096-bit |
| 64-bit matching | `src/tasks/matcher.rs` | Strategy + Chain + Facade pattern |
| Variable-length matching | `src/tasks/matcher_bytes.rs` | HashBytes version of matcher strategies |
| BK-tree (64-bit) | `src/tasks/bktree.rs` | Pure algorithm, no GPU dependency |
| BK-tree (variable-length) | `src/tasks/bktree_bytes.rs` | HashBytes BK-tree |
| Variable-length hash type | `src/tasks/hash_bytes.rs` | `HashBytes`(Vec\<u8\>) wrapper |
| GPU Hamming distance | `src/tasks/hamming.wgsl` | 3 entry points: matrix + nearest + large |
| GPU pairs filter | `src/tasks/hamming_pairs.wgsl` | 2 entry points: symmetric + asymmetric pairs |
| GPU matcher | `src/tasks/gpu_matcher.rs` | GPU Hamming distance matcher (5 pipelines: matrix + nearest + large + pairs + pairs_asymmetric) |
| GPU image matcher | `src/tasks/gpu_image_matcher.rs` | End-to-end: image → hash → match |
| PDQ hash | `src/tasks/pdq_hash.rs` + `pdq_hash.wgsl` | DCT-based 256-bit (feature: `pdq`) |
| RGBA color conversion | `src/tasks/color_convert.wgsl` | RGBA→grayscale (czkawka formula) |
| czkawka compat module | `src/czkawka_compat.rs` | `CzkawkaGpuAccelerator` (feature: `czkawka-compat`) |
| czkawka compat tests | `tests/czkawka_compat_test.rs` | 22 tests covering all P0 integration items |
| Integration tests | `tests/*_test.rs` | One per algorithm |
| Cache data test | `tests/cache_gpu_matcher_test.rs` | Czkawka cache data GPU matching validation |
| Benchmarks | `benches/*_bench.rs` | Criterion, no harness |

## Error Handling Pattern

```rust
match GpuContext::new_sync() {
    Ok(mut ctx) => {
        let hasher = PerceptualHasher::with_hash_size(
            &mut ctx, HashAlgorithm::Mean, HashSize::new(16)
        )?;
        let hashes = hasher.compute(&ctx, &images, &dimensions)?;
    }
    Err(GpuError::NoAdapter) => {
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
2. Use macros: `declare_phash_computer!` + `impl_phash_computer_simple!` (generates `hash_size: HashSize` field)
3. Add variant to `HashAlgorithm` enum in `phasher.rs`
4. Add match arm in `PerceptualHasher::with_full_config_and_gpu_resize()`
5. Export in `src/tasks/mod.rs` (with `#[doc(hidden)]`)

### Mode C: Variable-Length Hash Matcher

1. Create `src/tasks/new_matcher_bytes.rs`
2. Implement `HashMatcherBytes` trait with consistent API
3. Use `*Bytes` suffix to distinguish from 64-bit version
4. Use `hash_bytes_to_u32()` for GPU u32 alignment
5. Export in `src/tasks/mod.rs` and `src/lib.rs`

### Mode D: Convolution-Based Filter

1. Create `src/tasks/new_filter.rs`
2. Wrap `GpuConvolution`, provide `_gpu` zero-copy method
3. Export in `src/tasks/mod.rs` (with `#[doc(hidden)]`)

### WGSL Shader Requirements

- Must use binding layout: `@binding(0)` input, `@binding(1)` output, `@binding(2)` uniform params
- Params struct (32 bytes, 16-byte aligned):
```wgsl
struct Params {
    image_count: u32,
    width: u32,
    height: u32,
    hash_size_bits: u32,
    hash_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};
@group(0) @binding(2) var<uniform> params: Params;
```
- Entry point must be `fn main(@builtin(global_invocation_id) gid: vec3<u32>)`
- Must be a separate `.wgsl` file (NOT inline string)
- Use `include_str!("shader.wgsl")` to load
- **Kernel arrays**: Use `var<storage, read>` binding (NOT uniform) to avoid 16-byte alignment issues

## Build & Test Commands

```bash
# Build (default: cpu-fallback enabled)
cargo build

# Build with optional features
cargo build --features image             # image crate integration
cargo build --features pdq               # PDQ perceptual hash
cargo build --features "image,pdq"       # both

# Test
cargo test                               # all tests
cargo test sha256                        # specific test by name
cargo test --test gpu_resize_test        # specific test file
cargo test --test cache_gpu_matcher_test # cache data GPU matching test
cargo test --features pdq                # tests requiring pdq feature

# Benchmark
cargo bench                              # all benchmarks
cargo bench --bench sha256_bench         # specific benchmark

# Lint
cargo clippy
cargo clippy --features image

# Run example
cargo run --example demo

# Generate docs
cargo doc --features image --no-deps
```
