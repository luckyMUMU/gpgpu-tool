# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPGPU-tool is a Rust GPU-accelerated algorithm library built on wgpu v24. It provides SHA-256 hashing, perceptual image hashing (6 algorithms), BK-tree similarity search, GPU convolution, GPU-accelerated hash matching, and more. The core principle: **GPU-first, CPU-fallback** — all algorithms have GPU implementations with automatic CPU degradation when GPU is unavailable.

**Library crate only** — there is no `main.rs`. Run examples via `cargo run --example`.

## Build & Test Commands

```bash
# Build (default: cpu-fallback enabled)
cargo build

# Build with optional features
cargo build --features image          # image crate integration
cargo build --features pdq            # PDQ perceptual hash
cargo build --features "image,pdq"    # both

# Test
cargo test                            # all tests
cargo test sha256                     # specific test by name
cargo test --test gpu_resize_test     # specific test file
cargo test --test cache_gpu_matcher_test  # cache data GPU matching test
cargo test --features pdq             # tests requiring pdq feature

# Benchmark
cargo bench                           # all benchmarks
cargo bench --bench sha256_bench      # specific benchmark

# Lint
cargo clippy
cargo clippy --features image
```

## Architecture

Two-layer design: **capability layer** (wgpu abstractions) and **business layer** (algorithms).

```
src/
├── lib.rs              # Public API exports (~40 re-exports)
├── context.rs          # GpuContext: device/queue/pipeline cache (fxhash-based)
├── buffer.rs           # GpuBuffer: CPU↔GPU data transfer
├── buffer_pool.rs      # BufferPool: size-tiered GPU buffer reuse (2^n bins)
├── pipeline.rs         # ComputePipeline: shader compile + dispatch
├── batch.rs            # GpuBatchSubmitter: merge multiple dispatches into one submit
├── error.rs            # GpuError: unified error enum (thiserror)
├── pixel_pack.rs       # u8↔u32 pixel format conversion (GPU shaders use u32)
├── poll_counter.rs     # GPU poll counter
├── pipeline_builder.rs # Declarative pipeline builder (chain GPU processing steps)
├── backend_dispatcher.rs # GPU/CPU backend selection trait
└── tasks/              # Business-layer algorithms
    ├── sha256.rs + sha256.wgsl         # SHA-256 GPU parallel hash
    ├── sha256_cpu.rs                   # SHA-256 CPU fallback (cpu-fallback feature)
    ├── sha256_chained.wgsl             # SHA-256 chained compute shader
    ├── phasher.rs                      # Perceptual hash orchestrator (6 algorithms + GPU resize)
    ├── phasher_cpu.rs                  # Perceptual hash CPU fallback
    ├── hash_common.rs                  # Shared traits, macros, HashSize, PhashParams
    ├── mean_hash.rs / .wgsl            # Mean hash (thin wrapper via macros)
    ├── median_hash.rs / .wgsl          # Median hash
    ├── gradient_hash.rs / .wgsl        # Gradient hash
    ├── block_hash.rs / .wgsl           # Block hash
    ├── vert_gradient_hash.rs / .wgsl   # Vertical gradient hash
    ├── double_gradient_hash.rs / .wgsl # Double gradient hash
    ├── convolution.rs + convolution.wgsl # GPU 2D convolution (Full2D / Separable)
    ├── gaussian_blur.rs                # GPU Gaussian blur (wraps GpuConvolution)
    ├── gpu_resize.rs + resize.wgsl     # GPU image resize (zero-copy pipeline support)
    ├── dihedral.rs                     # D4 group 8 rotations/flips (64/256/1024/4096-bit)
    ├── matcher.rs                      # 64-bit hash matching: strategy + chain + facade
    ├── matcher_bytes.rs                # Variable-length hash matching (HashBytes)
    ├── bktree.rs                       # BK-tree approximate nearest neighbor (64-bit)
    ├── bktree_bytes.rs                 # BK-tree for variable-length hashes
    ├── hash_bytes.rs                   # HashBytes type (Vec<u8> wrapper)
    ├── hamming.wgsl                    # GPU Hamming distance (matrix + nearest-neighbor + large-hash)
    ├── gpu_matcher.rs                  # GPU-accelerated hash matcher (3 pipelines)
    ├── gpu_image_matcher.rs            # End-to-end GPU image matcher
    ├── pdq_hash.rs + pdq_hash.wgsl     # PDQ hash (feature: pdq), GPU DCT + CPU quantization
    └── mod.rs
```

### Key Architectural Patterns

- **Pipeline caching**: `GpuContext` caches compiled shaders via custom fxhash (not std hash), keyed by `(hash, workgroup_size)`. Avoid recompilation.
- **Buffer pooling**: `BufferPool` uses 2^n size-tiered bins with max_per_class cap. Buffers >1MB are always destroyed on release.
- **Batch submission**: `GpuBatchSubmitter` encodes multiple dispatches into one `CommandEncoder`, submits once via `queue.submit()`. Use `submit()` + `wait_all()` pattern (not futures).
- **Zero-copy pipelines**: `_gpu` suffixed methods return `GpuBuffer` instead of `Vec<u8>`, allowing GPU→GPU data flow without CPU round-trips (e.g., resize→hash).
- **Perceptual hash macros**: 6 hash algorithms are thin wrappers using `declare_phash_computer!` + `impl_phash_computer_simple!` macros from `hash_common.rs`.
- **Fixed bind group layout**: All compute pipelines use 3 bindings: input (storage read), output (storage read_write), params (uniform or storage). See `ComputePipeline` in `pipeline.rs`.
- **Variable-length hash**: `HashBytes`(Vec<u8>) supports arbitrary bit widths; `*Bytes` suffixed matchers/BK-trees provide consistent API with 64-bit versions.
- **GPU matcher 3-pipeline architecture**: distance_matrix (N×M matrix) + nearest_neighbor (≤1024-bit) + nearest_neighbor_large (≤4096-bit, workgroup_size=32).
- **Dihedral transforms**: CPU-only bit matrix operations for 64/256/1024/4096-bit hashes, enabling rotation-invariant matching without GPU.

### Feature Flags

| Feature | Default | Purpose |
|---------|---------|---------|
| `cpu-fallback` | ✅ | Enables CPU implementations (`Sha256Cpu`, `PHasherCpu`), depends on `sha2` crate |
| `image` | ❌ | Enables `image` crate integration for `DynamicImage` input |
| `pdq` | ❌ | Enables PDQ perceptual hash (DCT-based 256-bit hash) |

## Conventions

- **WGSL shaders** are separate `.wgsl` files co-located in `src/tasks/`, not inline strings
- **CPU fallback** files use `_cpu.rs` suffix, same directory as GPU implementation, matching API signatures
- **Perceptual hash algorithms** follow thin-wrapper pattern: `.rs` + `.wgsl` pair, delegate to `hash_common.rs` macros
- **Variable-length hash** files use `_bytes.rs` suffix, providing `HashBytes`-based versions of 64-bit APIs
- **Tests**: one `*_test.rs` file per algorithm in `tests/`, shared utilities in `tests/common/`
- **Benchmarks**: one `*_bench.rs` file per algorithm in `benches/`, `harness = false` (Criterion)
- **No `unsafe`** blocks in the codebase
- **Uniform alignment**: WGSL uniform buffers require 16-byte alignment; kernel arrays use storage binding instead
- **GPU pipeline selection**: `u32_per_hash > 32` → large-hash pipeline (workgroup_size=32), else standard (workgroup_size=256)

## Known Gotchas

- **GPU dispatch overhead**: ~1.6ms per dispatch; small batches may be slower than CPU
- **convolution.wgsl**: `ConvParams` kernel array uses storage binding (not uniform) to avoid 16-byte alignment issues
- **PDQ hash**: requires `--features pdq` to compile; not in default build
- **DX12 backend**: `hash_size=64` may fail shader compilation due to register pressure
- **hamming.wgsl shared memory**: `db_tile_nearest_large` uses 16KB workgroup memory (32×128 u32); `db_tile_nearest` uses 8KB (256×32 u32)
- **hash_bytes_to_u32**: Empty or zero-length hashes return empty result (defensive check added)
- **Cache data test**: JSON file has ~2.3M entries; `load_hashes_from_cache` reads only first N bytes and truncates JSON to avoid parsing entire 1.4GB file

## Performance Benchmarks

GPU matching with Czkawka cache data (20000 × 1024-bit hashes, threshold=20):

| Method | Scale | Time | Matches |
|--------|-------|------|---------|
| CPU linear scan (200 queries × 20000 db) | 4M comparisons | 17.1s | 200 |
| CPU estimated full (20000 × 20000) | 400M comparisons | ~1711s | — |
| **GPU nearest neighbor (20000 × 20000)** | **400M comparisons** | **2.2s** | 20000 |
| GPU distance matrix (500 × 500) | 250K comparisons | 14.6ms | 532 |

**GPU acceleration: 778x** vs CPU for full-scale nearest-neighbor search.

## Adding a New Algorithm

**Full algorithm** (like SHA-256): Create `src/tasks/new_algo.rs` + `new_algo.wgsl`, implement struct with `Arc<ComputePipeline>`, follow the pack→buffer→dispatch→download pattern.

**Perceptual hash**: Create `src/tasks/new_hash.rs` + `new_hash.wgsl`, use `declare_phash_computer!` + `impl_phash_computer_simple!` macros, add variant to `HashAlgorithm` enum in `phasher.rs`.

**Convolution-based filter**: Create `src/tasks/new_filter.rs`, wrap `GpuConvolution`, provide `_gpu` zero-copy method.

**Variable-length hash matcher**: Create `src/tasks/new_matcher_bytes.rs`, implement `HashMatcherBytes` trait, add `*Bytes` suffix to distinguish from 64-bit version.
