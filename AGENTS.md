# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-20
**Commit:** 7bb56f8
**Branch:** master

## OVERVIEW

wgpu-compute-engine: Cross-platform GPU compute engine using wgpu. Provides GPU parallel acceleration for CPU-intensive tasks (SHA-256, perceptual image hashing). Rust 2021 edition, wgpu v24.

## STRUCTURE

```
wgpu-tool/
├── src/                    # Library code (no main.rs - lib crate only)
│   ├── lib.rs              # Public API exports
│   ├── context.rs          # GpuContext: device/queue/pipeline cache
│   ├── buffer.rs           # GpuBuffer: CPU-GPU data transfer
│   ├── buffer_pool.rs      # BufferPool: buffer reuse by size tier
│   ├── pipeline.rs         # ComputePipeline: compute dispatch
│   ├── batch.rs            # GpuBatchSubmitter: async batch submit
│   ├── error.rs            # GpuError: unified error type
│   └── tasks/              # Business-layer algorithms + WGSL shaders
├── tests/                  # Integration tests (per-algorithm)
├── benches/                # Criterion benchmarks (harness=false)
├── examples/               # Usage demo (demo.rs)
├── openspec/               # Design docs & change management
└── Cargo.toml              # Feature flags: image (optional)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| GPU init/dispatch | `src/context.rs` | Sync (`new_sync`) and async (`new`) creation |
| Buffer management | `src/buffer.rs`, `src/buffer_pool.rs` | Pool tiers by size |
| Pipeline caching | `src/context.rs` → `PipelineCache` | fxhash-based, auto-compile |
| Batch submit | `src/batch.rs` | Async submit, wait_all pattern |
| SHA-256 GPU | `src/tasks/sha256.rs` + `.wgsl` | 654 lines, largest file |
| Image hashing | `src/tasks/{mean,median,block,gradient,double_gradient,vert_gradient}_hash.rs` | All 14-line wrappers + `phasher.rs` (197 lines) |
| Hash common utils | `src/tasks/hash_common.rs` | Shared perceptual hash logic |
| Error handling | `src/error.rs` | `GpuError` enum, thiserror |
| Integration tests | `tests/*_test.rs` | One per algorithm |
| Benchmarks | `benches/*_bench.rs` | Criterion, no harness |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `GpuContext` | struct | `context.rs:72` | Main entry: device, queue, pipeline cache |
| `GpuBuffer` | struct | `buffer.rs` | CPU-GPU buffer transfer |
| `BufferPool` | struct | `buffer_pool.rs` | Size-tiered buffer reuse |
| `ComputePipeline` | struct | `pipeline.rs` | Compute pipeline creation & dispatch |
| `GpuBatchSubmitter` | struct | `batch.rs` | Async batch job submission |
| `GpuError` | enum | `error.rs:5` | Unified error type |
| `Sha256Computer` | struct | `tasks/sha256.rs` | GPU SHA-256 parallel hash |
| `PHasher` | struct | `tasks/phasher.rs` | Perceptual hash orchestrator |

## CONVENTIONS

- **Rust 2021 edition**, wgpu v24
- **No `main.rs`** - library crate only, examples via `cargo run --example`
- **WGSL shaders** co-located with Rust wrappers in `src/tasks/` (`.wgsl` alongside `.rs`)
- **Feature flag**: `image` feature gates image-related deps (optional in prod, required in dev)
- **Benchmarks**: `harness = false` in Cargo.toml - custom Criterion setup
- **Error type**: `thiserror` for `GpuError`, no `Result` alias
- **Async pattern**: `pollster::block_on` for sync wrappers

## ANTI-PATTERNS (THIS PROJECT)

- **No `unsafe`** blocks found - pure safe Rust
- **No TODO/FIXME/HACK** markers - clean codebase
- **No `main.rs`** - do not add binary entry point without discussion
- **WGSL inline** - shaders are separate `.wgsl` files, NOT inline strings

## UNIQUE STYLES

- **Pipeline cache** uses custom `fxhash` (not std hash) for shader source hashing
- **BufferPool** uses size-tiered bins (not exact match)
- **Batch submitter** uses `submit()` + `wait_all()` pattern (not futures)
- **Perceptual hash** modules are thin 14-line wrappers delegating to `phasher.rs` + WGSL

## COMMANDS

```bash
# Build
cargo build
cargo build --release
cargo build --features image

# Run example
cargo run --example demo

# Test
cargo test

# Benchmark (all)
cargo bench

# Benchmark (specific)
cargo bench --bench sha256_bench

# Lint
cargo clippy
```

## NOTES

- `target/` contains ~2000 build artifacts - excluded from all analysis
- GPU adapter selection: `HighPerformance` preference, all backends
- wgpu backends: Vulkan, Metal, DX12, WebGPU (auto-selected)
- SHA-256 is the reference implementation (654 lines) - new algorithms should follow its pattern
- Image hash algorithms share common infrastructure via `phasher.rs`
