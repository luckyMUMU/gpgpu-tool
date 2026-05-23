# src/ - Core GPU Compute Infrastructure

## OVERVIEW

Capability layer: GPU context, buffer management, pipeline caching, batch submission. No business logic here - pure wgpu abstraction.

## WHERE TO LOOK

| Component | File | Lines | Role | Dependencies |
|-----------|------|-------|------|-------------|
| Public API | `lib.rs` | ~130 | Crate root and documentation | (exports all modules) |
| GPU context | `context.rs` | ~170 | `GpuContext`: device/queue/pipeline cache, sync/async creation | error, pipeline |
| Buffer | `buffer.rs` | ~197 | `GpuBuffer`: CPU-GPU data transfer, upload/download | error, buffer_pool |
| Buffer pool | `buffer_pool.rs` | ~168 | `BufferPool`: size-tiered buffer reuse | buffer |
| Pipeline | `pipeline.rs` | ~197 | `ComputePipeline`: shader creation + dispatch | buffer, error |
| Batch submit | `batch.rs` | ~174 | `GpuBatchSubmitter`: async batch job pattern | buffer, error, pipeline, context |
| Error type | `error.rs` | ~32 | `GpuError`: thiserror enum, 9 variants | (none) |
| Tasks | `tasks/` | — | Business-layer algorithms | buffer, context, pipeline |

## CONVENTIONS

- **No `main.rs`** - library crate only
- **Pipeline cache**: Uses custom `fxhash` (not std hash) in `context.rs`
- **BufferPool**: Size-tiered bins, not exact match
- **Batch pattern**: `submit()` + `wait_all()` (not futures)
- **Sync wrapper**: `pollster::block_on` for async→sync conversion

## ANTI-PATTERNS

- **Do NOT add `main.rs`** without discussion
- **Do NOT use std hash** for pipeline cache - fxhash is intentional
- **Do NOT suppress errors** with `as any` - this is Rust, use proper `Result`
