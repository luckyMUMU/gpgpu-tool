# src/ - Core GPU Compute Infrastructure

## OVERVIEW

Capability layer: GPU context, buffer management, pipeline caching, batch submission. No business logic here - pure wgpu abstraction.

## WHERE TO LOOK

| Component | File | Role |
|-----------|------|------|
| GPU context | `context.rs` | `GpuContext`: device/queue/pipeline cache, sync/async creation |
| Buffer | `buffer.rs` | `GpuBuffer`: CPU-GPU data transfer, upload/download |
| Buffer pool | `buffer_pool.rs` | `BufferPool`: size-tiered buffer reuse |
| Pipeline | `pipeline.rs` | `ComputePipeline`: shader creation + dispatch |
| Batch submit | `batch.rs` | `GpuBatchSubmitter`: async batch job pattern |
| Error type | `error.rs` | `GpuError`: thiserror enum, 9 variants |

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
