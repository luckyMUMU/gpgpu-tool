# src/tasks/ - GPU Algorithm Implementations

## OVERVIEW

Business-layer GPU algorithms: SHA-256 parallel hash + 6 perceptual image hash variants. Each algorithm pairs a `.rs` wrapper with a `.wgsl` shader.

## STRUCTURE

```
tasks/
├── sha256.rs / .wgsl        # SHA-256 GPU hash (654 lines, reference impl)
├── phasher.rs               # Perceptual hash orchestrator (197 lines)
├── hash_common.rs           # Shared utilities + macros (202 lines)
├── mean_hash.rs / .wgsl     # Mean hash (14-line wrapper)
├── median_hash.rs / .wgsl   # Median hash (14-line wrapper)
├── gradient_hash.rs / .wgsl # Gradient hash (14-line wrapper)
├── block_hash.rs / .wgsl    # Block hash (14-line wrapper)
├── vert_gradient_hash.rs / .wgsl  # Vertical gradient (14-line wrapper)
└── double_gradient_hash.rs / .wgsl # Double gradient (14-line wrapper)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new algorithm | Follow `sha256.rs` pattern | Computer struct + WGSL + mod.rs export |
| Perceptual hash orchestration | `phasher.rs` | `PerceptualHasher`, `HashAlgorithm` enum |
| Shared phash logic | `hash_common.rs` | `PerceptualHashComputer` trait, `compute_phash()`, macros |
| Hash algorithm sizes | `phasher.rs:24` | `target_size()` match arms |
| WGSL shaders | Co-located `.wgsl` files | Same name as `.rs` module |

## CONVENTIONS

- **Perceptual hash wrappers**: 14-line thin delegators using `declare_phash_computer!` + `impl_phash_computer_simple!` macros from `hash_common.rs`
- **SHA-256**: Full implementation (654 lines) - reference for new algorithms
- **WGSL inclusion**: `include_str!("name.wgsl")` at top of `.rs` file
- **Workgroup size**: Default `[256, 1, 1]` unless algorithm needs different

## ANTI-PATTERNS

- **Do NOT inline WGSL** - always use separate `.wgsl` files
- **Do NOT duplicate phash logic** - use `compute_phash()` from `hash_common.rs`
- **Do NOT skip trait impl** - all algorithms must implement `PerceptualHashComputer`
