# tests/ - Integration Tests

## OVERVIEW

Per-algorithm integration tests. One test file per algorithm, plus shared utilities in `common/`.

## STRUCTURE

| Test file | Tests | Notes |
|-----------|-------|-------|
| `bktree_test.rs` | `BkTree` construction, find, find_nearest, GPU hash compatibility | ~5 tests, includes large dataset (10K+ hashes) |
| `sha256_test.rs` | `Sha256Computer` correctness (empty, short, long, batch) | ~6 tests |
| `mean_hash_test.rs` | `MeanHashComputer` correctness (various sizes, batch) | ~3 tests |
| `median_hash_test.rs` | `MedianHashComputer` correctness | ~3 tests |
| `gradient_hash_test.rs` | `GradientHashComputer` correctness | ~3 tests |
| `block_hash_test.rs` | `BlockHashComputer` correctness | ~3 tests |
| `vert_gradient_hash_test.rs` | `VertGradientHashComputer` correctness | ~3 tests |
| `double_gradient_hash_test.rs` | `DoubleGradientHashComputer` correctness | ~3 tests |
| `large_image_perf_test.rs` | Large image batch performance (CPU vs GPU) | ~3 tests, processing 5000+ images |
| `gpu_resize_test.rs` | GPU zero-copy resize pipeline correctness | 2 tests verifying CPU vs GPU hash consistency |
| `real_image_bench` integration | Uses `tests/data/` real-world images | shared with bench workloads |
| `common/` | Shared test utilities | - |
| `data/` | Test data files | - |

## CONVENTIONS

- **One test per algorithm**: `*_test.rs` naming
- **GPU vs CPU validation**: Tests compare GPU results against CPU reference
- **No test framework**: Standard Rust `#[test]` attributes
- **Test data**: Images in `data/` directory

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new algorithm test | Follow `sha256_test.rs` pattern |
| Shared utilities | `common/` directory |
| Test images | `data/` directory |
