# tests/ - Integration Tests

## OVERVIEW

Per-algorithm integration tests. One test file per algorithm, plus shared utilities in `common/`.

## STRUCTURE

```
tests/
├── sha256_test.rs           # SHA-256 GPU vs CPU validation
├── mean_hash_test.rs        # Mean hash tests
├── median_hash_test.rs      # Median hash tests
├── gradient_hash_test.rs    # Gradient hash tests
├── block_hash_test.rs       # Block hash tests
├── vert_gradient_hash_test.rs  # Vertical gradient tests
├── double_gradient_hash_test.rs # Double gradient tests
├── real_image_hash_test.rs  # Real image hash tests
├── large_image_perf_test.rs # Large image performance
├── common/                  # Shared test utilities
└── data/                    # Test data files
```

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
