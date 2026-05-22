# benches/ - Performance Benchmarks

## OVERVIEW

Criterion benchmarks with `harness = false`. One benchmark file per algorithm, plus shared utilities.

## STRUCTURE

```
benches/
├── sha256_bench.rs          # SHA-256 performance
├── mean_hash_bench.rs       # Mean hash benchmark
├── median_hash_bench.rs     # Median hash benchmark
├── gradient_hash_bench.rs   # Gradient hash benchmark
├── block_hash_bench.rs      # Block hash benchmark
├── vert_gradient_hash_bench.rs  # Vertical gradient
├── double_gradient_hash_bench.rs # Double gradient
├── common.rs                # Shared benchmark utilities
├── common/                  # Additional shared code
├── performance_report.md    # Performance analysis
└── performance_report_optimized.md # Optimized performance report
```

## CONVENTIONS

- **`harness = false`**: Custom Criterion setup in `Cargo.toml`
- **One bench per algorithm**: `*_bench.rs` naming
- **Reports**: Generated in `target/criterion/`
- **Shared code**: `common.rs` + `common/` directory

## COMMANDS

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench sha256_bench
```

## WHERE TO LOOK

| Task | Location |
|------|----------|
| Add new benchmark | Follow `sha256_bench.rs` pattern |
| Shared utilities | `common.rs` |
| Performance reports | `performance_report*.md` |
