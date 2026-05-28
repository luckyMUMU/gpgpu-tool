use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gpgpu_tool::tasks::bktree::{BkTree, hamming_distance};

fn generate_clustered_hashes(clusters: usize, per_cluster: usize, spread: u32) -> Vec<u64> {
    use rand::Rng;
    let mut rng = rand::rng();
    let mut hashes = Vec::with_capacity(clusters * per_cluster);
    for _ in 0..clusters {
        let center: u64 = rng.random();
        for _ in 0..per_cluster {
            let mut h = center;
            for _ in 0..spread {
                let bit = 1u64 << (rng.random_range(0..64));
                h ^= bit;
            }
            hashes.push(h);
        }
    }
    hashes
}

fn bench_bktree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bktree_build");
    for size in [1_000, 10_000, 100_000] {
        let hashes = generate_clustered_hashes(size / 10, 10, 5);
        group.bench_with_input(BenchmarkId::from_parameter(size), &hashes, |b, hashes| {
            b.iter(|| {
                BkTree::from_hashes(black_box(hashes.iter().copied()))
            });
        });
    }
    group.finish();
}

fn bench_bktree_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("bktree_find");
    for size in [1_000, 10_000, 100_000] {
        let hashes = generate_clustered_hashes(size / 10, 10, 5);
        let tree = BkTree::from_hashes(hashes.iter().copied());
        let queries: Vec<u64> = hashes[0..10].to_vec();

        group.bench_with_input(BenchmarkId::from_parameter(size), &queries, |b, queries| {
            b.iter(|| {
                for &q in queries {
                    black_box(tree.find(black_box(q), 5));
                }
            });
        });
    }
    group.finish();
}

fn bench_bktree_find_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("bktree_find_nearest");
    for size in [1_000, 10_000, 100_000] {
        let hashes = generate_clustered_hashes(size / 10, 10, 5);
        let tree = BkTree::from_hashes(hashes.iter().copied());
        let queries: Vec<u64> = hashes[0..10].to_vec();

        group.bench_with_input(BenchmarkId::from_parameter(size), &queries, |b, queries| {
            b.iter(|| {
                for &q in queries {
                    black_box(tree.find_nearest(black_box(q)));
                }
            });
        });
    }
    group.finish();
}

fn bench_brute_force_find(c: &mut Criterion) {
    let mut group = c.benchmark_group("brute_force_find");
    for size in [1_000, 10_000, 100_000] {
        let hashes = generate_clustered_hashes(size / 10, 10, 5);
        let queries: Vec<u64> = hashes[0..10].to_vec();

        group.bench_with_input(BenchmarkId::from_parameter(size), &queries, |b, queries| {
            b.iter(|| {
                for &q in queries {
                    let results: Vec<(u64, u32)> = hashes.iter()
                        .filter_map(|&h| {
                            let d = hamming_distance(h, q);
                            if d <= 5 { Some((h, d)) } else { None }
                        })
                        .collect();
                    black_box(results);
                }
            });
        });
    }
    group.finish();
}

fn bench_hamming_distance(c: &mut Criterion) {
    c.bench_function("hamming_distance", |b| {
        let a = 0x1234567890abcdefu64;
        let b_val = 0xfedcba0987654321u64;
        b.iter(|| {
            black_box(hamming_distance(black_box(a), black_box(b_val)))
        });
    });
}

criterion_group!(
    benches,
    bench_bktree_build,
    bench_bktree_find,
    bench_bktree_find_nearest,
    bench_brute_force_find,
    bench_hamming_distance,
);
criterion_main!(benches);
