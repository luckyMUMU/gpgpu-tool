// Hamming distance GPU compute shader — shared memory tile optimized.
//
// Three entry points:
//   hamming_distance_matrix      — workgroup_size(16,16,1), computes N×M distance matrix
//   find_nearest_neighbor        — workgroup_size(256), each query finds nearest neighbor (≤1024-bit)
//   find_nearest_neighbor_large  — workgroup_size(32), for 1024-bit~4096-bit hashes
//
// Input: queries[N] and database[M] as u32 arrays
// Params: HammingParams { n, m, u32_per_hash, threshold }
// Output: distances[N*M] (u32 distance matrix) for hamming_distance_matrix
// Output: nearest[N] as (index: u32, distance: u32) pairs for find_nearest_neighbor*
//
// Optimization: shared memory tile to reduce global memory bandwidth.
//   - hamming_distance_matrix: each query/db entry loaded once per 16×16 tile,
//     reused 16 times from shared memory (16× bandwidth reduction).
//   - find_nearest_neighbor: 256 threads cooperatively load 256 db entries,
//     each thread scans from shared memory (256× bandwidth reduction for db reads).
//   - find_nearest_neighbor_large: 32 threads cooperatively load 32 db entries,
//     each thread scans from shared memory (32× bandwidth reduction for db reads).
//
// Shared memory budget (MAX_U32_PER_HASH=128):
//   hamming_distance_matrix:      16×128×4 + 16×128×4 = 16384 bytes = 16KB
//   find_nearest_neighbor:        256×32×4            = 32768 bytes = 32KB (≤1024-bit only)
//   find_nearest_neighbor_large:  32×128×4            = 16384 bytes = 16KB (≤4096-bit)
//   All well under typical 64KB per workgroup limit.

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> database: array<u32>;
@group(0) @binding(2) var<storage, read_write> distances: array<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// params.x = n (query count)
// params.y = m (database count)
// params.z = u32_per_hash (number of u32s per hash, e.g. 2 for u64)
// params.w = threshold (max distance for find_nearest_neighbor, u32::MAX = no limit)

// Maximum u32s per hash for shared memory sizing. Supports up to 4096-bit hashes (hash_size=64).
// Note: shared memory must be statically sized in WGSL; actual processing uses params.z (u32_per_hash).
// Stride is MAX_U32_PER_HASH + 1 (129) to avoid 32-way bank conflicts: 128 % 32 == 0 causes
// all threads in a warp to access the same bank. 129 % 32 == 1 distributes across banks.
const MAX_U32_PER_HASH: u32 = 128u;
const STRIDE: u32 = 129u;

// ── 模块作用域 var<workgroup> 声明（WGSL 规范要求）──────────────
// hamming_distance_matrix 使用的共享内存 tile
var<workgroup> query_tile: array<u32, 2064>;      // 16 * 129 = 2064 u32s = 8256 bytes
var<workgroup> db_tile_matrix: array<u32, 2064>;  // 16 * 129 = 2064 u32s = 8256 bytes

// find_nearest_neighbor 使用的共享内存 tile（仅用于 ≤1024-bit 哈希）
// Note: stride 32 has bank conflicts (32 % 32 == 0) but this shader's sequential scan
// pattern makes the impact negligible. The FXC compiler rejects non-power-of-2 strides
// in non-uniform control flow, so we keep stride 32 here.
var<workgroup> db_tile_nearest: array<u32, 8192>;  // 256 * 32 = 8192 u32s = 32768 bytes

// find_nearest_neighbor_large 使用的共享内存 tile（用于 ≤4096-bit 哈希）
var<workgroup> db_tile_nearest_large: array<u32, 4128>;  // 32 * 129 = 4128 u32s = 16512 bytes

// ─────────────────────────────────────────────────────────────────────
// Entry point 1: Hamming distance matrix with shared memory tile
// ─────────────────────────────────────────────────────────────────────
//
// Workgroup layout: 16×16 = 256 threads
//   - lid.x (0..15) = query index within tile
//   - lid.y (0..15) = database index within tile
//
// Data flow:
//   1. First row (lid.y==0) loads 16 queries cooperatively
//   2. First column (lid.x==0) loads 16 db entries cooperatively
//   3. workgroupBarrier() — all data visible
//   4. Each thread reads from shared memory, computes XOR + popcount
//
// Bandwidth savings: each query is loaded once (instead of 16 times),
// each db entry is loaded once (instead of 16 times) → 16× reduction.

@compute @workgroup_size(16, 16, 1)
fn hamming_distance_matrix(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qi = gid.x;  // global query index
    let di = gid.y;  // global database index
    let lx = lid.x;  // local query index (0..15)
    let ly = lid.y;  // local database index (0..15)

    let n = params.x;
    let m = params.y;
    let u32_per_hash = params.z;

    // ── Phase 1: cooperatively load 16 queries ──
    // Only the first row (ly == 0) participates.
    if (ly == 0u && lx < n) {
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = queries[qi * u32_per_hash + i];
        }
    } else if (ly == 0u) {
        // Out-of-bounds: zero-fill to avoid undefined reads
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = 0u;
        }
    }

    // ── Phase 2: cooperatively load 16 database entries ──
    // Only the first column (lx == 0) participates.
    if (lx == 0u && di < m) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_matrix[ly * STRIDE + i] = database[di * u32_per_hash + i];
        }
    } else if (lx == 0u) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_matrix[ly * STRIDE + i] = 0u;
        }
    }

    // ── Phase 3: synchronize ──
    workgroupBarrier();

    // ── Phase 4: compute from shared memory ──
    if (qi < n && di < m) {
        var dist: u32 = 0u;
        for (var i = 0u; i < u32_per_hash; i++) {
            dist += countOneBits(
                query_tile[lx * STRIDE + i] ^ db_tile_matrix[ly * STRIDE + i]
            );
        }
        distances[qi * m + di] = dist;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Entry point 2: Find nearest neighbor with shared memory tile
// ─────────────────────────────────────────────────────────────────────
//
// 仅用于 ≤1024-bit 哈希（u32_per_hash ≤ 32）。
// 对于更大的哈希，使用 find_nearest_neighbor_large。
//
// Workgroup layout: 256×1 = 256 threads
//   - Each thread handles one query (global_invocation_id.x)
//   - 256 threads cooperate to load 256 db entries per tile

@compute @workgroup_size(256)
fn find_nearest_neighbor(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qi = gid.x;  // query index (one per thread)
    let tid = lid.x;  // local thread index (0..255)

    let n = params.x;
    let m = params.y;
    let u32_per_hash = params.z;
    let threshold = params.w;

    if (qi >= n) {
        return;
    }

    var best_index: u32 = 0xFFFFFFFFu;
    var best_distance: u32 = 0xFFFFFFFFu;

    // ── Tile loop: process database in chunks of 256 ──
    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= m) { break; }

        let db_global_idx = tile_start + tid;

        // ── Phase 1: cooperatively load 256 db entries ──
        if (db_global_idx < m) {
            for (var i = 0u; i < u32_per_hash; i++) {
                db_tile_nearest[tid * 32u + i] = database[db_global_idx * u32_per_hash + i];
            }
        } else {
            // Pad out-of-bounds entries with zeros
            for (var i = 0u; i < u32_per_hash; i++) {
                db_tile_nearest[tid * 32u + i] = 0u;
            }
        }

        // ── Phase 2: synchronize ──
        workgroupBarrier();

        // ── Phase 3: scan shared memory tile ──
        var local_di: u32 = 0u;
        loop {
            if (local_di >= 256u || tile_start + local_di >= m) { break; }

            var dist: u32 = 0u;
            for (var i = 0u; i < u32_per_hash; i++) {
                dist += countOneBits(
                    queries[qi * u32_per_hash + i] ^ db_tile_nearest[local_di * 32u + i]
                );
            }

            if (dist < best_distance) {
                best_distance = dist;
                best_index = tile_start + local_di;
            }

            local_di++;
        }

        // ── Phase 4: sync before next tile load ──
        workgroupBarrier();

        tile_start += 256u;
    }

    // Apply threshold: if best_distance > threshold, invalidate result
    if (best_distance > threshold) {
        best_index = 0xFFFFFFFFu;
        best_distance = 0xFFFFFFFFu;
    }

    // Output as (index, distance) pairs interleaved
    distances[qi * 2u] = best_index;
    distances[qi * 2u + 1u] = best_distance;
}

// ─────────────────────────────────────────────────────────────────────
// Entry point 3: Find nearest neighbor for large hashes (≤4096-bit)
// ─────────────────────────────────────────────────────────────────────
//
// 用于 u32_per_hash > 32 的场景（如 hash_size=64，4096-bit = 128 u32s）。
// 使用更小的 workgroup (32 threads) 以控制共享内存用量。
//
// Workgroup layout: 32×1 = 32 threads
//   - Each thread handles one query (global_invocation_id.x)
//   - 32 threads cooperate to load 32 db entries per tile
//
// Shared memory: 32 × 128 × 4 = 16384 bytes = 16KB

@compute @workgroup_size(32)
fn find_nearest_neighbor_large(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qi = gid.x;  // query index (one per thread)
    let tid = lid.x;  // local thread index (0..31)

    let n = params.x;
    let m = params.y;
    let u32_per_hash = params.z;
    let threshold = params.w;

    if (qi >= n) {
        return;
    }

    var best_index: u32 = 0xFFFFFFFFu;
    var best_distance: u32 = 0xFFFFFFFFu;

    // ── Tile loop: process database in chunks of 32 ──
    var tile_start: u32 = 0u;
    loop {
        if (tile_start >= m) { break; }

        let db_global_idx = tile_start + tid;

        // ── Phase 1: cooperatively load 32 db entries ──
        if (db_global_idx < m) {
            for (var i = 0u; i < u32_per_hash; i++) {
                db_tile_nearest_large[tid * STRIDE + i] = database[db_global_idx * u32_per_hash + i];
            }
        } else {
            // Pad out-of-bounds entries with zeros
            for (var i = 0u; i < u32_per_hash; i++) {
                db_tile_nearest_large[tid * STRIDE + i] = 0u;
            }
        }

        // ── Phase 2: synchronize ──
        workgroupBarrier();

        // ── Phase 3: scan shared memory tile ──
        var local_di: u32 = 0u;
        loop {
            if (local_di >= 32u || tile_start + local_di >= m) { break; }

            var dist: u32 = 0u;
            for (var i = 0u; i < u32_per_hash; i++) {
                dist += countOneBits(
                    queries[qi * u32_per_hash + i] ^ db_tile_nearest_large[local_di * STRIDE + i]
                );
            }

            if (dist < best_distance) {
                best_distance = dist;
                best_index = tile_start + local_di;
            }

            local_di++;
        }

        // ── Phase 4: sync before next tile load ──
        workgroupBarrier();

        tile_start += 32u;
    }

    // Apply threshold: if best_distance > threshold, invalidate result
    if (best_distance > threshold) {
        best_index = 0xFFFFFFFFu;
        best_distance = 0xFFFFFFFFu;
    }

    // Output as (index, distance) pairs interleaved
    distances[qi * 2u] = best_index;
    distances[qi * 2u + 1u] = best_distance;
}
