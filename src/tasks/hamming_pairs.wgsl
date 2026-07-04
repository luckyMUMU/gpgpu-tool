// GPU 端 Hamming 距离候选对过滤着色器。
//
// 与 hamming.wgsl 的 hamming_distance_matrix 不同，此着色器不输出完整 N×M 距离矩阵，
// 而是在 GPU 端直接过滤 threshold，仅输出满足条件的候选对三元组 (query_idx, db_idx, distance)。
//
// 使用原子计数器管理输出位置，避免下载完整矩阵到 CPU。
// 对称模式：自动过滤 query_idx == db_idx 的自身匹配。
//
// 绑定布局（5 bindings）：
//   0: queries    — 查询哈希数据（storage, read）
//   1: database   — 数据库哈希数据（storage, read）
//   2: pairs      — 候选对输出（storage, read_write），每 3 个 u32 为一组 (qi, di, dist)
//   3: counter    — 原子计数器（storage, read_write），array<atomic<u32>, 1>
//   4: params     — 参数（uniform）
//
// params: vec4<u32>
//   .x = n (query count)
//   .y = m (database count)
//   .z = u32_per_hash
//   .w = threshold (max distance)
//
// Workgroup: 16×16 = 256 threads，使用共享内存 tile 优化（与 hamming_distance_matrix 相同）。

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> database: array<u32>;
@group(0) @binding(2) var<storage, read_write> pairs: array<u32>;
@group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>, 1>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;

// 共享内存 tile（与 hamming.wgsl 相同的布局）
const MAX_U32_PER_HASH: u32 = 128u;
const STRIDE: u32 = 129u;

var<workgroup> query_tile: array<u32, 2064>;      // 16 * 129 = 2064 u32s
var<workgroup> db_tile_pairs: array<u32, 2064>;   // 16 * 129 = 2064 u32s

@compute @workgroup_size(16, 16, 1)
fn hamming_distance_pairs(
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
    let threshold = params.w;

    // ── Phase 1: cooperatively load 16 queries ──
    if (ly == 0u && lx < n) {
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = queries[qi * u32_per_hash + i];
        }
    } else if (ly == 0u) {
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = 0u;
        }
    }

    // ── Phase 2: cooperatively load 16 database entries ──
    if (lx == 0u && di < m) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_pairs[ly * STRIDE + i] = database[di * u32_per_hash + i];
        }
    } else if (lx == 0u) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_pairs[ly * STRIDE + i] = 0u;
        }
    }

    // ── Phase 3: synchronize ──
    workgroupBarrier();

    // ── Phase 4: compute distance and filter ──
    if (qi < n && di < m) {
        var dist: u32 = 0u;
        for (var i = 0u; i < u32_per_hash; i++) {
            dist += countOneBits(
                query_tile[lx * STRIDE + i] ^ db_tile_pairs[ly * STRIDE + i]
            );
        }

        // 过滤：距离 ≤ threshold 且不是自身匹配（对称模式）
        if (dist <= threshold && qi != di) {
            // 原子获取输出槽位
            let slot = atomicAdd(&counter[0], 1u);
            // 写入三元组 (query_idx, db_idx, distance)
            pairs[slot * 3u] = qi;
            pairs[slot * 3u + 1u] = di;
            pairs[slot * 3u + 2u] = dist;
        }
    }
}

/// 非对称模式候选对过滤入口点。
///
/// 与 `hamming_distance_pairs` 逻辑相同，但不过滤 `qi != di`，
/// 适用于 queries 和 database 是不同集合的场景（非对称匹配）。
@compute @workgroup_size(16, 16, 1)
fn hamming_distance_pairs_asymmetric(
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
    let threshold = params.w;

    // ── Phase 1: cooperatively load 16 queries ──
    if (ly == 0u && lx < n) {
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = queries[qi * u32_per_hash + i];
        }
    } else if (ly == 0u) {
        for (var i = 0u; i < u32_per_hash; i++) {
            query_tile[lx * STRIDE + i] = 0u;
        }
    }

    // ── Phase 2: cooperatively load 16 database entries ──
    if (lx == 0u && di < m) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_pairs[ly * STRIDE + i] = database[di * u32_per_hash + i];
        }
    } else if (lx == 0u) {
        for (var i = 0u; i < u32_per_hash; i++) {
            db_tile_pairs[ly * STRIDE + i] = 0u;
        }
    }

    // ── Phase 3: synchronize ──
    workgroupBarrier();

    // ── Phase 4: compute distance and filter (no self-match filter) ──
    if (qi < n && di < m) {
        var dist: u32 = 0u;
        for (var i = 0u; i < u32_per_hash; i++) {
            dist += countOneBits(
                query_tile[lx * STRIDE + i] ^ db_tile_pairs[ly * STRIDE + i]
            );
        }

        // 过滤：距离 ≤ threshold（非对称模式不过滤自身匹配）
        if (dist <= threshold) {
            let slot = atomicAdd(&counter[0], 1u);
            pairs[slot * 3u] = qi;
            pairs[slot * 3u + 1u] = di;
            pairs[slot * 3u + 2u] = dist;
        }
    }
}
