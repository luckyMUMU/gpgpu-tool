// 链式 SHA-256：单 dispatch 内处理消息的全部 block。
// 与原始 sha256.wgsl 共享完全相同的压缩逻辑。
// 输入布局: [block_counts...][all_blocks...]

const K: array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
);

fn ch(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (~x & z); }
fn maj(x: u32, y: u32, z: u32) -> u32 { return (x & y) ^ (x & z) ^ (y & z); }
fn big_sigma0(x: u32) -> u32 { return rotate_right(x, 2u) ^ rotate_right(x, 13u) ^ rotate_right(x, 22u); }
fn big_sigma1(x: u32) -> u32 { return rotate_right(x, 6u) ^ rotate_right(x, 11u) ^ rotate_right(x, 25u); }
fn small_sigma0(x: u32) -> u32 { return rotate_right(x, 7u) ^ rotate_right(x, 18u) ^ (x >> 3u); }
fn small_sigma1(x: u32) -> u32 { return rotate_right(x, 17u) ^ rotate_right(x, 19u) ^ (x >> 10u); }
fn rotate_right(v: u32, n: u32) -> u32 { return (v >> n) | (v << (32u - n)); }

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> hashes: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@group(0) @binding(3) var<storage, read> prefix_sums: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let msg_idx = gid.x;
    let message_count = params.x;
    if (msg_idx >= message_count) { return; }

    // O(1) 前缀和查找：prefix_sums[msg_idx] = Σ(block_counts[0..msg_idx]) * 16
    let data_offset = message_count + prefix_sums[msg_idx];
    let block_count = input_data[msg_idx];

    // 初始哈希
    var h0 = 0x6a09e667u; var h1 = 0xbb67ae85u;
    var h2 = 0x3c6ef372u; var h3 = 0xa54ff53au;
    var h4 = 0x510e527fu; var h5 = 0x9b05688cu;
    var h6 = 0x1f83d9abu; var h7 = 0x5be0cd19u;

    // w 数组声明在循环外，避免某些 WGSL 后端在循环内重复声明的问题
    var w: array<u32, 64>;

    for (var blk = 0u; blk < block_count; blk = blk + 1u) {
        let block_base = data_offset + blk * 16u;
        for (var i = 0u; i < 16u; i = i + 1u) {
            w[i] = input_data[block_base + i];
        }

        // 消息调度
        for (var i = 16u; i < 64u; i = i + 1u) {
            let s0 = small_sigma0(w[i - 15u]);
            let s1 = small_sigma1(w[i - 2u]);
            w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
        }

        // 工作变量（以 s_ 前缀避免与标准库潜在冲突）
        var sa = h0; var sb = h1; var sc = h2; var sd = h3;
        var se = h4; var sf = h5; var sg = h6; var sh = h7;

        // 64 轮压缩
        for (var i = 0u; i < 64u; i = i + 1u) {
            let S1 = big_sigma1(se);
            let ch_val = ch(se, sf, sg);
            let temp1 = sh + S1 + ch_val + K[i] + w[i];
            let S0 = big_sigma0(sa);
            let maj_val = maj(sa, sb, sc);
            let temp2 = S0 + maj_val;

            sh = sg; sg = sf; sf = se; se = sd + temp1;
            sd = sc; sc = sb; sb = sa; sa = temp1 + temp2;
        }

        // 累加当前 block 的压缩结果到哈希状态
        h0 = h0 + sa; h1 = h1 + sb; h2 = h2 + sc; h3 = h3 + sd;
        h4 = h4 + se; h5 = h5 + sf; h6 = h6 + sg; h7 = h7 + sh;
    }

    // 输出
    let out_base = msg_idx * 8u;
    hashes[out_base + 0u] = h0;
    hashes[out_base + 1u] = h1;
    hashes[out_base + 2u] = h2;
    hashes[out_base + 3u] = h3;
    hashes[out_base + 4u] = h4;
    hashes[out_base + 5u] = h5;
    hashes[out_base + 6u] = h6;
    hashes[out_base + 7u] = h7;
}
