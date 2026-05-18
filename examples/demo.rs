use wgpu_compute_engine::{GpuContext, tasks::sha256::Sha256Computer};

fn main() {
    env_logger::init();

    println!("初始化 GPU 上下文...");
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    println!("适配器: {}", ctx.adapter_info());

    let sha256 = Sha256Computer::new(&mut ctx).expect("Sha256Computer 创建失败");

    let messages = vec![
        b"hello world".to_vec(),
        b"wgpu-compute-engine".to_vec(),
        vec![0xABu8; 100],
    ];

    println!("\n计算 {} 条消息的 SHA-256 哈希...", messages.len());
    let results = sha256.compute(&ctx, &messages).expect("计算失败");

    for (i, (msg, hash)) in messages.iter().zip(results.iter()).enumerate() {
        let hex_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        let msg_preview = if msg.len() > 32 {
            format!("{:?}... ({} bytes)", &msg[..32], msg.len())
        } else {
            format!("{:?}", msg)
        };
        println!("  [{}] {} => {}", i, msg_preview, hex_hash);
    }

    println!("\n完成！");
}
