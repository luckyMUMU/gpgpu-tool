//! 从 Czkawka 缓存 JSON 文件中提取哈希值，保存为二进制格式。
//!
//! 输出格式：
//!   - 文件头：8 字节 u64 LE = 哈希数量
//!   - 文件头：4 字节 u32 LE = 每条哈希字节数
//!   - 数据区：每条哈希连续存储（hash_size=32 → 128 字节/条）
//!
//! 用法：cargo run --example extract_cache_hashes

use std::fs;
use std::io::{BufWriter, Seek, Write};

use serde::de::{Deserializer as _, SeqAccess, Visitor};
use serde::Deserialize;
use std::fmt;

/// Czkawka 缓存条目（JSON 反序列化用）
#[derive(Deserialize)]
struct CacheEntry {
    #[allow(dead_code)]
    path: String,
    hash: Vec<u8>,
}

/// 流式处理 JSON 数组的 Visitor，逐条提取哈希并写入输出文件
struct HashExtractor {
    out_file: BufWriter<fs::File>,
    count: u64,
    hash_bytes_len: u32,
    skipped_empty: u64,
    skipped_invalid: u64,
    total_entries: u64,
}

impl<'de> Visitor<'de> for &mut HashExtractor {
    type Value = ();

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a JSON array of cache entries")
    }

    fn visit_seq<A: SeqAccess<'de>>(mut self, mut seq: A) -> Result<(), A::Error> {
        while let Some(entry) = seq.next_element::<CacheEntry>()? {
            self.total_entries += 1;

            // 过滤无效哈希（与 Czkawka Invalid Hash Filtering 规范一致）
            if entry.hash.is_empty() {
                self.skipped_empty += 1;
                continue;
            }
            if entry.hash.iter().all(|&b| b == 0) || entry.hash.iter().all(|&b| b == 0xFF) {
                self.skipped_invalid += 1;
                continue;
            }

            if self.hash_bytes_len == 0 {
                self.hash_bytes_len = entry.hash.len() as u32;
                println!(
                    "检测到哈希长度: {} 字节 ({}-bit)",
                    self.hash_bytes_len,
                    self.hash_bytes_len * 8
                );
            }

            self.out_file.write_all(&entry.hash).expect("写哈希数据失败");
            self.count += 1;

            if self.count % 100000 == 0 {
                println!(
                    "已处理 {} 条有效哈希 (总条目: {})...",
                    self.count, self.total_entries
                );
            }
        }
        Ok(())
    }
}

fn main() {
    let json_path = "tests/data/cache_similar_images_32_Gradient_Gaussian_100.json";
    let output_dir = "tests/data/hash";
    let output_path = format!("{}/cache_32_Gradient_Gaussian_1024bit.bin", output_dir);

    fs::create_dir_all(output_dir).expect("创建输出目录失败");

    println!("读取缓存文件: {}", json_path);

    let file_size = fs::metadata(json_path).map(|m| m.len()).unwrap_or(0);
    println!("文件大小: {:.2} GB", file_size as f64 / (1024.0 * 1024.0 * 1024.0));

    let file = fs::File::open(json_path).unwrap_or_else(|e| panic!("打开缓存 JSON 失败: {}", e));
    let mut out_file = BufWriter::new(
        fs::File::create(&output_path)
            .unwrap_or_else(|e| panic!("创建输出文件失败 {}: {}", output_path, e)),
    );

    // 占位头部，后面回填
    out_file.write_all(&0u64.to_le_bytes()).expect("写头部失败");
    out_file.write_all(&0u32.to_le_bytes()).expect("写头部失败");

    // 使用 serde_json 流式反序列化 + Visitor 逐条读取 JSON 数组
    let mut de = serde_json::Deserializer::from_reader(file);
    let mut extractor = HashExtractor {
        out_file,
        count: 0,
        hash_bytes_len: 0,
        skipped_empty: 0,
        skipped_invalid: 0,
        total_entries: 0,
    };

    de.deserialize_any(&mut extractor).expect("JSON 解析失败");

    // 回填头部
    extractor.out_file.seek(std::io::SeekFrom::Start(0)).expect("seek 失败");
    extractor.out_file.write_all(&extractor.count.to_le_bytes()).expect("写数量失败");
    extractor.out_file.write_all(&extractor.hash_bytes_len.to_le_bytes()).expect("写哈希长度失败");

    println!("\n提取完成:");
    println!("  总条目: {}", extractor.total_entries);
    println!("  有效哈希: {} 条", extractor.count);
    println!("  跳过空哈希: {} 条", extractor.skipped_empty);
    println!("  跳过无效哈希(全零/全FF): {} 条", extractor.skipped_invalid);
    println!(
        "  每条哈希: {} 字节 ({}-bit)",
        extractor.hash_bytes_len,
        extractor.hash_bytes_len * 8
    );
    println!("  输出文件: {}", output_path);

    let out_size = fs::metadata(&output_path).map(|m| m.len()).unwrap_or(0);
    println!("  文件大小: {:.2} MB", out_size as f64 / (1024.0 * 1024.0));
}
