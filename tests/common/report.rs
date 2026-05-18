use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// 测试报告生成器，输出 Markdown 格式报告。
pub struct TestReport {
    pub gpu_info: String,
    pub test_results: Vec<TestResult>,
    pub bench_results: Vec<BenchResult>,
}

/// 单个功能测试结果。
#[derive(Clone)]
pub struct TestResult {
    pub algorithm: String,
    pub size: String,
    pub test_type: String,
    pub passed: bool,
}

/// 单个基准测试结果。
pub struct BenchResult {
    pub algorithm: String,
    pub size: String,
    pub batch_size: usize,
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
}

impl TestReport {
    pub fn new(gpu_info: String) -> Self {
        Self {
            gpu_info,
            test_results: Vec::new(),
            bench_results: Vec::new(),
        }
    }

    pub fn add_test_result(&mut self, algorithm: &str, size: &str, test_type: &str, passed: bool) {
        self.test_results.push(TestResult {
            algorithm: algorithm.to_string(),
            size: size.to_string(),
            test_type: test_type.to_string(),
            passed,
        });
    }

    pub fn add_bench_result(
        &mut self,
        algorithm: &str,
        size: &str,
        batch_size: usize,
        gpu_time_ms: f64,
        cpu_time_ms: f64,
    ) {
        self.bench_results.push(BenchResult {
            algorithm: algorithm.to_string(),
            size: size.to_string(),
            batch_size,
            gpu_time_ms,
            cpu_time_ms,
        });
    }

    /// 生成 Markdown 报告并写入文件。
    pub fn generate(&self, output_path: &str) {
        let mut md = String::new();

        // 标题
        md.push_str("# 感知哈希测试报告\n\n");

        // 1. 测试摘要
        md.push_str("## 1. 测试摘要\n\n");
        md.push_str(&format!("- **测试时间**: {}\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
        md.push_str(&format!("- **GPU 信息**: {}\n", self.gpu_info));
        let total = self.test_results.len();
        let passed = self.test_results.iter().filter(|r| r.passed).count();
        md.push_str(&format!("- **总测试数**: {}\n", total));
        md.push_str(&format!("- **通过率**: {}% ({}/{})\n\n", passed * 100 / total.max(1), passed, total));

        // 2. 功能测试结果
        md.push_str("## 2. 功能测试结果\n\n");
        md.push_str("| 算法 | 尺寸 | 单图像 | 批量 | 空输入 |\n");
        md.push_str("|------|------|--------|------|--------|\n");

        // 按算法分组
        let mut grouped: HashMap<String, Vec<TestResult>> = HashMap::new();
        for result in self.test_results.clone() {
            grouped.entry(result.algorithm.clone()).or_default().push(result);
        }

        for (algorithm, results) in &grouped {
            let sizes = ["8x8", "16x16", "32x32", "9x8", "8x9", "9x9"];
            for size in &sizes {
                let size_results: Vec<_> = results.iter().filter(|r| r.size == *size).collect();
                if size_results.is_empty() {
                    continue;
                }
                let single: Option<&TestResult> = size_results.iter().find(|r| r.test_type == "single").map(|r| *r);
                let batch: Option<&TestResult> = size_results.iter().find(|r| r.test_type == "batch").map(|r| *r);
                let empty: Option<&TestResult> = size_results.iter().find(|r| r.test_type == "empty").map(|r| *r);

                md.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    algorithm,
                    size,
                    format_test_result(single),
                    format_test_result(batch),
                    format_test_result(empty)
                ));
            }
        }
        md.push('\n');

        // 3. 性能基准测试
        md.push_str("## 3. 性能基准测试\n\n");
        md.push_str("| 算法 | 尺寸 | Batch | GPU(ms) | CPU(ms) | 加速比 |\n");
        md.push_str("|------|------|-------|---------|---------|--------|\n");

        for result in &self.bench_results {
            let speedup = if result.gpu_time_ms > 0.0 {
                result.cpu_time_ms / result.gpu_time_ms
            } else {
                0.0
            };
            md.push_str(&format!(
                "| {} | {} | {} | {:.2} | {:.2} | {:.1}x |\n",
                result.algorithm,
                result.size,
                result.batch_size,
                result.gpu_time_ms,
                result.cpu_time_ms,
                speedup
            ));
        }
        md.push('\n');

        // 4. 尺寸影响分析
        md.push_str("## 4. 尺寸影响分析\n\n");
        md.push_str("| 算法 | 小尺寸 | 中尺寸 | 大尺寸 | 趋势 |\n");
        md.push_str("|------|--------|--------|--------|------|\n");

        let mut algo_sizes: HashMap<String, Vec<&BenchResult>> = HashMap::new();
        for result in &self.bench_results {
            algo_sizes.entry(result.algorithm.clone()).or_default().push(result);
        }

        for (algorithm, results) in &algo_sizes {
            let small = results.iter().find(|r| r.size == "8x8" || r.size == "8x9" || r.size == "9x8" || r.size == "9x9" || r.size == "16x16");
            let medium = results.iter().find(|r| r.size == "16x16" || r.size == "16x17" || r.size == "17x16" || r.size == "17x17");
            let large = results.iter().find(|r| r.size == "32x32" || r.size == "32x33" || r.size == "33x32" || r.size == "33x33");

            let trend = if let (Some(s), Some(l)) = (small, large) {
                if l.gpu_time_ms > s.gpu_time_ms * 2.0 {
                    "超线性增长"
                } else if l.gpu_time_ms > s.gpu_time_ms {
                    "线性增长"
                } else {
                    "基本平稳"
                }
            } else {
                "数据不足"
            };

            md.push_str(&format!(
                "| {} | {:.2}ms | {:.2}ms | {:.2}ms | {} |\n",
                algorithm,
                small.map(|r| r.gpu_time_ms).unwrap_or(0.0),
                medium.map(|r| r.gpu_time_ms).unwrap_or(0.0),
                large.map(|r| r.gpu_time_ms).unwrap_or(0.0),
                trend
            ));
        }

        // 写入文件
        let path = Path::new(output_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(path, md).expect("写入报告失败");
    }
}

fn format_test_result(result: Option<&TestResult>) -> String {
    match result {
        Some(r) if r.passed => "✅".to_string(),
        Some(_) => "❌".to_string(),
        None => "-".to_string(),
    }
}
