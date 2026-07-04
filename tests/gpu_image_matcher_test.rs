use gpgpu_tool::tasks::phasher::HashAlgorithm;
use gpgpu_tool::{GpuContext, GpuImageMatcher};

/// 创建均匀灰度图像
fn make_uniform_image(width: u32, height: u32, value: u8) -> Vec<u8> {
    vec![value; (width * height) as usize]
}

/// Mean Hash 默认目标尺寸 8x8
const TARGET_SIZE: (u32, u32) = (8, 8);

#[test]
fn test_identical_images_distance_zero() {
    // 验证 GpuImageMatcher 内部使用的哈希 + 距离矩阵管线正确性：
    // 将同一图像重复放入 database 批次，所有条目哈希应一致（单批次内确定性），
    // 从而距离矩阵对角线为 0。
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean)
        .expect("GpuImageMatcher 创建失败");

    let img = make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, 128);

    // 同一图像重复 3 次，在同一批次中计算
    let images = vec![img.clone(), img.clone(), img];
    let all_dims = vec![TARGET_SIZE; 3];

    let matrix = matcher
        .compute_distance_matrix(&ctx, &images, &all_dims, &images, &all_dims)
        .expect("距离矩阵计算失败");

    assert_eq!(matrix.len(), 3, "应有 3 行");
    assert_eq!(matrix[0].len(), 3, "应有 3 列");

    // 对角线应为 0（同批次内相同图像哈希一致）
    for (i, row) in matrix.iter().enumerate() {
        assert_eq!(row[i], 0, "对角线 [{}][{}] 应为 0", i, i);
    }
}

/// 创建渐变灰度图像
fn make_gradient_image(width: u32, height: u32) -> Vec<u8> {
    (0..(width * height))
        .map(|i| (i % 256) as u8)
        .collect()
}

#[test]
fn test_different_images_have_distance() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean)
        .expect("GpuImageMatcher 创建失败");

    // 使用渐变图像 vs 反向渐变图像，确保 Mean Hash 产生不同结果
    // 注意：均匀图像的 Mean Hash 总是全 1（所有像素 >= 均值），
    // 所以两个均匀图像的距离为 0，不适合测试"不同图像距离 > 0"。
    let img1 = make_gradient_image(TARGET_SIZE.0, TARGET_SIZE.1);
    let img2: Vec<u8> = img1.iter().map(|&p| 255 - p).collect();
    let dims = vec![TARGET_SIZE];

    let matrix = matcher
        .compute_distance_matrix(&ctx, &[img1], &dims, &[img2], &dims)
        .expect("距离矩阵计算失败");

    assert_eq!(matrix.len(), 1);
    assert_eq!(matrix[0].len(), 1);
    assert!(
        matrix[0][0] > 0,
        "不同图像距离应 > 0，实际为 {}",
        matrix[0][0]
    );
}

#[test]
fn test_find_similar_returns_matches() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean)
        .expect("GpuImageMatcher 创建失败");

    // 使用目标尺寸，跳过 resize
    let query_img = make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, 100);
    let similar_img = make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, 102); // 非常相似
    let different_img = make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, 255); // 完全不同
    let q_dims = vec![TARGET_SIZE];
    let db_dims = vec![TARGET_SIZE; 2];

    let results = matcher
        .find_similar(
            &ctx,
            &[query_img],
            &q_dims,
            &[similar_img, different_img],
            &db_dims,
            10, // 阈值
        )
        .expect("find_similar 失败");

    assert_eq!(results.len(), 1, "应有 1 个 query 的结果");
    assert!(
        !results[0].is_empty(),
        "至少应找到 1 个匹配（相似图像）"
    );
}

#[test]
fn test_distance_matrix_shape() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean)
        .expect("GpuImageMatcher 创建失败");

    // 使用目标尺寸，跳过 resize
    let img = make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, 128);

    // 3 queries × 5 database
    let queries = vec![img.clone(), img.clone(), img.clone()];
    let database = vec![
        img.clone(),
        img.clone(),
        img.clone(),
        img.clone(),
        img,
    ];
    let q_dims = vec![TARGET_SIZE; 3];
    let db_dims = vec![TARGET_SIZE; 5];

    let matrix = matcher
        .compute_distance_matrix(&ctx, &queries, &q_dims, &database, &db_dims)
        .expect("距离矩阵计算失败");

    assert_eq!(matrix.len(), 3, "矩阵应有 3 行");
    for (i, row) in matrix.iter().enumerate() {
        assert_eq!(row.len(), 5, "第 {} 行应有 5 列", i);
    }
}

#[test]
fn test_batch_scaling() {
    let mut ctx = GpuContext::new_sync().expect("GPU 初始化失败");
    let matcher = GpuImageMatcher::new(&mut ctx, HashAlgorithm::Mean)
        .expect("GpuImageMatcher 创建失败");

    // 12 个 database 图像（使用不同灰度值区分）
    let database: Vec<Vec<u8>> = (0..12)
        .map(|i| make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, (i * 20) as u8))
        .collect();
    let db_dims: Vec<(u32, u32)> = vec![TARGET_SIZE; 12];

    // 4 个 query 图像
    let queries: Vec<Vec<u8>> = (0..4)
        .map(|i| make_uniform_image(TARGET_SIZE.0, TARGET_SIZE.1, (i * 60) as u8))
        .collect();
    let q_dims: Vec<(u32, u32)> = vec![TARGET_SIZE; 4];

    // 测试 compute_distance_matrix
    let matrix = matcher
        .compute_distance_matrix(&ctx, &queries, &q_dims, &database, &db_dims)
        .expect("大批量距离矩阵计算失败");

    assert_eq!(matrix.len(), 4, "应有 4 行");
    for (i, row) in matrix.iter().enumerate() {
        assert_eq!(row.len(), 12, "第 {} 行应有 12 列", i);
    }

    // 测试 find_similar
    let results = matcher
        .find_similar(&ctx, &queries, &q_dims, &database, &db_dims, 30)
        .expect("大批量 find_similar 失败");

    assert_eq!(results.len(), 4, "应有 4 个 query 的结果");

    // 验证结果结构正确
    for (qi, matches) in results.iter().enumerate() {
        for m in matches {
            assert!(
                m.distance <= 30,
                "query[{}] 的匹配距离 {} 应 ≤ threshold",
                qi,
                m.distance
            );
        }
    }
}
