# Similar Images — Hash Computation & Distance Comparison

> Source files: `czkawka_core/src/tools/similar_images/mod.rs`,
> `czkawka_core/src/tools/similar_images/core.rs`,
> `czkawka_core/src/common/image.rs`

---

## 1. Overview

The similar images feature uses **perceptual hashing** to produce a compact fingerprint
for each image, then groups images whose fingerprints are within a configurable
**Hamming distance** threshold. The pipeline is:

```
Image file → Decode → Resize → Hash → Vec<u8> fingerprint
                                              │
                     BK-Tree search ←─────────┘
                          │
                     Group similar images
```

---

## 2. Hash Parameters

### 2.1 Hash Algorithm (`HashAlg`)

Provided by the `image_hasher` crate (v3.0). Six algorithms are available:

| Algorithm         | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `Mean`            | Each pixel compared to the global mean; 1 if above, 0 if below   |
| `Gradient`        | Horizontal gradient: compare each pixel to its right neighbor     |
| `VertGradient`    | Vertical gradient: compare each pixel to its bottom neighbor      |
| `DoubleGradient`  | Both horizontal and vertical gradients combined                   |
| `Blockhash`       | Divide image into blocks; compare mean of each block              |
| `Median`          | Each pixel compared to the global median                          |

### 2.2 Hash Size

Controls the resolution of the hash grid. Must be one of: **8, 16, 32, 64**.

| hash_size | Hash grid   | Output bytes | Output bits |
|-----------|-------------|-------------|-------------|
| 8         | 8×8         | 8           | 64          |
| 16        | 16×16       | 32          | 256         |
| 32        | 32×32       | 128         | 1,024       |
| 64        | 64×64       | 512         | 4,096       |

Larger hash sizes capture more detail but increase computation and comparison cost.

### 2.3 Resize Filter (`FilterType`)

Before hashing, the image is resized to `hash_size × hash_size` pixels. The filter
controls the quality of this downscale:

| Filter       | Character                          |
|--------------|------------------------------------|
| `Lanczos3`   | High quality, slow (default)       |
| `CatmullRom` | High quality, slightly faster      |
| `Triangle`   | Medium quality, moderate speed     |
| `Gaussian`   | Smooth, moderate speed             |
| `Nearest`    | Fastest, lowest quality            |

---

## 3. Hash Computation Pipeline

### 3.1 CPU Path (Default)

Entry point: `collect_image_file_entry()` in `core.rs:331`

```
1. get_dynamic_image_from_path(path)
   ├─ Detect format by extension
   ├─ Standard image → image::ImageReader::decode()
   ├─ RAW image (if feature enabled) → rawler / libraw-rs
   ├─ Read EXIF orientation tag
   └─ Auto-rotate image per EXIF

2. img.dimensions() → (width, height)

3. Build hasher:
   HasherConfig::new()
       .hash_size(hash_size as u32, hash_size as u32)
       .hash_alg(hash_alg)
       .resize_filter(image_filter)
       .to_hasher()

4. hasher.hash_image(&img) → HashMat
   └─ .as_bytes().to_vec() → Vec<u8>  (the ImHash)
```

Parallelism: `hash_images_cpu()` uses `rayon::into_par_iter()` to process all
non-cached files concurrently.

### 3.2 GPU Path (Optional, requires `gpu` feature)

Entry point: `hash_images_gpu()` in `core.rs:196`

```
Phase 1 — Parallel decode (rayon scope):
   For each non-cached file:
     ├─ get_dynamic_image_from_path() → decode to RGBA8888
     ├─ Group by (width, height) — same-size images batched together
     └─ Store (ImagesEntry, RGBA pixels) per group

Phase 2 — GPU batch hash:
   For each resolution group:
     ├─ Convert RGBA → grayscale: (R*77 + G*150 + B*29) >> 8
     ├─ wgpu_compute_engine::PerceptualHasher::compute()
     └─ On GPU failure → fallback to CPU path for that group
```

### 3.3 Invalid Hash Filtering

After hash computation, entries with the following hash values are **excluded**
from similarity comparison (but are still saved to cache):

| Condition                  | Meaning                                            |
|----------------------------|----------------------------------------------------|
| `hash.is_empty()`          | Hash computation failed entirely                   |
| All bytes are `0x00`       | Algorithm could not extract features (e.g. mostly transparent) |
| All bytes are `0xFF`       | Algorithm saturated (e.g. solid white image)       |

---

## 4. Distance Metric: Hamming Distance

### 4.1 Definition

The distance between two hashes is the **bitwise Hamming distance**: the count of
differing bits when the two byte arrays are XORed.

```rust
struct Hamming;
impl bk_tree::Metric<ImHash> for Hamming {
    fn distance(&self, a: &ImHash, b: &ImHash) -> u32 {
        hamming_bitwise_fast(a, b)
    }
}
```

Implementation: `hamming-bitwise-fast` crate — XORs corresponding bytes, then
counts set bits (popcount) using hardware instructions where available.

### 4.2 Examples

| Hash A (last byte) | Hash B (last byte) | XOR   | Popcount | Distance |
|--------------------|--------------------|-------|----------|----------|
| `0b00000001`       | `0b00000010`       | `0b11`| 2        | 2        |
| `0b00000001`       | `0b00000011`       | `0b10`| 1        | 1        |
| `0b00000000`       | `0b00001000`       | `0b1000`| 1      | 1        |

The distance is computed **per bit**, not per byte. This provides finer granularity
for similarity assessment.

### 4.3 Similarity Threshold Table

`SIMILAR_VALUES` maps each `(hash_size, SimilarityPreset)` pair to a maximum
Hamming distance:

```rust
pub const SIMILAR_VALUES: [[u32; 6]; 4] = [
    //  VeryHigh  High  Medium  Small  VerySmall  Minimal
    [   1,        2,    5,      7,     14,        40    ],  // hash_size = 8
    [   2,        5,    15,     30,    40,        40    ],  // hash_size = 16
    [   4,        10,   20,     40,    40,        40    ],  // hash_size = 32
    [   6,        20,   40,     40,    40,        40    ],  // hash_size = 64
];
```

The `max_difference` parameter (set by the user via similarity preset) defines the
tolerance threshold for the BK-Tree search.

---

## 5. Similarity Search: BK-Tree

### 5.1 Data Structure

A **BK-Tree** (Burkhard-Keller tree) is a tree structure optimized for metric-space
nearest-neighbor queries. It reduces the search from O(n²) to approximately O(n log n).

- Each node stores one `ImHash`.
- Children are organized by their distance to the parent.
- Search: given a query hash and tolerance `t`, only subtrees whose edge distance
  falls within `[d - t, d + t]` of the query are explored.

### 5.2 Search Modes

#### Tolerance = 0 (Exact Match)

Fast path — no BK-Tree needed. Simply iterate `image_hashes` and collect entries
where `Vec<ImagesEntry>.len() >= 2` (same hash, multiple files).

#### Tolerance > 0 (Similar Match)

```
Step 1: split_hashes()
   ├─ Identify "multi-image hashes" (same hash, ≥2 images)
   ├─ Reference folder mode:
   │     ├─ Reference folder hashes → base_hashes (query side)
   │     └─ Normal folder hashes → inserted into BK-Tree (database side)
   └─ Normal mode:
         ├─ All hashes → inserted into BK-Tree
         └─ All hashes → base_hashes (query side)

Step 2: Chunked parallel search (chunks of 1000 base_hashes)
   For each base_hash:
     ├─ bktree.find(hash, tolerance) → iterator of (distance, &hash)
     ├─ Filter: distance != 0 (skip self)
     ├─ Filter: not already a parent hash
     ├─ Filter: not a multi-image hash
     ├─ Filter: distance < existing best distance for this child
     └─ Sort results by distance ascending

Step 3: connect_results_simplified()
   ├─ Flatten + sort all partial results by distance
   ├─ Greedy assignment:
   │     For each (parent, (distance, child)):
   │       ├─ If child is already a parent → skip
   │       ├─ If child has a closer parent already → skip
   │       ├─ If parent itself has a closer parent → reparent:
   │       │     ├─ Remove parent from its old parent's children count
   │       │     ├─ If old parent has no children and is not multi-image → remove
   │       │     └─ Make parent a new parent with 1 child
   │       └─ Record: child → (parent, distance)
   └─ Result: hashes_parents (parent → child_count)
             hashes_similarity (child → (parent, distance))

Step 4: collect_hash_compare_result()
   ├─ For each parent with children > 0 OR multi-image:
   │     Collect all ImagesEntry for parent hash
   └─ For each child:
         Collect all ImagesEntry, set .difference = distance,
         append to parent's group
```

### 5.3 Reparenting Logic

The `connect_results_simplified` function handles the case where a hash that was
initially assigned as a child later turns out to be closer to a different parent:

```
Before reparenting:          After reparenting:
  P1 ← C1 (dist=5)            P1 (no children, removed)
  P1 ← C2 (dist=3)            C2 ← C1 (dist=5)  [C2 promoted to parent]
                                C2 ← P1? (only if P1 is closer to C2 than its current parent)
```

Rules:
- A hash that is already a **parent** (has children) is never reparented — too complex.
- A hash can only be reparented if the new distance is **strictly less** than the
  current distance to its parent.
- Multi-image hashes are always kept as parents (they have ≥2 images with the same hash).

---

## 6. GPU Comparison Path

When `use_gpu = true` and the `gpu` feature is enabled:

### 6.1 Symmetric Mode (no reference folders)

```
gpu_compare_hashes_auto(&base_hashes, tolerance)
  → Vec<(parent_idx, child_idx, distance)>
```

All hashes are both queries and database entries.

### 6.2 Asymmetric Mode (reference folders)

```
gpu_compare_hashes_asymmetric(&ref_hashes, &normal_hashes, tolerance)
  → Vec<(parent_idx, child_idx, distance)>
```

Reference hashes are queries; normal hashes are the database.

### 6.3 Result Processing

GPU matches are converted to the same `partial_results` format used by the CPU path
(via `build_partial_results()`), then fed into `connect_results_simplified()` for
grouping. This ensures identical grouping logic regardless of CPU/GPU path.

---

## 7. Post-Processing

After similarity groups are formed:

### 7.1 Exclude Same Size

If `exclude_images_with_same_size = true`:
- Within each group, only keep the first image per unique `size`.
- Groups reduced to ≤1 image are discarded.

### 7.2 Exclude Same Resolution

If `exclude_images_with_same_resolution = true`:
- Within each group, only keep the first image per unique `(width, height)`.
- Groups reduced to ≤1 image are discarded.

### 7.3 Reference Folder Filtering

If `use_reference_folders = true`:
- Convert `Vec<Vec<ImagesEntry>>` → `Vec<(ImagesEntry, Vec<ImagesEntry>)>`
- Each tuple: (reference image, similar non-reference images)
- Groups where no reference image exists are discarded.

---

## 8. Complete Flow Diagram

```
search()
 │
 ├─ check_for_similar_images()
 │    └─ DirTraversal → images_to_check: BTreeMap<Path, ImagesEntry>
 │
 ├─ hash_images()
 │    ├─ load_cache() → split into cached / non-cached
 │    ├─ hash_images_cpu() or hash_images_gpu()
 │    │    └─ For each non-cached file:
 │    │         decode → resize → perceptual hash → Vec<u8>
 │    ├─ save_cache() → merge new + old entries → write .bin (and .json)
 │    └─ Filter invalid hashes → image_hashes: IndexMap<ImHash, Vec<ImagesEntry>>
 │
 ├─ find_similar_hashes()
 │    ├─ tolerance=0 → exact match (same hash, ≥2 images)
 │    ├─ tolerance>0:
 │    │    ├─ split_hashes() → build BK-Tree
 │    │    ├─ Chunked parallel BK-Tree search (1000 per chunk)
 │    │    ├─ connect_results_simplified() → greedy grouping + reparenting
 │    │    └─ collect_hash_compare_result() → final groups
 │    ├─ exclude same size / same resolution
 │    └─ reference folder filtering
 │
 └─ delete_files()  (optional)
```

---

## 9. Key Data Structures Summary

```rust
type ImHash = Vec<u8>;

struct SimilarImages {
    common_data: CommonToolData,
    bktree: BKTree<ImHash, Hamming>,              // BK-Tree for fast nearest-neighbor
    image_hashes: IndexMap<ImHash, Vec<ImagesEntry>>,  // hash → all images with that hash
    images_to_check: BTreeMap<String, ImagesEntry>,    // path → entry (pre-hash)
    similar_vectors: Vec<Vec<ImagesEntry>>,             // final result groups
    similar_referenced_vectors: Vec<(ImagesEntry, Vec<ImagesEntry>)>,  // reference mode result
    params: SimilarImagesParameters,
}

struct SimilarImagesParameters {
    max_difference: u32,           // Hamming distance tolerance
    hash_size: u8,                 // 8 / 16 / 32 / 64
    hash_alg: HashAlg,             // Perceptual hash algorithm
    image_filter: FilterType,      // Resize filter
    exclude_images_with_same_size: bool,
    exclude_images_with_same_resolution: bool,
    use_gpu: bool,
}
```
