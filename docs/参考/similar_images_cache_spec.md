# Similar Images — Cache File Specification

> Source files: `czkawka_core/src/common/cache.rs`, `czkawka_core/src/common/cache/cleaning.rs`,
> `czkawka_core/src/common/config_cache_path.rs`, `czkawka_core/src/tools/similar_images/core.rs`

---

## 1. Cache File Naming

The cache file name encodes the hash parameters to ensure different parameter combinations
use independent caches:

```
cache_similar_images_{hash_size}_{hash_alg}_{image_filter}_{CACHE_IMAGE_VERSION}.bin
```

| Placeholder             | Values                                                       |
|-------------------------|--------------------------------------------------------------|
| `hash_size`             | `8` / `16` / `32` / `64`                                    |
| `hash_alg`              | `Mean` / `Gradient` / `VertGradient` / `DoubleGradient` / `Blockhash` / `Median` |
| `image_filter`          | `Lanczos3` / `Nearest` / `Triangle` / `Gaussian` / `CatmullRom` |
| `CACHE_IMAGE_VERSION`   | `100` (constant in `cache.rs:29`)                            |

Example: `cache_similar_images_8_Gradient_Lanczos3_100.bin`

When `save_also_as_json` is enabled, a `.json` counterpart is also written with the same
base name but `.json` extension (e.g. `cache_similar_images_8_Gradient_Lanczos3_100.json`).

---

## 2. Storage Location

Resolved by `config_cache_path.rs` using `directories_next::ProjectDirs`:

| Platform  | Default path                                                    | Override env var          |
|-----------|-----------------------------------------------------------------|---------------------------|
| Windows   | `C:\Users\{User}\AppData\Roaming\Qarmin\Czkawka\cache\`        | `CZKAWKA_CACHE_PATH`      |
| Linux     | `~/.cache/czkawka/`                                             | `CZKAWKA_CACHE_PATH`      |
| macOS     | `~/Library/Caches/pl.Qarmin.Czkawka/`                           | `CZKAWKA_CACHE_PATH`      |
| Android   | `$DATA_DIR/cache/{cache_name}/`                                 | `CZKAWKA_CACHE_PATH`      |

The config folder (for `cleaning_timestamps.json`) follows the same pattern with
`CZKAWKA_CONFIG_PATH` override.

---

## 3. Serialized Data Structure

### 3.1 Entry Type: `ImagesEntry`

Each cache entry is a serialized `ImagesEntry`:

```rust
pub struct ImagesEntry {
    pub path: PathBuf,           // Canonical file path
    pub size: u64,               // File size in bytes
    pub width: u32,              // Image width (pixels)
    pub height: u32,             // Image height (pixels)
    pub modified_date: u64,      // mtime as seconds since UNIX epoch
    pub hash: Vec<u8>,           // Perceptual hash bytes
    pub difference: u32,         // Hamming distance to parent (0 for group leaders)
}
```

### 3.2 On-Disk Format

The cache stores a **flat `Vec<ImagesEntry>`** (not a map). The path key is recoverable
from `ImagesEntry.path`, so the map structure is reconstructed at load time.

```
┌──────────────────────────────────────────────┐
│  bincode header (varint length prefix)       │
├──────────────────────────────────────────────┤
│  ImagesEntry #0                              │
│  ├─ path:     String (len-prefixed UTF-8)    │
│  ├─ size:     u64 (little-endian)            │
│  ├─ width:    u32 (little-endian)            │
│  ├─ height:   u32 (little-endian)            │
│  ├─ modified_date: u64 (little-endian)       │
│  ├─ hash:     Vec<u8> (len-prefixed bytes)   │
│  └─ difference: u32 (little-endian)          │
├──────────────────────────────────────────────┤
│  ImagesEntry #1                              │
│  ┊                                           │
├──────────────────────────────────────────────┤
│  ...                                         │
└──────────────────────────────────────────────┘
```

### 3.3 Serialization Options

- **Primary format**: `bincode` with `DefaultOptions::new().with_limit(8 GB)`
  - Little-endian, variable-length integer encoding for lengths
  - 8 GB deserialization limit prevents memory exhaustion from corrupt files
- **Optional format**: JSON (`serde_json::to_writer`) — enabled by `save_also_as_json`
  - Human-readable, useful for debugging; not used for loading unless `.bin` is absent

---

## 4. Cache Loading Flow

```
hash_images()
  │
  └─ hash_images_load_cache()
       │
       └─ load_and_split_cache_generalized_by_path()
            │
            ├─ 1. load_cache_from_file_generalized_by_path()
            │     ├─ open_cache_folder() — locate .bin (priority) or .json
            │     ├─ Deserialize Vec<ImagesEntry> from bincode or JSON
            │     ├─ Validate each entry against current scan files:
            │     │     ├─ path exists in current scan list?
            │     │     ├─ size matches?
            │     │     └─ modified_date matches?
            │     │   (any mismatch → entry is invalid, removed)
            │     ├─ If delete_outdated_cache && should_clean_cache:
            │     │     └─ Also check physical file existence (rayon parallel)
            │     └─ Convert Vec<ImagesEntry> → BTreeMap<String, ImagesEntry>
            │
            └─ 2. extract_loaded_cache()
                  ├─ records_already_cached:  entries found in cache AND current scan
                  │   (reuse cached hash — no recomputation needed)
                  └─ non_cached_files_to_check:  entries in current scan but NOT in cache
                      (need to compute hash)
```

### 4.1 Cache Hit Criteria

A cache entry is considered a **hit** for a given file when **all three** match:

| Field           | Comparison                                    |
|-----------------|-----------------------------------------------|
| `path`          | Exact string match (canonical path)           |
| `size`          | Exact `u64` equality                          |
| `modified_date` | Exact `u64` equality (seconds since epoch)    |

If `size` or `modified_date` differ, the cached hash is **stale** and the entry
is excluded from `records_already_cached`. The file will be re-hashed.

### 4.2 Return Values

`load_and_split_cache_generalized_by_path` returns three maps:

| Return value               | Content                                                    |
|----------------------------|------------------------------------------------------------|
| `loaded_hash_map`          | All valid entries from the cache file (for later merging)  |
| `records_already_cached`   | Current-scan files that hit the cache (hash reused)        |
| `non_cached_files_to_check`| Current-scan files that missed the cache (need hashing)    |

---

## 5. Cache Saving Flow

```
hash_images()
  │
  ├─ ... compute hashes for non_cached_files_to_check ...
  │
  └─ save_to_cache()
       │
       └─ save_and_connect_cache_generalized_by_path()
            │
            ├─ 1. Merge new results + loaded_hash_map into single BTreeMap
            │     (key = path string, value = ImagesEntry)
            │     — New entries overwrite old ones for the same path
            │
            └─ 2. save_cache_to_file_generalized()
                  ├─ Filter: only entries with size >= minimum_file_size
                  ├─ Collect values → Vec<&ImagesEntry>
                  ├─ Write .bin via bincode (BufWriter)
                  └─ If save_also_as_json: write .json via serde_json
```

### 5.1 Merge Semantics

- **Newly computed entries** take priority over stale cache entries for the same path.
- **Stale cache entries** (path not in current scan) are preserved — they may be
  valid for future scans when the file hasn't changed.
- **Minimum file size filter** is applied at save time only; entries below the
  threshold are silently dropped from the cache file.

---

## 6. Cache Cleaning

### 6.1 Automatic Cleaning (During Load)

When `delete_outdated_cache = true` **and** the cleaning interval has elapsed:

1. `should_clean_cache()` checks `cleaning_timestamps.json` for the last cleaning time.
2. Default interval: **7 days** (`7 * 24 * 60 * 60` seconds).
   - Configurable via `CZKAWKA_CACHE_CLEANING_INTERVAL_SECONDS` env var.
3. If the interval has passed, entries whose **physical file no longer exists**
   are removed from the loaded cache (rayon parallel check).
4. `update_cleaning_timestamp()` records the current time.

### 6.2 Manual Cleaning (`clean_all_cache_files`)

Iterates all cache files in the cache directory:

1. Identify cache type by filename pattern (e.g. `cache_similar_images_*` → `SimilarImages`).
2. Deserialize the entire file.
3. For each entry, check:
   - File exists on disk
   - `fs::metadata().len()` matches `entry.size`
   - `fs::metadata().modified()` matches `entry.modified_date`
4. Remove invalid entries (rayon parallel).
5. Write back using **atomic rename**:
   - Serialize to `cache_file.tmp`
   - `fs::rename("cache_file.tmp", "cache_file")` — atomic on most filesystems

### 6.3 Cleaning Timestamps File

Location: `{config_folder}/cleaning_timestamps.json`

```json
{
  "timestamps": [
    {
      "cache_file_name": "cache_similar_images_8_Gradient_Lanczos3_100.bin",
      "last_cleaned_timestamp": 1718000000
    }
  ]
}
```

---

## 7. File I/O Atomicity

| Operation   | Strategy                                                        |
|-------------|-----------------------------------------------------------------|
| **Save**    | `OpenOptions::new().truncate(true).write(true).create(true)` — overwrites in place |
| **Clean**   | Write to `.tmp` file → `fs::rename(.tmp, original)` — atomic replace |

> Note: The save path does **not** use atomic rename; it truncates and writes directly.
> A crash during save may corrupt the cache file. The cleaning path uses the safer
> atomic rename strategy.

---

## 8. Error Handling

| Scenario                          | Behavior                                                    |
|-----------------------------------|-------------------------------------------------------------|
| Cache file not found              | Returns `None` → all files treated as non-cached            |
| Cache file corrupt / deserialization fails | Warning logged, returns `None` → all files re-hashed |
| Cannot create cache directory     | Warning logged, cache disabled for this run                 |
| Cannot write cache file           | Warning logged, scan continues without cache persistence    |
| .bin missing but .json exists     | Falls back to JSON deserialization                          |

---

## 9. Version Compatibility

`CACHE_IMAGE_VERSION = 100` is embedded in the filename. When the `ImagesEntry`
struct changes in an incompatible way, this version number must be incremented,
causing old cache files to be ignored (they won't match the new filename pattern).

Existing cache files with old version numbers remain on disk until manually deleted
or cleaned by the cache cleaning mechanism (which will fail to parse them and skip).
