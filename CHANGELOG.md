# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MSRV metadata**: `rust-version = "1.87"` declared in `Cargo.toml` (`is_multiple_of` used in `io/gltf.rs` requires >= 1.87).
- **Serde support**: Optional `serde` feature flag enables `Serialize`/`Deserialize` for `MeshStorage`, `Vertex`, `HalfEdge`, `Face`, and `EdgeId` (via `slotmap/serde`). Roundtrip tests verify topological integrity.
- **MeshCache layer** (`cache` module): Lazy caching for `face_normal`, `face_area`, `vertex_normal`, `vertex_valence`, and `edge_length`. Supports full and granular invalidation.
- **Conjugate Gradient with Jacobi preconditioner** (`linalg::conjugate_gradient_preconditioned`): Dramatically faster convergence for large sparse systems. Original `conjugate_gradient` now delegates to PCG with identity preconditioner.
- **`Scalar` type alias**: `pub type Scalar = f64` and `pub type Vec3 = [Scalar; 3]` for configurable floating-point type. All public API floating-point parameters now use this alias.
- **Rayon parallelism extensions**:
  - `remesh`: Parallel vertex smoothing (gather-scatter pattern)
  - `decimate`: Parallel QEM quadric initialization and edge cost computation
  - `boolean`: Parallel face processing (edge-triangle intersection, splitting, classification)
  - `geodesics`: Parallel face gradient computation, edge length reduction, gradient normalization
  - `parameterization`: Parallel MVC per-vertex weight computation
  - `deformation`: Parallel ARAP rotation estimation, RHS assembly, Laplacian delta
- **Robust predicates in `decimate`**: `face_plane` now uses Shewchuk adaptive precision for degenerate triangle detection. New `would_collapse_create_degenerate` prevents collapsing edges that would produce degenerate triangles.
- **OFF format support**: `load_off` / `parse_off` / `save_off` / `format_off` (ASCII).
- **glTF/GLB format support**: `load_glb` / `parse_glb` / `save_glb` / `format_glb` (GLB binary, minimal subset).
- **Unified I/O entry points**: `load_mesh` / `save_mesh` dispatch by file extension (`.obj`, `.ply`, `.stl`, `.off`, `.glb`).
- **`MeshBuildError` error type**: Proper `Result`-based error handling for mesh building functions (replaces previous `panic!()`).
- **Property system tests**: 22 new tests for `property.rs` (17 → 39).
- **Storage tests**: 22 new tests for `storage.rs` (21 → 43).

### Changed
- **`io.rs` split into submodules**: 3216-line monolith refactored into `src/io/` directory with 7 files: `mod.rs`, `builder.rs`, `obj.rs`, `ply.rs`, `stl.rs`, `off.rs`, `gltf.rs`.
- **`topology_ops.rs` function splitting**: `extrude_region`, `split_edge`, `extrude_face`, `collapse_edge_impl`, `decimate_qem` extracted into sub-functions and helper structs (`CollapseInfo`, `RegionTopology`, etc.) for readability.
- **`PropertyStore<T>`**: `HashMap<usize, T>` → `Vec<Option<T>>` for O(1) direct indexing via slotmap key 32-bit index. Continuous memory layout improves cache locality.
- **Error handling**: 112 `unwrap()` calls in non-test code replaced with `expect("descriptive reason")`.
- **Logging**: 17 `eprintln!` / `println!` calls in library code replaced with `log::warn!` (requires `log` dependency).
- **Cotangent Laplacian deduplication**: Three redundant `build_full_cotan_laplacian` implementations consolidated into `linalg::build_cotan_laplacian` and `linalg::build_vertex_index` public functions.
- **`#[doc(hidden)]` for internal modules**: `direction_field`, `export`, `intrinsic`, `query`, `test_util`, `triangulation` no longer clutter public API documentation.
- **`Vertex` position type**: Now `Vec3` (`[Scalar; 3]`) instead of `[f64; 3]`.

### Fixed
- **O(n²) performance bug in `conformal.rs`**: Replaced `mesh.vertex_ids().nth(idx).unwrap()` with direct `v_idx.iter()` traversal.
- **`panic!()` in `io.rs`**: `build_mesh_from_vertices_and_faces` and `build_mesh_from_polygons` now return `Result<MeshStorage, MeshBuildError>`. All ~70+ call sites updated.
- **Clippy warnings**: Fixed `boolean.rs` (`manual_clamp`) and `bvh.rs` (`collapsible_if`). Zero warnings across all feature combinations.
- **Examples compilation**: All 15 examples updated for `build_mesh_from_vertices_and_faces` `Result` return type.

### Deprecated
- Nothing.

### Removed
- Nothing.

### Security
- Nothing.

## [0.1.0] - Unreleased

Initial pre-release. See [Unreleased] for full feature set.

[Unreleased]: https://github.com/nil-is-lin/halfedge/compare/v0.1.0...HEAD
