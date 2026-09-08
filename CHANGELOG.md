# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-09-06

### Breaking
- **`isotropic_remesh` signature**: Added a fifth parameter `preserve_boundary: bool`. When `true` (default in `quick_remesh`/`remesh_to_length`), split skips boundary edges and collapse skips edges touching boundary vertices, preserving open-mesh boundaries.

### Fixed
- **`flip_edge` non-manifold guard**: `flip_edge` now rejects flips where the opposite vertices `c` and `d` are already connected (`FlipCreatesNonManifoldEdge`), which would otherwise create a duplicate edge.

### Added
- **`reproject` implementation**: `isotropic_remesh(..., reproject=true)` now snapshots the original surface and projects each vertex back to its closest point on that surface (point-to-triangle closest point, brute-force `O(V·F)`), instead of being a no-op.
- Regression tests: `flip_edge_rejects_when_opposite_vertices_connected`, `collapse_interior_edge_with_one_boundary_vertex`, `remesh_preserve_boundary_false_allows_boundary_split`, `remesh_reproject_keeps_vertices_on_unit_sphere`.

### Changed
- `docs/remesh.tex` and `docs/topology_ops.tex` updated with the new API, `reproject`, `preserve_boundary`, and the `FlipCreatesNonManifoldEdge` error.

## [0.2.6] - 2026-09-06

### Fixed
- **`split_edge` boundary loop**: Splitting a boundary edge did not maintain the boundary ring's `next`/`prev` chain, so `isotropic_remesh` split of a long boundary edge broke the topology. Now the boundary half-edges and their neighbors are relinked.

### Added
- Regression test `split_boundary_edge_maintains_boundary_loop`.

### Changed
- `docs/topology_ops.tex` updated with the boundary-loop maintenance.

## [0.2.5] - 2026-09-06

### Fixed
- **`collapse_edge` boundary-vertex guard**: `collect_collapse_data` now rejects collapsing an edge whose two endpoints are both boundary vertices (`CollapseOnBoundaryVertices`), which would otherwise merge two boundary vertices and break boundary topology.

### Added
- Regression test `collapse_with_both_boundary_vertices_fails`.

### Changed
- `docs/topology_ops.tex` updated with the boundary-vertex guard.

## [0.2.4] - 2026-09-06

### Fixed
- **`isotropic_remesh` flip validity**: Before flipping edge `(a,b)` to `(c,d)`, `flip_for_valence` now checks whether `(c,d)` is already connected; if so it skips the flip. Previously this could create a non-manifold edge shared by 4 faces.

### Added
- Regression test `are_vertices_connected_basic`.

### Changed
- `docs/remesh.tex` updated with the flip validity check.

## [0.2.3] - 2026-09-06

### Fixed
- **`isotropic_remesh` non-manifold guard coverage**: Previously only `collapse_short_edges` skipped non-manifold vertices (detected once). `split_long_edges` and `flip_for_valence` now skip non-manifold vertices too, non-manifold vertices are re-detected each iteration, and `collapse_short_edges` re-checks merged vertices so non-manifold vertices produced mid-collapse are also skipped.

### Added
- Regression test `remesh_split_flip_skip_nonmanifold_vertices` covering the split + collapse + flip path on a pinch mesh.

### Changed
- `docs/remesh.tex` updated with the expanded non-manifold guard.

## [0.2.2] - 2026-09-06

### Fixed
- **`isotropic_remesh` collapse on non-manifold vertices**: `collapse_short_edges` collapsed edges whose endpoints are non-manifold (e.g. a pinch vertex shared by two fans). `VertexRing` only walks one fan there, so the collapse link condition passed incorrectly and left dangling half-edge references, later panicking in `flip_edge` or producing non-manifold edges. Non-manifold vertices are now detected (ring length ≠ true incoming half-edge count) and collapses touching them are skipped.

### Added
- Regression tests `nonmanifold_vertices_detects_pinch` and `remesh_collapse_skips_nonmanifold_endpoints`.

### Changed
- `docs/remesh.tex` updated with the non-manifold collapse guard.

## [0.2.1] - 2026-09-06

### Fixed
- **`mixed_area_at_vertex` origin/center confusion**: The Voronoi/mixed area for discrete curvature incorrectly used `h.twin.vertex` (the center vertex `v` itself) as the second triangle neighbor, degenerating every incident triangle (`|pb - pv|² = 0`) and forcing `mixed_area_at_vertex` to always return the `1e-14` clamp. Gaussian curvature therefore returned `≈ 2π/1e-14`. The second neighbor is now `h.next.vertex` (the tip of the next halfedge in the face), with the correct cotangent weights.
- **`gaussian_curvature` angle sum**: The same `h.twin.vertex` misuse made the per-vertex angle sum always zero (the zero-length vector was skipped). Now uses `h.next.vertex`.
- **`mean_curvature` factor-of-2**: The implementation returned `|Δv| / (2·A_mixed)` while the documented formula is `|Δv| / (4·A_mixed)`; corrected so the unit-sphere mean curvature is `≈ 1` instead of `≈ 2`.

### Added
- Regression tests `gaussian_curvature_unit_sphere`, `mean_curvature_unit_sphere`, `principal_curvatures_unit_sphere`, asserting unit-sphere curvatures ≈ 1.

### Changed
- `docs/geometry.tex` discrete-curvature section updated to reflect the corrected mixed-area and mean-curvature formulas.

## [0.2.0] - 2026-07-07

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

## [0.1.0] - 2026-07-05

Initial pre-release (minimal surface). See [0.2.0] for the full feature set.

[0.3.0]: https://github.com/nil-is-lin/halfedge/compare/v0.2.6...v0.3.0
[0.2.6]: https://github.com/nil-is-lin/halfedge/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/nil-is-lin/halfedge/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/nil-is-lin/halfedge/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/nil-is-lin/halfedge/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/nil-is-lin/halfedge/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/nil-is-lin/halfedge/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/nil-is-lin/halfedge/compare/v0.1.0...v0.2.0
[Unreleased]: https://github.com/nil-is-lin/halfedge/compare/v0.3.0...HEAD
