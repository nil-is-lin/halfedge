# halfedge

A half-edge mesh data structure library for Rust, providing comprehensive tools for 3D mesh processing: traversal, topology operations, geometry, subdivision, decimation, parameterization, geodesics, deformation, boolean operations, and more.

## Features

- **Half-edge data structure** with slotmap-based stable handles (`VertexId` / `HalfEdgeId` / `FaceId` / `EdgeId`)
- **Traversal**: eager & lazy iterators (zero-allocation), boundary loops, k-ring neighborhoods, undirected edges
- **Query DSL**: chainable builder (`v.halfedge_to(w).cw_rotated().dst_vert().run(&mesh)`)
- **Topology operations**: edge split / flip / collapse / extrude / poke, `add_triangle` builder
- **Geometry**: edge lengths, face areas/normals, cotan Laplacian, curvatures (Gaussian / mean / principal), dihedral angles, point-triangle distance, AABB, ray-mesh intersection
- **Subdivision**: Loop, Catmull-Clark, sqrt(3)
- **Decimation**: QEM (Quadric Error Metric) simplification
- **Parameterization**: Tutte embedding, harmonic (cotan), LSCM, MVC (Mean Value Coordinates)
- **Geodesics**: Heat Method (Crane et al. 2013), Dijkstra single/multi-source, shortest path backtracking
- **Deformation**: Laplacian surface editing (Sorkine 2004), ARAP (Sorkine & Alexa 2007)
- **Conformal mapping**: harmonic map, Mobius transform, discrete conformal scale factors
- **Boolean operations**: union / intersection / difference / symmetric difference
- **Remeshing**: isotropic remeshing
- **Triangulation**: ear-clipping & fan triangulation (planar / 3D)
- **Weld**: vertex welding by distance threshold
- **Primitives**: cube / UV-sphere / cylinder / cone / grid / torus builders
- **I/O**: OBJ (n-gon), PLY (ASCII), STL (ASCII & binary) load/save
- **Property system**: OpenMesh-style dynamic properties (`Any + TypeId` type erasure)
- **Builtin attributes**: newtype wrappers for vertex normal / UV / color / face normal
- **Validation**: full topology self-check (twin / next / manifold / degeneracy)
- **Connectivity**: connected components (face / vertex BFS), merge / split
- **Orientation**: consistency detection & repair
- **BVH**: bounding volume hierarchy (AABB tree) for ray / nearest query
- **Sparse linear algebra**: symmetric system builder + conjugate gradient
- **Intrinsic Delaunay**: intrinsic edge flips for Delaunay triangulation (Fisher 2007)
- **Direction fields**: N-RoSy fields via covariant Laplacian eigenvalue (Knoppel 2013)
- **SDF & Marching Cubes**: signed distance functions + isosurface extraction (Lorensen 1987)
- **Mesh repair**: hole filling, degenerate face removal, isolated vertex cleanup
- **Robust predicates**: Shewchuk orient2d / orient3d / incircle / insphere
- **Parallelism**: rayon-based parallel iterators

## Quick start

```toml
[dependencies]
halfedge = "0.1"
```

```rust
use halfedge::storage::{MeshStorage, Vertex};
use halfedge::topology_ops::add_triangle;

let mut mesh = MeshStorage::new();
let v0 = mesh.add_vertex(Vertex::new([0.0, 0.0, 0.0]));
let v1 = mesh.add_vertex(Vertex::new([1.0, 0.0, 0.0]));
let v2 = mesh.add_vertex(Vertex::new([0.0, 1.0, 0.0]));
add_triangle(&mut mesh, v0, v1, v2).unwrap();
```

## Module overview

| Module | Description |
|--------|-------------|
| `ids` | Strong-type handles (`VertexId` / `HalfEdgeId` / `FaceId` / `EdgeId`) |
| `storage` | `MeshStorage` container (slotmap-backed stable handles) |
| `traversal` | Eager/lazy neighborhood iterators, boundary loops, k-ring, undirected edges |
| `query` | Chainable query DSL (Builder pattern) |
| `topology_ops` | split / flip / collapse / extrude / poke, `add_triangle` builder |
| `geometry` | Lengths, areas, normals, cotan Laplacian, curvatures, AABB, ray intersection |
| `subdiv` | Loop / Catmull-Clark / sqrt(3) |
| `decimate` | QEM (Quadric Error Metric) simplification |
| `parameterization` | Tutte / Harmonic / LSCM / MVC |
| `geodesics` | Heat Method / Dijkstra (single & multi-source) / shortest path |
| `deformation` | Laplacian surface editing / ARAP |
| `conformal` | Harmonic map, Mobius transform, discrete conformal scale factors |
| `boolean` | Union / intersection / difference / symmetric difference |
| `remesh` | Isotropic remeshing |
| `triangulation` | Ear-clipping & fan triangulation (planar / 3D) |
| `weld` | Vertex welding by distance threshold |
| `connectivity` | Connected components (face / vertex BFS), merge / split |
| `orientation` | Face orientation consistency detection & repair |
| `bvh` | Bounding volume hierarchy (AABB tree) for ray / nearest query |
| `primitives` | Cube / sphere / cylinder / cone / grid / torus builders |
| `io` | OBJ (n-gon) / PLY (ASCII) / STL (ASCII & binary) load/save |
| `export` | wgpu-compatible vertex / index buffers |
| `property` | OpenMesh-style dynamic properties (`Any + TypeId` type erasure) |
| `builtin_attrs` | Newtype wrappers for vertex normal / UV / color / face normal |
| `validate` | Topology self-check (twin / next / manifold / degeneracy) |
| `linalg` | Sparse linear algebra (symmetric system + conjugate gradient) |
| `intrinsic` | Intrinsic Delaunay triangulation (edge flips) |
| `direction_field` | N-RoSy direction fields (covariant Laplacian eigenvalue) |
| `sdf` | SDF primitives, CSG operations, Marching Cubes |
| `repair` | Hole filling, degenerate face removal, isolated vertex cleanup |
| `predicates` | Shewchuk robust geometric predicates (orient2d / orient3d / incircle / insphere) |
| `test_util` | Test fixtures (`build_icosphere`) |

## Examples

The [`examples/`](examples/) directory contains 15 standalone runnable examples. 14 produce text output via `println!`; the 15th is a GPU-accelerated interactive 3D viewer.

| Example | Module | Run |
|---------|--------|-----|
| `storage_basic` | `storage` | `cargo run --example storage_basic` |
| `build_mesh` | `io` | `cargo run --example build_mesh` |
| `obj_io` | `io` | `cargo run --example obj_io` |
| `traversal` | `traversal` | `cargo run --example traversal` |
| `topology_ops` | `topology_ops` | `cargo run --example topology_ops` |
| `extrude_face` | `topology_ops` | `cargo run --example extrude_face` |
| `geometry_query` | `geometry` | `cargo run --example geometry_query` |
| `laplacian_smooth` | `geometry` | `cargo run --example laplacian_smooth` |
| `point_triangle_distance` | `geometry` | `cargo run --example point_triangle_distance` |
| `loop_subdivision` | `subdiv` | `cargo run --example loop_subdivision` |
| `property` | `property` | `cargo run --example property` |
| `validate` | `validate` | `cargo run --example validate` |
| `export_wgpu` | `export` | `cargo run --example export_wgpu` |
| `icosphere` | `test_util` | `cargo run --example icosphere` |
| `engvis_viewer` | visualization | `cargo run --example engvis_viewer` |

### Interactive 3D viewer

`engvis_viewer` uses [`engvis-renderer`](https://crates.io/crates/engvis-renderer) (wgpu 27 / egui 0.33 / winit 0.30) to render halfedge mesh operations in real time. It is declared as a `dev-dependency` and does not affect the published library.

```sh
cargo run --example engvis_viewer                          # default icosphere
cargo run --example engvis_viewer -- icosphere 2           # 2-level icosphere
cargo run --example engvis_viewer -- subdivision loop 2    # Loop subdivision x2
cargo run --example engvis_viewer -- extrude               # face extrusion
cargo run --example engvis_viewer -- smooth 10             # Laplacian smoothing
cargo run --example engvis_viewer -- topology split        # edge split / flip / collapse
```

The right-side UI panel switches between 5 operations and their parameters at runtime; no restart required.

## Design

- **Strong typing**: all handles are slotmap keys with compile-time distinction
- **Eager & lazy iterators**: pre-collected `Vec` versions (allow `&mut mesh` during iteration) and lazy versions (zero heap allocation, hold `&MeshStorage` borrow)
- **Robust degeneracy guards**: all panic paths covered (empty collections, zero-length, division by zero, index out of bounds, `unwrap`)
- **Symmetric sparse system builder**: `SparseSystem::add` writes both `(i,j)` and `(j,i)` for symmetric matrices
- **477 unit tests** covering all modules

## Documentation

Each module has a companion LaTeX design document in [`docs/`](docs/) (compiled to PDF) with algorithm derivations, TikZ flowcharts, and complexity analysis.

A Chinese tutorial website built with [mdbook](https://rust-lang.github.io/mdBook/) is in [`book/`](book/). Build it locally:

```sh
cargo install mdbook
mdbook serve book --open
```

API documentation: <https://docs.rs/halfedge>

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
