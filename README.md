# halfedge

A half-edge mesh data structure library for Rust, providing comprehensive tools for 3D mesh processing: traversal, topology operations, geometry, subdivision, decimation, parameterization, geodesics, deformation, boolean operations, and more.

## Features

- **Half-edge data structure** with slotmap-based stable handles (`VertexId` / `HalfEdgeId` / `FaceId` / `EdgeId`)
- **Traversal**: eager & lazy iterators (zero-allocation), boundary loops, k-ring neighborhoods
- **Topology operations**: edge split / flip / collapse / extrude / poke, `add_triangle` builder
- **Geometry**: edge lengths, face areas/normals, cotan Laplacian, dihedral angles, point-triangle distance, AABB
- **Subdivision**: Loop, Catmull-Clark, √3
- **Decimation**: QEM (Quadric Error Metric) simplification
- **Parameterization**: Tutte embedding, harmonic (cotan), LSCM, MVC (Mean Value Coordinates)
- **Geodesics**: Heat Method (Crane et al. 2013), Dijkstra single/multi-source, shortest path backtracking
- **Deformation**: Laplacian surface editing (Sorkine 2004), ARAP (Sorkine & Alexa 2007)
- **Boolean operations**: union / intersection / difference
- **Remeshing**: isotropic remeshing
- **I/O**: OBJ (with n-gon) and PLY (ASCII) load/save
- **Property system**: OpenMesh-style dynamic properties (`Any + TypeId` type erasure)
- **Validation**: full topology self-check (twin / next / manifold / degeneracy)
- **Connectivity**: connected components (face / vertex BFS)
- **Orientation**: consistency detection & repair
- **BVH**: bounding volume hierarchy for acceleration
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
| `ids` | Strong-type handles |
| `storage` | `MeshStorage` container (slotmap) |
| `traversal` | Eager/lazy neighborhood iterators, boundary loops, k-ring |
| `query` | Chainable query DSL (Builder pattern) |
| `topology_ops` | split / flip / collapse / extrude / poke |
| `geometry` | Lengths, areas, normals, cotan Laplacian, AABB |
| `subdiv` | Loop / Catmull-Clark / √3 |
| `decimate` | QEM simplification |
| `parameterization` | Tutte / Harmonic / LSCM / MVC |
| `geodesics` | Heat Method / Dijkstra / shortest path |
| `deformation` | Laplacian / ARAP |
| `boolean` | Union / intersection / difference |
| `remesh` | Isotropic remeshing |
| `bvh` | Bounding volume hierarchy |
| `io` | OBJ / PLY load/save |
| `export` | wgpu-compatible buffers |
| `property` | OpenMesh-style dynamic properties |
| `validate` | Topology self-check |
| `linalg` | Sparse linear algebra (CG) |

## Design

- **Strong typing**: all handles are slotmap keys with compile-time distinction
- **Eager & lazy iterators**: pre-collected `Vec` versions (allow `&mut mesh` during iteration) and lazy versions (zero heap allocation, hold `&MeshStorage` borrow)
- **Robust degeneracy guards**: all panic paths covered (empty collections, zero-length, division by zero, index out of bounds, `unwrap`)
- **Symmetric sparse system builder**: `SparseSystem::add` writes both `(i,j)` and `(j,i)` for symmetric matrices
- **394 unit tests** covering all modules

## Documentation

Each module has a companion LaTeX design document in [`docs/`](docs/) (compiled to PDF) with algorithm derivations, TikZ flowcharts, and complexity analysis.

API documentation: <https://docs.rs/halfedge>

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
