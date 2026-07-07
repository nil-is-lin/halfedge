# Contributing to halfedge

Thanks for your interest in contributing! This document covers setup, conventions, and the pull request workflow.

## Development setup

**Requirements**: Rust stable (1.87+), edition 2024.

```sh
git clone https://github.com/nil-is-lin/halfedge.git
cd halfedge
cargo build
cargo test
```

## Maintainers

This project is currently maintained by a single author. Contributions, code
reviews, and **co-maintainer volunteers** are very welcome — please open a PR or
reach out. Bus-factor is a known risk pre-1.0; broadening the maintainer base is a
priority as the project approaches release.

Optional tooling:

- [`mdbook`](https://rust-lang.github.io/mdBook/) for the tutorial site: `cargo install mdbook && mdbook serve book --open`
- A LaTeX distribution if you want to rebuild the design docs in `docs/`

## Code conventions

- **Formatting**: `cargo fmt --all` must produce no diffs. CI checks this with `--check`.
- **Clippy**: zero warnings. CI runs `cargo clippy --all-targets -- -D warnings`.
- **No `unsafe`**: the codebase is `unsafe`-free by design. If you believe `unsafe` is necessary, open an issue first to discuss.
- **No `panic!()` in library code**: public APIs return `Result<_, E>`; internal assertions use `.expect("reason")` with a descriptive message, never bare `.unwrap()`.
- **No `eprintln!()`/`println!()` in library code**: use `log::debug!`/`log::warn!` instead.

## Testing

- Every new public function needs at least one unit test.
- Run the full suite before pushing:

```sh
cargo test --all-features
cargo test --doc
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

- Doc examples (`/// ```rust`) are compiled by `cargo test --doc` — keep them runnable.
- Benchmarks (`cargo bench`) are in `benches/`; add one if your change affects a hot path.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) loosely:

```
<type>: <short summary in lowercase>

<optional body explaining why, not what>
```

Types: `fix`, `feat`, `refactor`, `docs`, `test`, `perf`, `chore`.

Examples:

```
feat: add intrinsic Delaunay edge flip
fix: handle degenerate triangle in QEM collapse
refactor: split geometry.rs into submodules
docs: add cotan Laplacian derivation to docs/
```

## Pull request workflow

1. Fork and create a branch from `main` (not `master`).
2. Make your changes. Keep the diff focused — one logical change per PR.
3. Ensure all CI checks pass locally (formatting, clippy, tests).
4. If adding a public API, document it with `///` and include a doc example.
5. If changing behaviour, update `CHANGELOG.md` under the `[Unreleased]` section.
6. Open a PR and fill in the template. Reference any related issues.

## Project structure

```
src/
  ids.rs              # VertexId / HalfEdgeId / FaceId / EdgeId handles
  storage.rs          # MeshStorage (slotmap-backed container)
  traversal.rs        # eager/lazy iterators, boundary loops, k-ring
  query.rs            # chainable query DSL
  topology_ops/       # split / flip / collapse / extrude / poke
  geometry/           # areas, normals, curvatures, distance, AABB
  subdiv/             # Loop / Catmull-Clark / sqrt(3)
  decimate.rs         # QEM simplification
  parameterization.rs # Tutte / Harmonic / LSCM / MVC
  geodesics.rs        # Heat Method / Dijkstra / shortest path
  deformation.rs      # Laplacian editing / ARAP
  boolean.rs          # union / intersection / difference
  remesh.rs           # isotropic remeshing
  io/                 # OBJ / PLY / STL / OFF / glTF
  ...
```

Each module has a companion design document in `docs/` (LaTeX, compiled to PDF). If you add a new algorithm, consider adding a design doc explaining the maths.

## Reporting issues

- Bugs: use the bug report template. Include a minimal repro (a `cargo run --example` snippet is ideal).
- Feature requests: use the feature request template. Describe the use case, not just the solution.

## License

By contributing, you agree that your contributions will be dual-licensed under MIT or Apache-2.0, at the option of the user.
