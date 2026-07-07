//! 几何算法基准：测地线、参数化、简化、内蕴 Delaunay、方向场、Marching Cubes
//!
//! 运行：`cargo bench --bench geometry_algs`
//!
//! ## 测试网格
//! - `build_icosphere(n)`：闭合三角网格，V = 10·4^n + 2，F = 20·4^n
//!   - n=1: V=42, F=80；n=2: V=162, F=320；n=3: V=642, F=1280
//! - `build_grid(w, d, sx, sz)`：平面网格（有边界），用于参数化
//!
//! ## 算法分组
//! 1. **geodesic**：Heat Method（Poisson 求解）vs Dijkstra（优先队列）
//! 2. **parameterization**：Tutte 重心映射 vs LSCM
//! 3. **decimate**：QEM 简化（icosphere(3) → 目标面数）
//! 4. **intrinsic**：内蕴 Delaunay 三角剖分
//! 5. **direction_field**：N-RoSy 方向场（协变拉普拉斯特征值）
//! 6. **marching_cubes**：SDF 等值面提取

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use halfedge::{
    McParams, SdfSphere, VertexId, build_grid, build_icosphere, compute_intrinsic_lengths,
    decimate_qem, dijkstra_geodesic, geodesic_distance_from_vertex, intrinsic_delaunay, lscm,
    march_sdf, smoothest_cross_field, smoothest_nrosy, tutte_embedding,
};

// ============================================================
// 辅助：取第一个顶点作为测地线源点
// ============================================================

fn first_vertex(mesh: &halfedge::MeshStorage) -> VertexId {
    mesh.vertex_ids().next().expect("非空网格")
}

// ============================================================
// 1. 测地线：Heat Method vs Dijkstra
// ============================================================

fn bench_geodesic(c: &mut Criterion) {
    let mut group = c.benchmark_group("geodesic");
    group.sample_size(30);

    for n in 1..=3usize {
        let mesh = build_icosphere(n);
        let v = mesh.vertex_count();
        let src = first_vertex(&mesh);
        println!("[geodesic] icosphere({}): V={}", n, v);

        group.bench_with_input(BenchmarkId::new("heat_method", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(geodesic_distance_from_vertex(mesh, src));
            });
        });

        group.bench_with_input(BenchmarkId::new("dijkstra", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(dijkstra_geodesic(mesh, src));
            });
        });
    }

    group.finish();
}

// ============================================================
// 2. 参数化：Tutte vs LSCM（需要边界网格）
// ============================================================

fn bench_parameterization(c: &mut Criterion) {
    let mut group = c.benchmark_group("parameterization");
    group.sample_size(30);

    // build_grid(width, depth, segments_x, segments_z)
    for segs in [4, 8, 16usize] {
        let mesh = build_grid(2.0, 2.0, segs, segs);
        let v = mesh.vertex_count();
        let f = mesh.face_count();
        println!("[parameterization] grid({}): V={}, F={}", segs, v, f);

        group.bench_with_input(BenchmarkId::new("tutte", segs), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(tutte_embedding(mesh));
            });
        });

        group.bench_with_input(BenchmarkId::new("lscm", segs), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(lscm(mesh));
            });
        });
    }

    group.finish();
}

// ============================================================
// 3. QEM 简化（需要 clone，因为操作消耗 mesh）
// ============================================================

fn bench_decimate(c: &mut Criterion) {
    let mut group = c.benchmark_group("decimate");
    group.sample_size(20);

    for n in 2..=3usize {
        let mesh = build_icosphere(n);
        let f0 = mesh.face_count();
        let target = f0 / 4; // 简化到 1/4
        println!("[decimate] icosphere({}): F0={}, target={}", n, f0, target);

        group.bench_with_input(BenchmarkId::new("qem", n), &mesh, |b, mesh| {
            b.iter_batched(
                || mesh.clone(),
                |mut m| {
                    black_box(decimate_qem(&mut m, target).ok());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================
// 4. 内蕴 Delaunay 三角剖分（需要 clone）
// ============================================================

fn bench_intrinsic(c: &mut Criterion) {
    let mut group = c.benchmark_group("intrinsic");
    group.sample_size(20);

    for n in 1..=3usize {
        let mesh = build_icosphere(n);
        let e = mesh.halfedge_count() / 2;
        println!("[intrinsic] icosphere({}): E={}", n, e);

        group.bench_with_input(BenchmarkId::new("compute_lengths", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(compute_intrinsic_lengths(mesh));
            });
        });

        group.bench_with_input(BenchmarkId::new("delaunay", n), &mesh, |b, mesh| {
            b.iter_batched(
                || {
                    let m = mesh.clone();
                    let l = compute_intrinsic_lengths(&m);
                    (m, l)
                },
                |(mut m, mut l)| {
                    black_box(intrinsic_delaunay(&mut m, &mut l));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// ============================================================
// 5. N-RoSy 方向场
// ============================================================

fn bench_direction_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("direction_field");
    group.sample_size(20);

    for n in 1..=3usize {
        let mesh = build_icosphere(n);
        let f = mesh.face_count();
        println!("[direction_field] icosphere({}): F={}", n, f);

        // N=1 向量场
        group.bench_with_input(BenchmarkId::new("nrosy_n1", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(smoothest_nrosy(mesh, 1));
            });
        });

        // N=2 交叉场
        group.bench_with_input(BenchmarkId::new("cross_field_n2", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(smoothest_cross_field(mesh));
            });
        });
    }

    group.finish();
}

// ============================================================
// 6. Marching Cubes（SDF → 网格）
// ============================================================

fn bench_marching_cubes(c: &mut Criterion) {
    let mut group = c.benchmark_group("marching_cubes");
    group.sample_size(20);

    let sphere = SdfSphere {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
    };

    for res in [16, 32, 64usize] {
        let params = McParams {
            origin: [-2.0, -2.0, -2.0],
            cell_size: [4.0 / res as f64; 3],
            resolution: [res, res, res],
            isolevel: 0.0,
        };
        println!("[marching_cubes] resolution={}", res);

        group.bench_with_input(BenchmarkId::new("sphere", res), &params, |b, params| {
            b.iter(|| {
                black_box(march_sdf(&sphere, params));
            });
        });
    }

    group.finish();
}

// ============================================================
// 7. SOA vs AoS 位置访问对比
// ============================================================

/// 对比「SlotMap<Vertex> 逐顶点遍历」与「SOA positions_dense 连续遍历」
/// 在 mesh_volume / mesh_aabb / 纯累加三种场景下的性能差异。
fn bench_soa_vs_aos(c: &mut Criterion) {
    use halfedge::geometry::{mesh_aabb, mesh_volume, surface_area};
    let mut group = c.benchmark_group("soa_vs_aos");
    group.sample_size(50);

    for n in 1..=3usize {
        let mesh = build_icosphere(n);
        let v = mesh.vertex_count();
        let f = mesh.face_count();
        println!("[soa_vs_aos] icosphere({}): V={}, F={}", n, v, f);

        // 场景 A：mesh_volume（已改用 SOA 缓存）
        group.bench_with_input(BenchmarkId::new("mesh_volume_soa", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(mesh_volume(mesh));
            });
        });

        // 场景 B：mesh_aabb（已改用 SOA 缓存）
        group.bench_with_input(BenchmarkId::new("mesh_aabb_soa", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(mesh_aabb(mesh));
            });
        });

        // 场景 C：纯 AoS 遍历（基线：直接遍历 Vertex.position）
        group.bench_with_input(BenchmarkId::new("iter_vertex_aos", n), &mesh, |b, mesh| {
            b.iter(|| {
                let mut sum = [0.0f64; 3];
                for v in mesh.vertices() {
                    sum[0] += v.position[0];
                    sum[1] += v.position[1];
                    sum[2] += v.position[2];
                }
                black_box(sum);
            });
        });

        // 场景 D：纯 SOA 遍历（基线：遍历 positions_dense 切片）
        group.bench_with_input(
            BenchmarkId::new("iter_positions_soa", n),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    let mut sum = [0.0f64; 3];
                    for p in mesh.positions_dense() {
                        sum[0] += p[0];
                        sum[1] += p[1];
                        sum[2] += p[2];
                    }
                    black_box(sum);
                });
            },
        );

        // 场景 E：surface_area（已改用 SOA 缓存，但 face_area 内部仍走 SlotMap）
        group.bench_with_input(BenchmarkId::new("surface_area", n), &mesh, |b, mesh| {
            b.iter(|| {
                black_box(surface_area(mesh));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_geodesic,
    bench_parameterization,
    bench_decimate,
    bench_intrinsic,
    bench_direction_field,
    bench_marching_cubes,
    bench_soa_vs_aos
);
criterion_main!(benches);
