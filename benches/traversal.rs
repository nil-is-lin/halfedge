//! 遍历迭代器基准：eager（预收集） vs lazy（延迟）在 icosphere 上的对比
//!
//! 运行：`cargo bench --bench traversal`
//!
//! ## 测试网格
//! `build_icosphere(n)`，顶点数 $V = 10 \cdot 4^n + 2$：
//! - n=0: V=12, n=1: V=42, n=2: V=162, n=3: V=642, n=4: V=2562
//!
//! ## 测量维度
//! 对每组迭代器分别测量：
//! - (a) **构造耗时** `construct_*`：仅调用 `::new()` / `::lazy()`，迭代器立即 drop
//! - (b) **迭代耗时** `iter_*`：从总耗时减去构造耗时（criterion 不便直接拆分，
//!   故以 `total - construct` 估算，详见下方分析）
//! - (c) **峰值内存**：解析估算，启动时打印一次
//! - (d) **总耗时** `total_*`：构造 + 完整消费（`count()`）
//!
//! ## 内存估算（峰值，所有迭代器同时存活）
//! - eager: 每个迭代器分配 `Vec<HalfEdgeId>`，总量 $\approx N \times (\bar{d} \times 8 + 24)$ 字节
//!   （$N$ = 顶点/面数，$\bar{d}$ = 平均度数，顶点环 $\bar{d} \approx 6$，面边界 $\bar{d} = 3$）
//! - lazy: 零堆分配，仅栈上 $\sim 40$ 字节状态
//!
//! ## 预期结论
//! - 小网格（n=0,1）：eager 总耗时略低（Vec 连续内存 + 无借用开销）
//! - 大网格（n=3,4）：lazy 总耗时接近 eager，但内存为零
//! - 构造耗时：eager 包含 Vec 分配 + 收集；lazy 仅 CW 探测，应明显更快

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use halfedge::build_icosphere;
use halfedge::traversal::{FaceHalfEdges, VertexAdjacentVerts, VertexRing};
use halfedge::{FaceId, VertexId};

// ============================================================
// 辅助：预收集所有顶点/面 ID，避免 ids() 迭代器借用影响计时
// ============================================================

fn vertex_ids(mesh: &halfedge::MeshStorage) -> Vec<VertexId> {
    mesh.vertex_ids().collect()
}

fn face_ids(mesh: &halfedge::MeshStorage) -> Vec<FaceId> {
    mesh.face_ids().collect()
}

// ============================================================
// 1. VertexRing: eager vs lazy
// ============================================================

fn bench_vertex_ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertex_ring");
    group.sample_size(50);

    for n in 0..=4usize {
        let mesh = build_icosphere(n);
        let vids = vertex_ids(&mesh);
        let v = mesh.vertex_count();
        let he = mesh.halfedge_count();
        let avg_degree = he as f64 / v as f64;

        // 内存估算（所有顶点同时构造迭代器）
        let eager_mem_kb = (avg_degree * 8.0 + 24.0) * v as f64 / 1024.0;
        println!(
            "[vertex_ring] icosphere({}): V={}, d̄={:.2}, eager 峰值内存≈{:.1} KB, lazy 峰值内存=0",
            n, v, avg_degree, eager_mem_kb
        );

        // (a) 构造耗时 - eager
        group.bench_with_input(BenchmarkId::new("construct_eager", n), &vids, |b, vids| {
            b.iter(|| {
                for &vid in vids {
                    black_box(VertexRing::new(&mesh, vid));
                }
            });
        });

        // (a) 构造耗时 - lazy
        group.bench_with_input(BenchmarkId::new("construct_lazy", n), &vids, |b, vids| {
            b.iter(|| {
                for &vid in vids {
                    black_box(VertexRing::lazy(&mesh, vid));
                }
            });
        });

        // (d) 总耗时 - eager
        group.bench_with_input(BenchmarkId::new("total_eager", n), &vids, |b, vids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &vid in vids {
                    sum += VertexRing::new(&mesh, vid).count() as u64;
                }
                black_box(sum);
            });
        });

        // (d) 总耗时 - lazy
        group.bench_with_input(BenchmarkId::new("total_lazy", n), &vids, |b, vids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &vid in vids {
                    sum += VertexRing::lazy(&mesh, vid).count() as u64;
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

// ============================================================
// 2. FaceHalfEdges: eager vs lazy
// ============================================================

fn bench_face_halfedges(c: &mut Criterion) {
    let mut group = c.benchmark_group("face_halfedges");
    group.sample_size(50);

    for n in 0..=4usize {
        let mesh = build_icosphere(n);
        let fids = face_ids(&mesh);
        let f = mesh.face_count();

        let eager_mem_kb = (3.0 * 8.0 + 24.0) * f as f64 / 1024.0;
        println!(
            "[face_halfedges] icosphere({}): F={}, eager 峰值内存≈{:.1} KB, lazy 峰值内存=0",
            n, f, eager_mem_kb
        );

        group.bench_with_input(BenchmarkId::new("construct_eager", n), &fids, |b, fids| {
            b.iter(|| {
                for &fid in fids {
                    black_box(FaceHalfEdges::new(&mesh, fid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("construct_lazy", n), &fids, |b, fids| {
            b.iter(|| {
                for &fid in fids {
                    black_box(FaceHalfEdges::lazy(&mesh, fid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("total_eager", n), &fids, |b, fids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &fid in fids {
                    sum += FaceHalfEdges::new(&mesh, fid).count() as u64;
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("total_lazy", n), &fids, |b, fids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &fid in fids {
                    sum += FaceHalfEdges::lazy(&mesh, fid).count() as u64;
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

// ============================================================
// 3. VertexAdjacentVerts: eager vs lazy
// ============================================================

fn bench_adjacent_verts(c: &mut Criterion) {
    let mut group = c.benchmark_group("adjacent_verts");
    group.sample_size(50);

    for n in 0..=4usize {
        let mesh = build_icosphere(n);
        let vids = vertex_ids(&mesh);
        let v = mesh.vertex_count();
        let he = mesh.halfedge_count();
        let avg_degree = he as f64 / v as f64;

        let eager_mem_kb = (avg_degree * 8.0 + 24.0) * v as f64 / 1024.0;
        println!(
            "[adjacent_verts] icosphere({}): V={}, d̄={:.2}, eager 峰值内存≈{:.1} KB, lazy 峰值内存=0",
            n, v, avg_degree, eager_mem_kb
        );

        group.bench_with_input(BenchmarkId::new("construct_eager", n), &vids, |b, vids| {
            b.iter(|| {
                for &vid in vids {
                    black_box(VertexAdjacentVerts::new(&mesh, vid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("construct_lazy", n), &vids, |b, vids| {
            b.iter(|| {
                for &vid in vids {
                    black_box(VertexAdjacentVerts::lazy(&mesh, vid));
                }
            });
        });

        group.bench_with_input(BenchmarkId::new("total_eager", n), &vids, |b, vids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &vid in vids {
                    sum += VertexAdjacentVerts::new(&mesh, vid).count() as u64;
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("total_lazy", n), &vids, |b, vids| {
            b.iter(|| {
                let mut sum = 0u64;
                for &vid in vids {
                    sum += VertexAdjacentVerts::lazy(&mesh, vid).count() as u64;
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vertex_ring,
    bench_face_halfedges,
    bench_adjacent_verts
);
criterion_main!(benches);
