//! 拓扑操作基准：split / flip / collapse 在 icosphere(3) 上的耗时与方差
//!
//! 运行：`cargo bench --bench topology_ops`
//!
//! ## 测试网格
//! `build_icosphere(3)`：V=642, F=1280, E=1920（闭合球面，所有边均为内部边）
//!
//! ## 测量方法
//! 每次迭代通过 `iter_batched` 在 setup 中 `mesh.clone()`（不计入计时），
//! 然后在计时区对预选的 100 条内部边顺序执行操作：
//! - 成功则计数 +1
//! - 失败（边已被前序操作影响 / 链接条件不满足）则跳过
//! - 累计 100 次成功或遍历完候选边后结束
//!
//! 输出的耗时为 100 次操作的总时间，除以 100 得到单次平均耗时。
//! criterion 自动报告 mean / median / std. dev.（方差）。
//!
//! ## 候选边选取
//! 遍历所有半边，筛选 `face.is_some() && twin.is_some()` 的内部边，
//! 且每对 twin 只取一个方向（`he < twin`），用固定种子打乱后取前 N 条。

use std::hint::black_box;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use halfedge::build_icosphere;
use halfedge::topology_ops::{collapse_edge, flip_edge, split_edge};
use halfedge::{HalfEdgeId, MeshStorage};

/// 目标操作次数。
const OPS_TARGET: u32 = 100;

/// 收集所有内部边（每对 twin 仅取一个方向）。
fn collect_internal_edges(mesh: &MeshStorage) -> Vec<HalfEdgeId> {
    let mut edges = Vec::new();
    for he_id in mesh.halfedge_ids() {
        if let Some(h) = mesh.get_halfedge(he_id)
            && h.face.is_some()
            && let Some(twin) = h.twin
            && he_id < twin
        {
            edges.push(he_id);
        }
    }
    edges
}

/// 简单的确定性伪随机洗牌（LCG），保证每次运行结果可复现。
fn deterministic_shuffle<T>(v: &mut [T]) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..v.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (state >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

// ============================================================
// split_edge 基准
// ============================================================

fn bench_split(c: &mut Criterion) {
    let mesh = build_icosphere(3);
    let mut candidates = collect_internal_edges(&mesh);
    deterministic_shuffle(&mut candidates);

    println!(
        "[split] icosphere(3): V={}, E={}, 候选边={}",
        mesh.vertex_count(),
        mesh.halfedge_count() / 2,
        candidates.len()
    );

    let mut group = c.benchmark_group("topology_ops/split");
    group.sample_size(30);

    group.bench_function("split_100_edges", |b| {
        b.iter_batched(
            || mesh.clone(),
            |mut m| {
                let mut ok = 0u32;
                for &he in &candidates {
                    if split_edge(&mut m, he).is_ok() {
                        ok += 1;
                        if ok >= OPS_TARGET {
                            break;
                        }
                    }
                }
                black_box(ok)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================
// flip_edge 基准
// ============================================================

fn bench_flip(c: &mut Criterion) {
    let mesh = build_icosphere(3);
    let mut candidates = collect_internal_edges(&mesh);
    deterministic_shuffle(&mut candidates);

    println!(
        "[flip] icosphere(3): V={}, E={}, 候选边={}",
        mesh.vertex_count(),
        mesh.halfedge_count() / 2,
        candidates.len()
    );

    let mut group = c.benchmark_group("topology_ops/flip");
    group.sample_size(30);

    group.bench_function("flip_100_edges", |b| {
        b.iter_batched(
            || mesh.clone(),
            |mut m| {
                let mut ok = 0u32;
                for &he in &candidates {
                    if flip_edge(&mut m, he).is_ok() {
                        ok += 1;
                        if ok >= OPS_TARGET {
                            break;
                        }
                    }
                }
                black_box(ok)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================
// collapse_edge 基准
// ============================================================

fn bench_collapse(c: &mut Criterion) {
    let mesh = build_icosphere(3);
    let mut candidates = collect_internal_edges(&mesh);
    deterministic_shuffle(&mut candidates);

    println!(
        "[collapse] icosphere(3): V={}, E={}, 候选边={}",
        mesh.vertex_count(),
        mesh.halfedge_count() / 2,
        candidates.len()
    );

    let mut group = c.benchmark_group("topology_ops/collapse");
    group.sample_size(30);

    group.bench_function("collapse_100_edges", |b| {
        b.iter_batched(
            || mesh.clone(),
            |mut m| {
                let mut ok = 0u32;
                for &he in &candidates {
                    // collapse_edge 内置链接条件检查，失败返回 Err，跳过即可
                    if collapse_edge(&mut m, he).is_ok() {
                        ok += 1;
                        if ok >= OPS_TARGET {
                            break;
                        }
                    }
                }
                black_box(ok)
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_split, bench_flip, bench_collapse);
criterion_main!(benches);
