//! Loop 细分基准：`loop_subdivide` 在不同规模 icosphere 上的 scaling
//!
//! 运行：`cargo bench --bench subdiv`
//!
//! ## 测试网格
//! `build_icosphere(n)` 对 n=0..=3 各执行 1 次 `loop_subdivide`：
//! - n=0: V=12 → 162
//! - n=1: V=42 → 642
//! - n=2: V=162 → 2562
//! - n=3: V=642 → 10242
//!
//! ## 测量目标
//! - 各级别的单次 `loop_subdivide` 耗时
//! - 顶点数增长（应精确为 4 倍 + 调整项）
//! - 验证耗时与顶点数的关系是否为线性 $O(V + E)$
//!
//! ## 预期结论
//! 相邻级别顶点数比约 4×，若耗时比也接近 4×，则验证线性复杂度。

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use halfedge::{build_icosphere, loop_subdivide};

fn bench_loop_subdivide(c: &mut Criterion) {
    let mut group = c.benchmark_group("subdiv/loop");
    // loop_subdivide 在大网格上较慢，减少样本数以控制总时长
    group.sample_size(10);

    for n in 0..=3usize {
        let mesh = build_icosphere(n);
        let v_before = mesh.vertex_count();
        let f_before = mesh.face_count();
        let he_before = mesh.halfedge_count();

        // 预热 + 正确性验证（不计入计时）
        let check = loop_subdivide(&mesh);
        let v_after = check.vertex_count();
        let f_after = check.face_count();
        println!(
            "[loop_subdivide] icosphere({}): V={}→{}, F={}→{}, HE={}→{} (耗时见下表)",
            n,
            v_before,
            v_after,
            f_before,
            f_after,
            he_before,
            check.halfedge_count()
        );
        // Loop 细分：每个三角面分裂为 4 个，V_new = V + E/2, F_new = 4F
        assert_eq!(f_after, f_before * 4, "面数应精确 4 倍");
        // 顶点数 V_new = V_old + E_old（每条边产生一个中点），E_old = HE/2
        let expected_v = v_before + he_before / 2;
        assert_eq!(v_after, expected_v, "顶点数应为 V + E");

        group.bench_with_input(BenchmarkId::new("icosphere", n), &mesh, |b, mesh| {
            b.iter(|| {
                let result = loop_subdivide(mesh);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_loop_subdivide);
criterion_main!(benches);
