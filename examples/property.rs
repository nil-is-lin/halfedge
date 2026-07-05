//! 示例：OpenMesh 风格属性系统
//!
//! 演示 [`halfedge::property`] 模块的核心用法：
//! - 为顶点/半边/面附加任意类型的自定义数据
//! - 类型擦除（同一容器内并存多种类型）
//! - 加权拉普拉斯平滑（用顶点属性作权重）
//! - 删除元素时同步清理属性
//!
//! 运行：`cargo run --example property`

use halfedge::property::{MeshProperties, PropertyHandle};
use halfedge::traversal::VertexAdjacentVerts;
use halfedge::{VertexId, build_icosphere, build_mesh_from_vertices_and_faces};

type Vec3 = [f64; 3];

fn main() {
    println!("====== 1. 基础：注册并读写顶点属性 ======");
    let mesh = build_icosphere(1);
    let mut props = MeshProperties::new();

    // 注册一个 f64 类型的顶点属性（语义：顶点权重）
    let weight: PropertyHandle<f64> = props.add_vertex_prop();
    for v in mesh.vertex_ids() {
        props.set_vertex_prop(weight, v, 1.0);
    }
    println!(
        "已注册顶点属性 f64，已设置 {} 个顶点的权重",
        mesh.vertex_count()
    );
    println!(
        "vertex_prop_type_count = {}",
        props.vertex_prop_type_count()
    );

    // 读取
    let v0 = mesh.vertex_ids().next().unwrap();
    println!("v{:?}.weight = {:?}", v0, props.get_vertex_prop(weight, v0));

    println!("\n====== 2. 类型擦除：同一容器并存多种类型 ======");
    // 同时注册 i32 标签、String 名称、bool 选中状态
    let label: PropertyHandle<i32> = props.add_vertex_prop();
    let name: PropertyHandle<String> = props.add_vertex_prop();
    let selected: PropertyHandle<bool> = props.add_vertex_prop();

    for (i, v) in mesh.vertex_ids().enumerate() {
        props.set_vertex_prop(label, v, i as i32);
        props.set_vertex_prop(name, v, format!("v{}", i));
        props.set_vertex_prop(selected, v, i % 3 == 0);
    }
    println!("已追加 i32 / String / bool 三种顶点属性");
    println!(
        "vertex_prop_type_count = {}",
        props.vertex_prop_type_count()
    );

    // 互不干扰地读取
    let v = mesh.vertex_ids().next().unwrap();
    println!(
        "v{:?}: label={:?}, name={:?}, selected={:?}, weight={:?}",
        v,
        props.get_vertex_prop(label, v),
        props.get_vertex_prop(name, v),
        props.get_vertex_prop(selected, v),
        props.get_vertex_prop(weight, v),
    );

    println!("\n====== 3. 加权拉普拉斯平滑 ======");
    // 加权公式：p_v' = Σ w_u * p_u / Σ w_u  (u ∈ N(v))
    // 对 icosphere 顶部顶点赋较大权重（如 5.0），其余为 1.0，
    // 使得顶部顶点附近的邻居被「拉向」顶部位置更多。

    let mesh = build_icosphere(1);
    let mut props = MeshProperties::new();
    let w: PropertyHandle<f64> = props.add_vertex_prop();

    // 找 z 最高的顶点（北极），赋权重 5.0；其余 1.0
    let mut north_pole: Option<VertexId> = None;
    let mut max_z = f64::NEG_INFINITY;
    for v in mesh.vertex_ids() {
        let p = mesh.get_vertex(v).unwrap().position;
        if p[2] > max_z {
            max_z = p[2];
            north_pole = Some(v);
        }
    }
    let north_pole = north_pole.unwrap();

    for v in mesh.vertex_ids() {
        let w_v = if v == north_pole { 5.0 } else { 1.0 };
        props.set_vertex_prop(w, v, w_v);
    }
    println!(
        "北极顶点 = {:?}，权重 = 5.0；其余顶点权重 = 1.0",
        north_pole
    );

    // 取北极的邻居，计算加权平均位置与等权平均位置
    let pole_pos = mesh.get_vertex(north_pole).unwrap().position;
    let neighbors: Vec<VertexId> = VertexAdjacentVerts::new(&mesh, north_pole).collect();
    println!("北极邻居数 = {}", neighbors.len());

    // 等权平均
    let mut sum_uniform = [0.0_f64; 3];
    for u in &neighbors {
        let p = mesh.get_vertex(*u).unwrap().position;
        for k in 0..3 {
            sum_uniform[k] += p[k];
        }
    }
    let n = neighbors.len() as f64;
    let uniform_avg = [sum_uniform[0] / n, sum_uniform[1] / n, sum_uniform[2] / n];

    // 加权平均（这里权重是邻居自身的权重 w_u）
    let mut sum_weighted = [0.0_f64; 3];
    let mut sum_w = 0.0_f64;
    for u in &neighbors {
        let p = mesh.get_vertex(*u).unwrap().position;
        let w_u = *props.get_vertex_prop(w, *u).unwrap_or(&1.0);
        for k in 0..3 {
            sum_weighted[k] += w_u * p[k];
        }
        sum_w += w_u;
    }
    let weighted_avg = [
        sum_weighted[0] / sum_w,
        sum_weighted[1] / sum_w,
        sum_weighted[2] / sum_w,
    ];

    println!(
        "北极原位置           = [{:.4}, {:.4}, {:.4}]",
        pole_pos[0], pole_pos[1], pole_pos[2]
    );
    println!(
        "邻居等权平均         = [{:.4}, {:.4}, {:.4}]",
        uniform_avg[0], uniform_avg[1], uniform_avg[2]
    );
    println!(
        "邻居加权平均         = [{:.4}, {:.4}, {:.4}]",
        weighted_avg[0], weighted_avg[1], weighted_avg[2]
    );
    let uniform_disp = dist(pole_pos, uniform_avg);
    let weighted_disp = dist(pole_pos, weighted_avg);
    println!(
        "等权位移 = {:.6}，加权位移 = {:.6}（邻居权重相同时应相等）",
        uniform_disp, weighted_disp
    );

    // 演示权重差异场景：把某个邻居权重调大，平滑位置应偏向它
    if let Some(u) = neighbors.first().copied() {
        let u_pos = mesh.get_vertex(u).unwrap().position;
        props.set_vertex_prop(w, u, 10.0);
        let mut sum_w = [0.0_f64; 3];
        let mut sum_ww = 0.0_f64;
        for u2 in &neighbors {
            let p = mesh.get_vertex(*u2).unwrap().position;
            let w_u = *props.get_vertex_prop(w, *u2).unwrap_or(&1.0);
            for k in 0..3 {
                sum_w[k] += w_u * p[k];
            }
            sum_ww += w_u;
        }
        let biased = [sum_w[0] / sum_ww, sum_w[1] / sum_ww, sum_w[2] / sum_ww];
        println!(
            "\n将邻居 {:?} 权重设为 10.0（位置 [{:.4}, {:.4}, {:.4}]）后：",
            u, u_pos[0], u_pos[1], u_pos[2]
        );
        println!(
            "  加权平均位置 = [{:.4}, {:.4}, {:.4}]（应更靠近该邻居）",
            biased[0], biased[1], biased[2]
        );
    }

    println!("\n====== 4. 半边属性：缓存边长 ======");
    let mesh = build_icosphere(1);
    let mut props = MeshProperties::new();
    let edge_len: PropertyHandle<f64> = props.add_halfedge_prop();

    // 为每条半边缓存其两端距离（注意 twin 共享同一长度）
    for he in mesh.halfedge_ids() {
        let h = mesh.get_halfedge(he).unwrap();
        let tip = h.vertex;
        let origin = mesh.get_halfedge(h.twin.unwrap()).unwrap().vertex;
        let p_tip = mesh.get_vertex(tip).unwrap().position;
        let p_origin = mesh.get_vertex(origin).unwrap().position;
        let l = dist(p_tip, p_origin);
        props.set_halfedge_prop(edge_len, he, l);
    }
    println!(
        "已为 {} 条半边缓存长度，halfedge_prop_type_count = {}",
        mesh.halfedge_count(),
        props.halfedge_prop_type_count()
    );

    // 统计边长分布
    let mut min_l = f64::INFINITY;
    let mut max_l = 0.0_f64;
    let mut sum_l = 0.0_f64;
    let mut count = 0_usize;
    for he in mesh.halfedge_ids() {
        let l = *props.get_halfedge_prop(edge_len, he).unwrap();
        min_l = min_l.min(l);
        max_l = max_l.max(l);
        sum_l += l;
        count += 1;
    }
    println!(
        "边长统计：min={:.4}, max={:.4}, avg={:.4} (共 {} 条半边)",
        min_l,
        max_l,
        sum_l / count as f64,
        count
    );

    println!("\n====== 5. 面属性：缓存面法向 ======");
    let mesh = build_icosphere(1);
    let mut props = MeshProperties::new();
    let normal: PropertyHandle<Vec3> = props.add_face_prop();

    for f in mesh.face_ids() {
        let n = compute_face_normal(&mesh, f);
        if let Some(n) = n {
            props.set_face_prop(normal, f, n);
        }
    }
    println!(
        "已为 {} 个面缓存法向，face_prop_type_count = {}",
        mesh.face_count(),
        props.face_prop_type_count()
    );

    // 取第一个面的法向，应大致朝外（与面心同向）
    let f0 = mesh.face_ids().next().unwrap();
    let n0 = props.get_face_prop(normal, f0).copied().unwrap();
    println!(
        "面 {:?} 的法向 = [{:.4}, {:.4}, {:.4}]",
        f0, n0[0], n0[1], n0[2]
    );

    println!("\n====== 6. 删除元素时同步清理属性 ======");
    // 构造一个单三角形，删除其中一个顶点并验证属性同步清理
    let verts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let faces = vec![[0, 1, 2]];
    let mut mesh = build_mesh_from_vertices_and_faces(&verts, &faces);
    let mut props = MeshProperties::new();
    let w: PropertyHandle<f64> = props.add_vertex_prop();

    let vids: Vec<VertexId> = mesh.vertex_ids().collect();
    for (i, v) in vids.iter().enumerate() {
        props.set_vertex_prop(w, *v, (i + 1) as f64);
    }
    println!("删除前：");
    for v in &vids {
        println!(
            "  v{:?} exists={}, weight={:?}",
            v,
            mesh.contains_vertex(*v),
            props.get_vertex_prop(w, *v)
        );
    }

    // 用包装函数同步清理属性
    let removed = halfedge::property::remove_vertex_with_props(&mut mesh, &mut props, vids[0]);
    println!(
        "\n删除 vids[0]={:?}（返回 Some={})",
        vids[0],
        removed.is_some()
    );

    println!("删除后：");
    for v in &vids {
        println!(
            "  v{:?} exists={}, weight={:?}",
            v,
            mesh.contains_vertex(*v),
            props.get_vertex_prop(w, *v)
        );
    }
    println!("（vids[0] 的属性已同步清除；vids[1]、vids[2] 的属性保留）");

    println!("\n====== 7. newtype 模式：同底层类型注册多个属性 ======");
    // f64 同时表达 Weight 和 Temperature 两种语义
    #[derive(Debug, PartialEq)]
    struct Weight(f64);
    #[derive(Debug, PartialEq)]
    struct Temperature(f64);

    let mesh = build_icosphere(0);
    let mut props = MeshProperties::new();
    let hw: PropertyHandle<Weight> = props.add_vertex_prop();
    let ht: PropertyHandle<Temperature> = props.add_vertex_prop();

    for (i, v) in mesh.vertex_ids().enumerate() {
        props.set_vertex_prop(hw, v, Weight(1.0 + 0.1 * i as f64));
        props.set_vertex_prop(ht, v, Temperature(20.0 + i as f64));
    }

    let v = mesh.vertex_ids().next().unwrap();
    println!(
        "v{:?}: Weight={:?}, Temperature={:?}",
        v,
        props.get_vertex_prop(hw, v),
        props.get_vertex_prop(ht, v),
    );
    println!(
        "vertex_prop_type_count = {}（两种 newtype 互不冲突）",
        props.vertex_prop_type_count()
    );

    println!("\n====== 全部演示完成 ======");
}

fn dist(a: Vec3, b: Vec3) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// 计算三角面法向（归一化）。退化或非三角面返回 `None`。
fn compute_face_normal(mesh: &halfedge::MeshStorage, f: halfedge::FaceId) -> Option<Vec3> {
    use halfedge::traversal::FaceHalfEdges;
    let mut verts = Vec::with_capacity(3);
    for he in FaceHalfEdges::new(mesh, f) {
        if verts.len() >= 3 {
            return None;
        }
        let v = mesh.get_halfedge(he)?.vertex;
        verts.push(mesh.get_vertex(v)?.position);
    }
    if verts.len() != 3 {
        return None;
    }
    let ab = [
        verts[1][0] - verts[0][0],
        verts[1][1] - verts[0][1],
        verts[1][2] - verts[0][2],
    ];
    let ac = [
        verts[2][0] - verts[0][0],
        verts[2][1] - verts[0][1],
        verts[2][2] - verts[0][2],
    ];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l < 1e-12 {
        return None;
    }
    Some([n[0] / l, n[1] / l, n[2] / l])
}
