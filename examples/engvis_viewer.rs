//! 示例：使用 engvis-renderer 交互式可视化 halfedge 网格操作
//!
//! 运行：
//!   cargo run --example engvis_viewer --features viewer                 # 默认 icosphere
//!   cargo run --example engvis_viewer --features viewer -- icosphere 2  # 2 级细分 icosphere
//!   cargo run --example engvis_viewer --features viewer -- subdivision loop 2
//!   cargo run --example engvis_viewer --features viewer -- extrude
//!   cargo run --example engvis_viewer --features viewer -- smooth 10
//!
//! 交互：
//!   - 鼠标左键点击网格 → 按当前「拾取模式」选中最近 顶点 / 边 / 面
//!     拾取模式可通过右侧面板切换，或快捷键 1(顶点) / 2(边) / 3(面)
//!   - 选中后点击右侧「应用」按钮执行拓扑操作：边：Split（分裂）/ Flip（翻转）/ Collapse（塌缩）；面：Extrude（挤出）
//!   - 选中态通过 builtin_attrs 的 VertexSelected / HalfEdgeSelected / FaceSelected
//!     标记类型持久化在 MeshStorage 上，并在 3D 视图中高亮显示。
//!   - 左键拖拽旋转视角，滚轮缩放，右键平移。

use std::collections::HashMap;
use std::env;

use engvis_core::{
    camera::OrbitCamera,
    material::PbrMaterial,
    mesh::Mesh as EngvisMesh,
    scene::{Scene, SceneNode},
};
use engvis_renderer::{AppCtx, EngvisApp, EventHandling, FrameCtx, RunConfig};
use glam::{Affine3A, Quat, Vec3};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::Key;

use halfedge::{
    FaceId, FaceSelected, HalfEdgeId, HalfEdgeSelected, MeshProperties, MeshStorage,
    PropertyHandle, VertexId, VertexSelected, add_face_selection, add_halfedge_selection,
    add_vertex_selection, build_cube, build_icosphere, build_uv_sphere, catmull_clark_subdivide,
    clear_face_selection, clear_halfedge_selection, clear_vertex_selection, collapse_edge,
    extrude_face, flip_edge, is_face_selected, laplacian_smooth_mesh, loop_subdivide,
    ray_triangle_intersection, select_edge, select_face, select_vertex, selected_edge_ids,
    selected_face_ids, selected_vertex_ids, split_edge, sqrt3_subdivide,
    traversal::{FaceHalfEdges, FaceVertices},
};

// ── 操作类型（生成预设网格） ─────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    Icosphere(u32),
    Subdivision(SubdivType, u32),
    Extrude,
    Smooth(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SubdivType {
    Loop,
    CatmullClark,
    Sqrt3,
}

impl Operation {
    fn default() -> Self {
        Operation::Icosphere(1)
    }

    fn from_args(args: &[String]) -> Self {
        if args.is_empty() {
            return Self::default();
        }
        match args[0].as_str() {
            "icosphere" => {
                Operation::Icosphere(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1))
            }
            "subdivision" => {
                let t = match args.get(1).map(|s| s.as_str()) {
                    Some("cc") | Some("catmull") | Some("catmull_clark") => {
                        SubdivType::CatmullClark
                    }
                    Some("sqrt3") => SubdivType::Sqrt3,
                    _ => SubdivType::Loop,
                };
                let n = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
                Operation::Subdivision(t, n.min(4))
            }
            "extrude" => Operation::Extrude,
            "smooth" => Operation::Smooth(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10)),
            _ => Self::default(),
        }
    }

    fn label(&self) -> String {
        match self {
            Operation::Icosphere(n) => format!("Icosphere (n={})", n),
            Operation::Subdivision(t, n) => format!("{:?} Subdivision x{}", t, n),
            Operation::Extrude => "Extrude Face".into(),
            Operation::Smooth(i) => format!("Laplacian Smooth (iters={})", i),
        }
    }
}

// ── 拾取模式 ────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PickMode {
    Vertex,
    Edge,
    Face,
}

impl PickMode {
    fn label(self) -> &'static str {
        match self {
            PickMode::Vertex => "顶点 (1)",
            PickMode::Edge => "边 (2)",
            PickMode::Face => "面 (3)",
        }
    }
}

// ── 应用到选中元素的拓扑操作 ────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EditAction {
    Split,
    Flip,
    Collapse,
    Extrude,
}

// ── 根据 operation 生成 halfedge mesh（预设） ───────────────
fn generate_mesh(op: &Operation) -> (MeshStorage, String) {
    match op {
        Operation::Icosphere(n) => {
            let mesh = build_icosphere(*n as usize);
            (mesh, format!("icosphere_{}", n))
        }
        Operation::Subdivision(t, n) => {
            let mut mesh = build_icosphere(0);
            for _ in 0..*n {
                mesh = match t {
                    SubdivType::Loop => loop_subdivide(&mesh),
                    SubdivType::CatmullClark => catmull_clark_subdivide(&mesh),
                    SubdivType::Sqrt3 => sqrt3_subdivide(&mesh),
                };
            }
            let name = format!("{:?}_x{}", t, n);
            (mesh, name)
        }
        Operation::Extrude => {
            let mut mesh = build_cube(1.0);
            let faces: Vec<FaceId> = mesh.face_ids().collect();
            let target_face = faces.into_iter().max_by(|&fa, &fb| {
                let na = face_centroid_z(&mesh, fa);
                let nb = face_centroid_z(&mesh, fb);
                na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(face) = target_face {
                let _ = extrude_face(&mut mesh, face, [0.0, 0.6, 0.0]);
            }
            (mesh, "extruded_cube".into())
        }
        Operation::Smooth(iters) => {
            let mut mesh = build_uv_sphere(1.0, 8, 6);
            laplacian_smooth_mesh(&mut mesh, 0.5, *iters as usize);
            (mesh, format!("smoothed_{}", iters))
        }
    }
}

/// 计算面重心的 z 坐标（用于挑选顶面）
fn face_centroid_z(mesh: &MeshStorage, face: FaceId) -> f64 {
    let verts: Vec<[f64; 3]> = FaceHalfEdges::new(mesh, face)
        .filter_map(|he| mesh.get_halfedge(he))
        .map(|h| h.vertex)
        .filter_map(|v| mesh.get_vertex(v))
        .map(|v| v.position)
        .collect();
    if verts.is_empty() {
        return f64::NEG_INFINITY;
    }
    verts.iter().map(|p| p[2]).sum::<f64>() / verts.len() as f64
}

// ── 小向量数学（f64） ───────────────────────────────────────
fn vsub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn vadd(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn vscale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn vdot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn vcross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn vlen(a: [f64; 3]) -> f64 {
    vdot(a, a).sqrt()
}
fn vnorm(a: [f64; 3]) -> [f64; 3] {
    let l = vlen(a);
    if l < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        vscale(a, 1.0 / l)
    }
}

/// 点到线段距离的平方
fn dist_point_seg_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = vsub(b, a);
    let ap = vsub(p, a);
    let len2 = vdot(ab, ab);
    let t = if len2 < 1e-18 {
        0.0
    } else {
        (vdot(ap, ab) / len2).clamp(0.0, 1.0)
    };
    let proj = vadd(a, vscale(ab, t));
    let d = vsub(p, proj);
    vdot(d, d)
}

// ── 标记几何（顶点八面体 / 边细长方体） ────────────────────
/// 返回以 `c` 为中心、半径 `r` 的八面体（已做双面处理）。
fn octahedron(c: [f64; 3], r: f64) -> (Vec<[f64; 3]>, Vec<u32>) {
    let verts: Vec<[f64; 3]> = vec![
        [c[0] + r, c[1], c[2]],
        [c[0] - r, c[1], c[2]],
        [c[0], c[1] + r, c[2]],
        [c[0], c[1] - r, c[2]],
        [c[0], c[1], c[2] + r],
        [c[0], c[1], c[2] - r],
    ];
    let idx: Vec<u32> = vec![
        0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5,
    ];
    // 双面：每个三角形额外加入反向缠绕，确保任意视角可见
    let mut doubled = Vec::with_capacity(idx.len() * 2);
    for w in idx.chunks(3) {
        doubled.extend_from_slice(w);
        doubled.extend_from_slice(&[w[0], w[2], w[1]]);
    }
    (verts, doubled)
}

/// 返回从 `a` 到 `b`、半厚 `t` 的细长方体（已做双面处理）。
fn edge_box(a: [f64; 3], b: [f64; 3], t: f64) -> (Vec<[f64; 3]>, Vec<u32>) {
    let dir = vnorm(vsub(b, a));
    let up = if dir[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let u = vnorm(vcross(dir, up));
    let v = vnorm(vcross(dir, u));
    let corner = |base: [f64; 3], su: f64, sv: f64| -> [f64; 3] {
        vadd(base, vadd(vscale(u, su * t), vscale(v, sv * t)))
    };
    let verts: Vec<[f64; 3]> = vec![
        corner(a, -1.0, -1.0),
        corner(a, 1.0, -1.0),
        corner(a, 1.0, 1.0),
        corner(a, -1.0, 1.0),
        corner(b, -1.0, -1.0),
        corner(b, 1.0, -1.0),
        corner(b, 1.0, 1.0),
        corner(b, -1.0, 1.0),
    ];
    let idx: Vec<u32> = vec![
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3,
        3, 7, 4, 3, 4, 0,
    ];
    let mut doubled = Vec::with_capacity(idx.len() * 2);
    for w in idx.chunks(3) {
        doubled.extend_from_slice(w);
        doubled.extend_from_slice(&[w[0], w[2], w[1]]);
    }
    (verts, doubled)
}

// ── EngvisApp 实现 ──────────────────────────────────────────
struct ViewerApp {
    /// 持久化的 halfedge 网格（交互操作直接修改它）
    mesh: MeshStorage,
    /// 持久化的属性表（选择态存储于此）
    props: MeshProperties,
    /// 选择态句柄
    vsel: PropertyHandle<VertexSelected>,
    esel: PropertyHandle<HalfEdgeSelected>,
    fsel: PropertyHandle<FaceSelected>,
    /// 生成预设网格的参数
    operation: Operation,
    mesh_name: String,
    /// 拾取模式
    pick_mode: PickMode,
    /// 下一帧执行拾取（由 on_event 的鼠标按下触发）
    pending_pick: bool,
    /// 下一帧执行的拓扑操作
    pending_action: Option<EditAction>,
    /// 下一帧依据 operation 重新生成网格
    regen_pending: bool,
    /// 下一帧重置相机视角
    reset_view: bool,
    /// 标记需要重建 engvis 网格
    dirty: bool,
    /// 统计信息
    stats: String,
    /// 上一次操作结果提示
    message: String,
    /// 是否已初始化 CJK 字体
    fonts_initialized: bool,
    /// 四元数跟踪的相机朝向（用于解除 pitch 限制）
    orbit_quat: Quat,
    /// 上一帧光标位置（计算鼠标 delta）
    prev_cursor_x: f64,
    prev_cursor_y: f64,
    /// 左键是否正在拖拽旋转
    orbit_dragging: bool,
}

impl ViewerApp {
    fn new(operation: Operation) -> Self {
        // 初始先用占位，on_setup 中 regenerate 填充真实网格
        let (mesh, name) = generate_mesh(&operation);
        let mut props = MeshProperties::new();
        let vsel = add_vertex_selection(&mut props);
        let esel = add_halfedge_selection(&mut props);
        let fsel = add_face_selection(&mut props);
        let mut app = Self {
            mesh,
            props,
            vsel,
            esel,
            fsel,
            operation,
            mesh_name: name,
            pick_mode: PickMode::Face,
            pending_pick: false,
            pending_action: None,
            regen_pending: false,
            reset_view: false,
            dirty: true,
            stats: String::new(),
            message: "左键点击网格进行选择".into(),
            fonts_initialized: false,
            orbit_quat: Quat::IDENTITY,
            prev_cursor_x: 0.0,
            prev_cursor_y: 0.0,
            orbit_dragging: false,
        };
        app.update_stats();
        app
    }

    /// 依据 operation 重新生成基础网格，并重置选择态属性。
    fn regenerate(&mut self) {
        let (mesh, name) = generate_mesh(&self.operation);
        self.mesh = mesh;
        self.mesh_name = name;
        self.props = MeshProperties::new();
        self.vsel = add_vertex_selection(&mut self.props);
        self.esel = add_halfedge_selection(&mut self.props);
        self.fsel = add_face_selection(&mut self.props);
        self.message = format!("生成预设：{}", self.mesh_name);
        self.dirty = true;
        self.update_stats();
    }

    /// 清空所有选择态（拓扑操作后会令旧 handle 失效，故操作后统一清空）。
    fn clear_selection(&mut self) {
        clear_vertex_selection(&self.mesh, &mut self.props, self.vsel);
        clear_halfedge_selection(&self.mesh, &mut self.props, self.esel);
        clear_face_selection(&self.mesh, &mut self.props, self.fsel);
    }

    /// 网格包围盒对角线长度（用于标记几何尺寸归一化）。
    fn mesh_scale(&self) -> f64 {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for v in self.mesh.vertex_ids() {
            if let Some(p) = self.mesh.get_vertex(v).map(|x| x.position) {
                for i in 0..3 {
                    min[i] = min[i].min(p[i]);
                    max[i] = max[i].max(p[i]);
                }
            }
        }
        let d = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3)
    }

    /// 将屏幕光标转换为世界空间射线（origin, dir）。
    fn screen_to_ray(
        &self,
        camera: &OrbitCamera,
        cursor_x: f64,
        cursor_y: f64,
        vp: &engvis_core::ViewportRect,
    ) -> Option<([f64; 3], [f64; 3])> {
        let w = vp.max_x - vp.min_x;
        let h = vp.max_y - vp.min_y;
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        let ndc_x = ((cursor_x - vp.min_x) / w) * 2.0 - 1.0;
        // 屏幕 y 向下，NDC y 向上
        let ndc_y = 1.0 - ((cursor_y - vp.min_y) / h) * 2.0;

        let eye = camera.position();
        let forward = (camera.target - eye).normalize();
        let right = camera.right();
        let up = camera.up();
        let tan_half = (camera.fov_y * 0.5).tan();
        let aspect = camera.aspect_ratio;
        let dx = ndc_x as f32 * tan_half * aspect;
        let dy = ndc_y as f32 * tan_half;
        let dir = (forward + right * dx + up * dy).normalize();

        let origin = [eye.x as f64, eye.y as f64, eye.z as f64];
        let direction = [dir.x as f64, dir.y as f64, dir.z as f64];
        Some((origin, direction))
    }

    /// 射线与网格求最近交点（对每个面做扇形三角化后逐三角求交，
    /// 因此非三角面如挤出侧面也能正确命中）。
    fn nearest_face(&self, origin: &[f64; 3], dir: &[f64; 3]) -> Option<(FaceId, [f64; 3])> {
        let mut best: Option<(f64, FaceId, [f64; 3])> = None;
        for f in self.mesh.face_ids() {
            let verts: Vec<VertexId> = FaceVertices::new(&self.mesh, f).collect();
            if verts.len() < 3 {
                continue;
            }
            for i in 1..verts.len() - 1 {
                let a = self.mesh.get_vertex(verts[0])?.position;
                let b = self.mesh.get_vertex(verts[i])?.position;
                let c = self.mesh.get_vertex(verts[i + 1])?.position;
                if let Some((t, _, _)) = ray_triangle_intersection(*origin, *dir, a, b, c) {
                    let p = [
                        origin[0] + t * dir[0],
                        origin[1] + t * dir[1],
                        origin[2] + t * dir[2],
                    ];
                    match best {
                        Some((bt, _, _)) if t < bt => best = Some((t, f, p)),
                        None => best = Some((t, f, p)),
                        _ => {}
                    }
                }
            }
        }
        best.map(|(_, f, p)| (f, p))
    }

    /// 按当前拾取模式选中最近元素。
    fn try_pick(
        &mut self,
        camera: &OrbitCamera,
        cursor_x: f64,
        cursor_y: f64,
        vp: &engvis_core::ViewportRect,
    ) {
        let Some((origin, dir)) = self.screen_to_ray(camera, cursor_x, cursor_y, vp) else {
            return;
        };
        let Some((face, hp)) = self.nearest_face(&origin, &dir) else {
            self.message = "未命中任何面".into();
            return;
        };
        match self.pick_mode {
            PickMode::Face => {
                select_face(&mut self.props, self.fsel, face);
                self.message = format!("选中面 {:?}", face);
            }
            PickMode::Vertex => {
                let verts: Vec<VertexId> = FaceVertices::new(&self.mesh, face).collect();
                let mut best: Option<(f64, VertexId)> = None;
                for v in verts {
                    if let Some(p) = self.mesh.get_vertex(v).map(|x| x.position) {
                        let d = vdot(vsub(p, hp), vsub(p, hp));
                        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                            best = Some((d, v));
                        }
                    }
                }
                if let Some((_, v)) = best {
                    select_vertex(&mut self.props, self.vsel, v);
                    self.message = format!("选中顶点 {:?}", v);
                }
            }
            PickMode::Edge => {
                let hes: Vec<HalfEdgeId> = FaceHalfEdges::new(&self.mesh, face).collect();
                let mut best: Option<(f64, HalfEdgeId)> = None;
                for he in hes {
                    let h = match self.mesh.get_halfedge(he) {
                        Some(h) => h,
                        None => continue,
                    };
                    let prev = match h.prev.and_then(|p| self.mesh.get_halfedge(p)) {
                        Some(p) => p,
                        None => continue,
                    };
                    let a = match self.mesh.get_vertex(h.vertex).map(|x| x.position) {
                        Some(p) => p,
                        None => continue,
                    };
                    let b = match self.mesh.get_vertex(prev.vertex).map(|x| x.position) {
                        Some(p) => p,
                        None => continue,
                    };
                    let d = dist_point_seg_sq(hp, a, b);
                    if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                        best = Some((d, he));
                    }
                }
                if let Some((_, he)) = best {
                    select_edge(&self.mesh, &mut self.props, self.esel, he);
                    self.message = format!("选中边 {:?}", he);
                }
            }
        }
        self.dirty = true;
        self.update_stats();
    }

    /// 将选中元素应用拓扑操作。
    fn apply_edit(&mut self, action: EditAction) {
        match action {
            EditAction::Split => {
                let he = selected_edge_ids(&self.mesh, &self.props, self.esel).next();
                match he {
                    Some(he) => match split_edge(&mut self.mesh, he) {
                        Ok(_) => self.message = "已分裂边 ✓".into(),
                        Err(e) => self.message = format!("分裂失败：{:?}", e),
                    },
                    None => self.message = "未选中边（请先用『边』模式点击）".into(),
                }
            }
            EditAction::Flip => {
                let he = selected_edge_ids(&self.mesh, &self.props, self.esel).next();
                match he {
                    Some(he) => match flip_edge(&mut self.mesh, he) {
                        Ok(_) => self.message = "已翻转边 ✓".into(),
                        Err(e) => self.message = format!("翻转失败：{:?}", e),
                    },
                    None => self.message = "未选中边（请先用『边』模式点击）".into(),
                }
            }
            EditAction::Collapse => {
                let he = selected_edge_ids(&self.mesh, &self.props, self.esel).next();
                match he {
                    Some(he) => match collapse_edge(&mut self.mesh, he) {
                        Ok(_) => self.message = "已塌缩边 ✓".into(),
                        Err(e) => self.message = format!("塌缩失败：{:?}", e),
                    },
                    None => self.message = "未选中边（请先用『边』模式点击）".into(),
                }
            }
            EditAction::Extrude => {
                let f = selected_face_ids(&self.mesh, &self.props, self.fsel).next();
                match f {
                    Some(f) => {
                        let scale = self.mesh_scale() * 0.3;
                        let n = face_normal_of(&self.mesh, f);
                        let off = vscale(n, scale);
                        match extrude_face(&mut self.mesh, f, off) {
                            Ok(_) => self.message = "已挤出面 ✓".into(),
                            Err(e) => self.message = format!("挤出失败：{:?}", e),
                        }
                    }
                    None => self.message = "未选中面（请先用『面』模式点击）".into(),
                }
            }
        }
        // 操作后旧选择 handle 可能失效，统一清空并重建
        self.clear_selection();
        self.dirty = true;
        self.update_stats();
    }

    /// 构建场景内容：(网格, 材质, 节点)。
    /// 仅当某类几何非空时才加入对应网格，避免渲染器创建空索引缓冲导致 panic。
    /// 普通面与选中面共用同一套顶点位置，按选择态分流到不同材质。
    fn build_scene_parts(&self) -> (Vec<EngvisMesh>, Vec<PbrMaterial>, Vec<SceneNode>) {
        let scale = self.mesh_scale();
        let r = scale * 0.02; // 顶点八面体半径
        let t = scale * 0.012; // 边长方体半厚

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut vmap: HashMap<VertexId, u32> = HashMap::new();
        let mut next = 0u32;
        for v in self.mesh.vertex_ids() {
            if let Some(p) = self.mesh.get_vertex(v).map(|x| x.position) {
                positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
                vmap.insert(v, next);
                next += 1;
            }
        }

        let mut normal_idx: Vec<u32> = Vec::new();
        let mut sel_idx: Vec<u32> = Vec::new();
        for f in self.mesh.face_ids() {
            let verts: Vec<VertexId> = FaceVertices::new(&self.mesh, f).collect();
            if verts.len() < 3 {
                continue;
            }
            for i in 1..verts.len() - 1 {
                let tri = [verts[0], verts[i], verts[i + 1]];
                let mut t_idx = [0u32; 3];
                let mut ok = true;
                for (k, vid) in tri.iter().enumerate() {
                    match vmap.get(vid) {
                        Some(&idx) => t_idx[k] = idx,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    continue;
                }
                if is_face_selected(&self.props, self.fsel, f) {
                    sel_idx.extend_from_slice(&t_idx);
                } else {
                    normal_idx.extend_from_slice(&t_idx);
                }
            }
        }

        // 标记几何：选中的顶点 → 八面体；选中的边 → 细长方体
        let mut mk_pos: Vec<[f32; 3]> = Vec::new();
        let mut mk_idx: Vec<u32> = Vec::new();
        for v in selected_vertex_ids(&self.mesh, &self.props, self.vsel) {
            if let Some(p) = self.mesh.get_vertex(v).map(|x| x.position) {
                let (vp, vi) = octahedron(p, r);
                let base = mk_pos.len() as u32;
                mk_pos.extend(
                    vp.into_iter()
                        .map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
                );
                for i in vi {
                    mk_idx.push(i + base);
                }
            }
        }
        for he in selected_edge_ids(&self.mesh, &self.props, self.esel) {
            let h = match self.mesh.get_halfedge(he) {
                Some(h) => h,
                None => continue,
            };
            let prev = match h.prev.and_then(|p| self.mesh.get_halfedge(p)) {
                Some(p) => p,
                None => continue,
            };
            let a = match self.mesh.get_vertex(h.vertex).map(|x| x.position) {
                Some(p) => p,
                None => continue,
            };
            let b = match self.mesh.get_vertex(prev.vertex).map(|x| x.position) {
                Some(p) => p,
                None => continue,
            };
            let (vp, vi) = edge_box(a, b, t);
            let base = mk_pos.len() as u32;
            mk_pos.extend(
                vp.into_iter()
                    .map(|x| [x[0] as f32, x[1] as f32, x[2] as f32]),
            );
            for i in vi {
                mk_idx.push(i + base);
            }
        }
        // 材质（每类几何一个），按加入顺序分配 material_index。
        // 注意 from_triangles 生成的子网格 material_index 固定为 0，
        // 必须显式改写，否则选中面/标记会错误地复用 surface 材质。
        let surface_mat = PbrMaterial {
            name: "surface".into(),
            albedo: [0.25, 0.65, 0.90, 1.0],
            ..Default::default()
        };
        let selected_mat = PbrMaterial {
            name: "selected".into(),
            albedo: [0.95, 0.45, 0.10, 1.0],
            ..Default::default()
        };
        let marker_mat = PbrMaterial {
            name: "marker".into(),
            albedo: [1.0, 0.85, 0.0, 1.0],
            ..Default::default()
        };

        let mut meshes: Vec<EngvisMesh> = Vec::new();
        let mut materials: Vec<PbrMaterial> = Vec::new();
        let mut nodes: Vec<SceneNode> = Vec::new();

        // 动态加入：只有非空的几何才生成网格，
        // 否则渲染器会因空索引缓冲而无法创建 GPU buffer（panic）。
        let mut add = |name: &str, mut mesh: EngvisMesh, mat: PbrMaterial| {
            let mi = meshes.len();
            let mat_idx = materials.len();
            if let Some(sm) = mesh.sub_meshes.first_mut() {
                sm.material_index = mat_idx;
            }
            meshes.push(mesh);
            materials.push(mat);
            nodes.push(SceneNode {
                name: name.to_string(),
                local_transform: Affine3A::IDENTITY,
                mesh_index: Some(mi),
                children: Vec::new(),
                visible: true,
            });
        };

        if !normal_idx.is_empty() {
            add(
                "surface",
                EngvisMesh::from_triangles(&self.mesh_name, &positions, &normal_idx),
                surface_mat,
            );
        }
        if !sel_idx.is_empty() {
            add(
                "selected",
                EngvisMesh::from_triangles("selected_faces", &positions, &sel_idx),
                selected_mat,
            );
        }
        if !mk_idx.is_empty() {
            add(
                "markers",
                EngvisMesh::from_triangles("markers", &mk_pos, &mk_idx),
                marker_mat,
            );
        }

        (meshes, materials, nodes)
    }

    /// 重建 engvis 网格并标记场景脏。
    fn rebuild_engvis(&mut self, frame: &mut FrameCtx) {
        self.update_stats();
        let (meshes, materials, nodes) = self.build_scene_parts();
        frame.scene.meshes = meshes;
        frame.scene.materials = materials;
        frame.scene.nodes = nodes;
        *frame.scene_dirty = true;
    }

    fn update_stats(&mut self) {
        let v = self.mesh.vertex_count();
        let f = self.mesh.face_count();
        let e = self.mesh.halfedge_count() / 2;
        let euler = v as i64 - e as i64 + f as i64;
        let sv = selected_vertex_ids(&self.mesh, &self.props, self.vsel).count();
        let se = selected_edge_ids(&self.mesh, &self.props, self.esel).count();
        let sf = selected_face_ids(&self.mesh, &self.props, self.fsel).count();
        self.stats = format!(
            "顶点 V   = {}\n边 E     = {}\n面 F     = {}\nEuler    = {}\n选中 顶点/边/面 = {} / {} / {}",
            v, e, f, euler, sv, se, sf
        );
    }
}

/// 计算面法向（用于挤出方向）。
fn face_normal_of(mesh: &MeshStorage, f: FaceId) -> [f64; 3] {
    let verts: Vec<VertexId> = FaceVertices::new(mesh, f).collect();
    if verts.len() < 3 {
        return [0.0, 0.0, 1.0];
    }
    let a = mesh.get_vertex(verts[0]).unwrap().position;
    let b = mesh.get_vertex(verts[1]).unwrap().position;
    let c = mesh.get_vertex(verts[2]).unwrap().position;
    let n = vnorm(vcross(vsub(b, a), vsub(c, a)));
    if vlen(n) < 1e-9 { [0.0, 0.0, 1.0] } else { n }
}

impl EngvisApp for ViewerApp {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: format!("halfedge viewer — {}", self.operation.label()),
            width: 1280,
            height: 800,
            sample_count: 4,
            ..Default::default()
        }
    }

    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        self.regenerate();
        let (meshes, materials, nodes) = self.build_scene_parts();

        Scene {
            meshes,
            materials,
            nodes,
            ..Default::default()
        }
    }

    fn on_ready(&mut self, scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_scene(scene);
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        // 首次调用时加载 CJK 字体，否则中文字符会显示为方框
        if !self.fonts_initialized {
            self.fonts_initialized = true;
            setup_cjk_fonts(egui_ctx);
        }

        egui::SidePanel::right("control_panel")
            .default_width(300.0)
            .show(egui_ctx, |ui| {
                ui.heading("halfedge viewer");
                ui.label("左键点击网格选择 · 左键拖拽旋转 · 滚轮缩放");
                ui.separator();

                // ── 生成预设 ──
                ui.label("预设网格");
                let mut new_op: Option<Operation> = None;
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(
                            matches!(self.operation, Operation::Icosphere(_)),
                            "icosphere",
                        )
                        .clicked()
                    {
                        new_op = Some(Operation::Icosphere(1));
                    }
                    if ui
                        .selectable_label(
                            matches!(self.operation, Operation::Subdivision(_, _)),
                            "subdiv",
                        )
                        .clicked()
                    {
                        new_op = Some(Operation::Subdivision(SubdivType::Loop, 1));
                    }
                    if ui
                        .selectable_label(matches!(self.operation, Operation::Extrude), "extrude")
                        .clicked()
                    {
                        new_op = Some(Operation::Extrude);
                    }
                    if ui
                        .selectable_label(matches!(self.operation, Operation::Smooth(_)), "smooth")
                        .clicked()
                    {
                        new_op = Some(Operation::Smooth(10));
                    }
                });
                if let Some(op) = new_op {
                    self.operation = op;
                    self.regen_pending = true;
                }

                ui.separator();

                // ── 预设参数（修改会重新生成网格） ──
                match self.operation {
                    Operation::Icosphere(ref mut n) => {
                        ui.label("细分级别 n");
                        if ui.add(egui::Slider::new(n, 0..=4).text("n")).changed() {
                            self.regen_pending = true;
                        }
                    }
                    Operation::Subdivision(ref mut t, ref mut n) => {
                        ui.label("细分类型");
                        egui::ComboBox::from_label("type")
                            .selected_text(format!("{:?}", t))
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(t, SubdivType::Loop, "Loop").clicked() {
                                    self.regen_pending = true;
                                }
                                if ui
                                    .selectable_value(t, SubdivType::CatmullClark, "CatmullClark")
                                    .clicked()
                                {
                                    self.regen_pending = true;
                                }
                                if ui.selectable_value(t, SubdivType::Sqrt3, "Sqrt3").clicked() {
                                    self.regen_pending = true;
                                }
                            });
                        ui.label("迭代次数 n");
                        if ui.add(egui::Slider::new(n, 1..=4).text("n")).changed() {
                            self.regen_pending = true;
                        }
                    }
                    Operation::Smooth(ref mut iters) => {
                        ui.label("平滑迭代次数");
                        if ui
                            .add(egui::Slider::new(iters, 0..=50).text("iters"))
                            .changed()
                        {
                            self.regen_pending = true;
                        }
                    }
                    Operation::Extrude => {
                        ui.label("（挤压 cube 顶面 +Y 方向）");
                    }
                }

                ui.separator();

                // ── 拾取模式 ──
                ui.label("拾取模式");
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(
                            self.pick_mode == PickMode::Vertex,
                            PickMode::Vertex.label(),
                        )
                        .clicked()
                    {
                        self.pick_mode = PickMode::Vertex;
                    }
                    if ui
                        .selectable_label(self.pick_mode == PickMode::Edge, PickMode::Edge.label())
                        .clicked()
                    {
                        self.pick_mode = PickMode::Edge;
                    }
                    if ui
                        .selectable_label(self.pick_mode == PickMode::Face, PickMode::Face.label())
                        .clicked()
                    {
                        self.pick_mode = PickMode::Face;
                    }
                });

                ui.separator();

                // ── 应用拓扑操作 ──
                ui.label("应用到选中元素");
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Split 边").clicked() {
                        self.pending_action = Some(EditAction::Split);
                    }
                    if ui.button("Flip 边").clicked() {
                        self.pending_action = Some(EditAction::Flip);
                    }
                    if ui.button("Collapse 边").clicked() {
                        self.pending_action = Some(EditAction::Collapse);
                    }
                    if ui.button("Extrude 面").clicked() {
                        self.pending_action = Some(EditAction::Extrude);
                    }
                });
                if ui.button("清空选择").clicked() {
                    self.clear_selection();
                    self.dirty = true;
                    self.update_stats();
                    self.message = "已清空选择".into();
                }
                if ui.button("重置视角 (Fit)").clicked() {
                    self.reset_view = true;
                }

                ui.separator();

                // ── 统计 ──
                ui.heading("网格统计");
                ui.label(&self.stats);

                ui.separator();
                ui.heading("提示");
                ui.label(&self.message);

                ui.separator();

                // ── 渲染选项 ──
                ui.heading("渲染选项");
                ui.checkbox(&mut frame.render_state.show_surface, "显示曲面");
                ui.checkbox(&mut frame.render_state.show_grid, "显示网格");
                ui.checkbox(&mut frame.render_state.edge_opts.enabled, "显示线框");
                ui.checkbox(&mut frame.render_state.vertex_opts.enabled, "显示顶点");

                ui.separator();
                ui.label(format!("FPS: {:.1}", frame.fps));
            });
    }

    fn on_frame(&mut self, frame: &mut FrameCtx) {
        // 重新生成预设网格
        if self.regen_pending {
            self.regenerate();
            self.regen_pending = false;
            self.dirty = true;
        }

        // 执行拾取（鼠标左键已在 on_event 中置位）
        if self.pending_pick {
            self.pending_pick = false;
            // 点在 UI 面板或视口外时不拾取
            if !frame.egui_wants_pointer && frame.viewport.contains(frame.cursor_x, frame.cursor_y)
            {
                self.try_pick(frame.camera, frame.cursor_x, frame.cursor_y, frame.viewport);
            }
        }

        // 执行拓扑操作
        if let Some(action) = self.pending_action.take() {
            self.apply_edit(action);
        }

        // 重置视角
        if self.reset_view {
            frame.camera.fit_to_scene(frame.scene);
            self.reset_view = false;
        }

        // 重建 engvis 网格
        if self.dirty {
            self.rebuild_engvis(frame);
            self.dirty = false;
        }

        // ── 解除 pitch 限制的四元数轨道旋转 ──
        // engvis-core 的 OrbitCamera 使用 yaw/pitch 且强制 clamp pitch
        // 到 ±89.4°。这里用四元数跟踪真实朝向，直接覆写 camera.yaw/pitch。
        //
        // 注意：InputState.apply_to_camera() 在 on_frame 之后也会 orbit，
        // 因此 yaw 需扣减同样的 delta 避免双倍旋转。pitch 直接覆写（即使
        // InputState 随后 clamp，下一帧会重新从四元数推导，等效无 clamp）。
        let dx = (frame.cursor_x - self.prev_cursor_x) as f32;
        let dy = (frame.cursor_y - self.prev_cursor_y) as f32;
        self.prev_cursor_x = frame.cursor_x;
        self.prev_cursor_y = frame.cursor_y;

        if self.orbit_dragging
            && !frame.egui_wants_pointer
            && frame.viewport.contains(frame.cursor_x, frame.cursor_y)
        {
            let sens = 0.005;
            let delta_yaw = -dx * sens;
            let delta_pitch = -dy * sens;

            // 用四元数累积旋转（无万向节死锁，无 pitch 限制）
            let rot_yaw = Quat::from_rotation_y(delta_yaw);
            let rot_pitch = Quat::from_rotation_x(delta_pitch);
            self.orbit_quat = (rot_yaw * self.orbit_quat * rot_pitch).normalize();

            // 从四元数推导 yaw/pitch
            let forward = self.orbit_quat * Vec3::Z;
            let derived_pitch = forward.y.asin();
            let derived_yaw = forward.x.atan2(forward.z);

            // yaw: 扣减 delta_yaw 因为 InputState 即将再 orbit(delta_yaw, …)
            frame.camera.yaw = derived_yaw - delta_yaw;
            // pitch: 直接覆写；InputState 随后 clamp，但下次 frame 重新从
            // 四元数推导，所以持续拖拽期间 pitch 始终不受 clamp 影响。
            frame.camera.pitch = derived_pitch;
        } else {
            // 未拖拽时从 camera 同步四元数（响应 fit_to_scene 等操作）
            let forward = Vec3::new(
                frame.camera.pitch.cos() * frame.camera.yaw.sin(),
                frame.camera.pitch.sin(),
                frame.camera.pitch.cos() * frame.camera.yaw.cos(),
            )
            .normalize();
            self.orbit_quat = Quat::from_rotation_arc(Vec3::Z, forward);
        }
    }

    fn on_event(&mut self, event: &WindowEvent) -> EventHandling {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => match state {
                ElementState::Pressed => {
                    self.orbit_dragging = true;
                    self.pending_pick = true;
                }
                ElementState::Released => {
                    self.orbit_dragging = false;
                }
            },
            WindowEvent::KeyboardInput { event: k, .. }
                if k.state == ElementState::Pressed
                    && let Key::Character(c) = &k.logical_key =>
            {
                match c.as_str() {
                    "1" => self.pick_mode = PickMode::Vertex,
                    "2" => self.pick_mode = PickMode::Edge,
                    "3" => self.pick_mode = PickMode::Face,
                    _ => {}
                }
            }
            _ => {}
        }
        EventHandling::Default
    }
}

/// 注册内嵌的 CJK 子集字体到 egui 上下文中。
///
/// 字体由构建时 `pyftsubset` 从 Hiragino Sans GB 裁剪生成，
/// 仅包含 viewer UI 中使用的 ~418 个字符（~204KB）。
fn setup_cjk_fonts(ctx: &egui::Context) {
    let font_data = include_bytes!("../assets/cjk_font.ttf");
    let mut fonts = egui::FontDefinitions::default();
    let font_name = "CJK".to_owned();
    fonts.font_data.insert(
        font_name.clone(),
        std::sync::Arc::new(
            egui::FontData::from_static(font_data).tweak(egui::FontTweak {
                scale: 0.95,
                ..Default::default()
            }),
        ),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, font_name.clone());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, font_name);
    ctx.set_fonts(fonts);
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let operation = Operation::from_args(&args);

    println!("=== halfedge engvis viewer ===");
    println!("预设: {}", operation.label());
    println!("交互: 左键点击网格选择 (1=顶点 2=边 3=面)，右侧面板应用拓扑操作");
    println!();

    engvis_renderer::run(ViewerApp::new(operation));
}
