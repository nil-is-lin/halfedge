//! 示例：使用 engvis-renderer 交互式可视化 halfedge 网格操作
//!
//! 运行：
//!   cargo run --example engvis_viewer --                       # 默认 icosphere
//!   cargo run --example engvis_viewer -- icosphere 2           # 2 级细分 icosphere
//!   cargo run --example engvis_viewer -- subdivision loop 2    # Loop 细分 2 次
//!   cargo run --example engvis_viewer -- extrude               # 挤压面
//!   cargo run --example engvis_viewer -- smooth 10             # Laplacian 平滑
//!   cargo run --example engvis_viewer -- topology split        # 边分裂
//!
//! 也可在窗口右侧 UI 面板实时切换操作与参数。

use std::env;

use engvis_core::{
    camera::OrbitCamera, material::PbrMaterial, mesh::Mesh as EngvisMesh, scene::Scene,
};
use engvis_renderer::{AppCtx, EngvisApp, EventHandling, FrameCtx, RunConfig};

use halfedge::{
    FaceId, MeshStorage, build_cube, build_icosphere, build_uv_sphere, catmull_clark_subdivide,
    collapse_edge, extrude_face, flip_edge, laplacian_smooth_mesh, loop_subdivide,
    mesh_to_vertex_index_buffers, split_edge, sqrt3_subdivide,
};

// ── 操作类型 ──────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Operation {
    Icosphere(u32),
    Subdivision(SubdivType, u32),
    Extrude,
    Smooth(u32),
    Topology(TopoOp),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SubdivType {
    Loop,
    CatmullClark,
    Sqrt3,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TopoOp {
    Split,
    Flip,
    Collapse,
}

impl Operation {
    /// 默认操作
    fn default() -> Self {
        Operation::Icosphere(1)
    }

    /// 从命令行参数解析
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
            "topology" => {
                let t = match args.get(1).map(|s| s.as_str()) {
                    Some("flip") => TopoOp::Flip,
                    Some("collapse") => TopoOp::Collapse,
                    _ => TopoOp::Split,
                };
                Operation::Topology(t)
            }
            _ => Self::default(),
        }
    }

    fn label(&self) -> String {
        match self {
            Operation::Icosphere(n) => format!("Icosphere (n={})", n),
            Operation::Subdivision(t, n) => format!("{:?} Subdivision x{}", t, n),
            Operation::Extrude => "Extrude Face".into(),
            Operation::Smooth(i) => format!("Laplacian Smooth (iters={})", i),
            Operation::Topology(t) => format!("Topology {:?}", t),
        }
    }
}

// ── halfedge → engvis 转换 ────────────────────────────────────
/// 将 halfedge 的 MeshStorage 转换为 engvis 的 GPU-ready Mesh。
///
/// 利用 `mesh_to_vertex_index_buffers` 导出顶点+索引，
/// 再由 `Mesh::from_triangles` 自动去重、修复绕向、计算平滑法向。
fn halfedge_to_engvis(mesh: &MeshStorage, name: &str) -> EngvisMesh {
    let (positions, indices) = mesh_to_vertex_index_buffers(mesh);
    EngvisMesh::from_triangles(name, &positions, &indices)
}

// ── 根据 operation 生成 halfedge mesh ─────────────────────────
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
            // 收集所有 face_id 后再挑选顶面（避免迭代器与闭包同时借用 mesh）
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
            // 低分辨率球，平滑后可见明显光滑效果
            let mut mesh = build_uv_sphere(1.0, 8, 6);
            laplacian_smooth_mesh(&mut mesh, 0.5, *iters as usize);
            (mesh, format!("smoothed_{}", iters))
        }
        Operation::Topology(t) => {
            let mut mesh = build_icosphere(0);
            // 先取出第一个半边 ID（断开迭代器对 mesh 的不可变借用）
            let first_he = mesh.halfedge_ids().next();
            if let Some(he_id) = first_he {
                match t {
                    TopoOp::Split => {
                        let _ = split_edge(&mut mesh, he_id);
                    }
                    TopoOp::Flip => {
                        let _ = flip_edge(&mut mesh, he_id);
                    }
                    TopoOp::Collapse => {
                        let _ = collapse_edge(&mut mesh, he_id);
                    }
                }
            }
            (mesh, format!("topo_{:?}", t))
        }
    }
}

/// 计算面重心的 z 坐标（用于挑选顶面）
fn face_centroid_z(mesh: &MeshStorage, face: FaceId) -> f64 {
    use halfedge::traversal::FaceHalfEdges;
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

// ── EngvisApp 实现 ────────────────────────────────────────────
struct ViewerApp {
    operation: Operation,
    /// 标记需要重新生成 mesh（UI 修改参数后置 true）
    dirty: bool,
    /// 控制台打印的统计信息
    stats: String,
}

impl ViewerApp {
    fn new(operation: Operation) -> Self {
        Self {
            operation,
            dirty: true,
            stats: String::new(),
        }
    }

    /// 重新生成 mesh 并返回 engvis Mesh + 材质
    fn rebuild(&mut self) -> (EngvisMesh, PbrMaterial, String) {
        let (mesh, name) = generate_mesh(&self.operation);
        let v = mesh.vertex_count();
        let f = mesh.face_count();
        let e = mesh.halfedge_count() / 2;
        let euler = v as i64 - e as i64 + f as i64;
        self.stats = format!(
            "顶点 V = {}\n边 E = {}\n面 F = {}\nEuler V-E+F = {}",
            v, e, f, euler
        );
        let engvis_mesh = halfedge_to_engvis(&mesh, &name);
        let material = PbrMaterial {
            name: "halfedge_default".into(),
            albedo: [0.25, 0.65, 0.90, 1.0],
            ..Default::default()
        };
        (engvis_mesh, material, name)
    }
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
        let (mesh, material, name) = self.rebuild();
        self.dirty = false;
        Scene::single_mesh(name, mesh, material)
    }

    fn on_ready(&mut self, scene: &Scene, camera: &mut OrbitCamera) {
        camera.fit_to_scene(scene);
    }

    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx) {
        egui::SidePanel::right("control_panel")
            .default_width(280.0)
            .show(egui_ctx, |ui| {
                ui.heading("halfedge viewer");
                ui.separator();

                // ── 操作类型选择 ──
                ui.label("操作类型");
                let mut new_op = None;
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
                        "subdivision",
                    )
                    .clicked()
                {
                    new_op = Some(Operation::Subdivision(SubdivType::Loop, 1));
                }
                if ui
                    .selectable_label(matches!(self.operation, Operation::Extrude), "extrude face")
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
                if ui
                    .selectable_label(matches!(self.operation, Operation::Topology(_)), "topology")
                    .clicked()
                {
                    new_op = Some(Operation::Topology(TopoOp::Split));
                }

                // 应用操作切换
                if let Some(op) = new_op {
                    self.operation = op;
                    self.dirty = true;
                }

                ui.separator();

                // ── 参数调整 ──
                match self.operation {
                    Operation::Icosphere(ref mut n) => {
                        ui.label("细分级别 n");
                        if ui.add(egui::Slider::new(n, 0..=4).text("n")).changed() {
                            self.dirty = true;
                        }
                    }
                    Operation::Subdivision(ref mut t, ref mut n) => {
                        ui.label("细分类型");
                        egui::ComboBox::from_label("type")
                            .selected_text(format!("{:?}", t))
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(t, SubdivType::Loop, "Loop").clicked() {
                                    self.dirty = true;
                                }
                                if ui
                                    .selectable_value(t, SubdivType::CatmullClark, "CatmullClark")
                                    .clicked()
                                {
                                    self.dirty = true;
                                }
                                if ui.selectable_value(t, SubdivType::Sqrt3, "Sqrt3").clicked() {
                                    self.dirty = true;
                                }
                            });
                        ui.label("迭代次数 n");
                        if ui.add(egui::Slider::new(n, 1..=4).text("n")).changed() {
                            self.dirty = true;
                        }
                    }
                    Operation::Smooth(ref mut iters) => {
                        ui.label("平滑迭代次数");
                        if ui
                            .add(egui::Slider::new(iters, 0..=50).text("iters"))
                            .changed()
                        {
                            self.dirty = true;
                        }
                    }
                    Operation::Topology(ref mut t) => {
                        ui.label("拓扑操作");
                        egui::ComboBox::from_label("op")
                            .selected_text(format!("{:?}", t))
                            .show_ui(ui, |ui| {
                                if ui.selectable_value(t, TopoOp::Split, "Split").clicked() {
                                    self.dirty = true;
                                }
                                if ui.selectable_value(t, TopoOp::Flip, "Flip").clicked() {
                                    self.dirty = true;
                                }
                                if ui
                                    .selectable_value(t, TopoOp::Collapse, "Collapse")
                                    .clicked()
                                {
                                    self.dirty = true;
                                }
                            });
                    }
                    Operation::Extrude => {
                        ui.label("（无参数，挤压 cube 顶面 +Y 方向）");
                    }
                }

                ui.separator();

                // ── 统计信息 ──
                ui.heading("网格统计");
                ui.label(&self.stats);

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
        // 若 UI 修改了参数，重新生成 mesh
        if self.dirty {
            let (mesh, material, name) = self.rebuild();
            // 替换场景中的第一个 mesh
            if let Some(first_mesh) = frame.scene.meshes.first_mut() {
                *first_mesh = mesh;
            } else {
                frame.scene.meshes.push(mesh);
            }
            // 更新材质
            if let Some(first_mat) = frame.scene.materials.first_mut() {
                *first_mat = material;
            }
            // 更新节点名
            if let Some(node) = frame.scene.nodes.first_mut() {
                node.name = name;
            }
            *frame.scene_dirty = true;
            self.dirty = false;
        }
    }

    fn on_event(&mut self, _event: &winit::event::WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let operation = Operation::from_args(&args);

    // 控制台打印初始信息
    println!("=== halfedge engvis viewer ===");
    println!("操作: {}", operation.label());
    println!("提示: 在窗口右侧 UI 面板实时切换操作与参数");
    println!();

    engvis_renderer::run(ViewerApp::new(operation));
}
