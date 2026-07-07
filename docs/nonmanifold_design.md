# 非流形（Non-manifold）支持设计草案

> 状态：草案 / 规划中（Roadmap P1）
> 目标：在保持现有 2-流形 API 兼容的前提下，支持非流形边/顶点

## 1. 背景

当前 `halfedge` 的数据结构（`Vertex` / `HalfEdge` / `Face` + `SlotMap`）严格假设
**2-流形（2-manifold）** 拓扑：

- 每条无向边最多被两个面共享（`twin` 存在性由此推断）；
- 每个顶点的邻域是一个或多个闭合扇形（star 是圆盘）；
- `validate_mesh` 的「流形约束」检查强制上述不变式。

这使得以下输入**无法表达**：

- **非流形边**：一条边被 3+ 个面共享（如两个四面体粘合成一个共享边）；
- **非流形顶点**：顶点的链接（link）不是单个圆盘（如三个面共享一个顶点，呈 T 形）；
- **自交 / 内部面**：模型的内部有面，而非仅边界表面。

CGAL 通过 `CGAL::SM_Check` / `Non_manifold` 标签与 `Polyhedron_3` 的不同
`Items` 策略支持非流形；OpenMesh 默认不支持，需特殊配置。

## 2. 方案对比

### 方案 A：Radial-Edge（径向边）

每条无向边维护一个**辐射环（radial cycle）**，列出所有以该边为边界的面，
而非仅 `twin` 一个。

- 数据结构：`RadialEdge { vertices: [V; 2], faces: Vec<F>, edges: Vec<HE> }`
- 优点：天然支持非流形，面数不限
- 缺点：完全替换半边模型，破坏现有 50+ 文件的 API；遍历语义改变

### 方案 B：NMEdge（非流形半边，Weiler 1994）

在半边基础上，允许一个半边有多个 `twin`（存储为 `Vec<HalfEdgeId>` 或
独立的 twin 邻接表），其余 next/prev/face 不变。

- 数据结构：`HalfEdge { vertex, next, prev, face, twins: Vec<HalfEdgeId> }`
- 优点：保留半边遍历语义（next/prev/face 不变），仅放宽 twin
- 缺点：`twins` 为 Vec 带来内存开销；遍历需区分「单独 twin」与「多个 twin」

### 方案 C：动态切换（推荐起点）

维持现有 2-流形 `HalfEdge`（单一 `twin: Option<HalfEdgeId>`）作为默认，
在 `feature = "nonmanifold"` 下切换为扩展结构：

```rust
#[cfg(not(feature = "nonmanifold"))]
pub struct HalfEdge { vertex, twin: Option<HalfEdgeId>, next, prev, face }

#[cfg(feature = "nonmanifold")]
pub struct HalfEdge {
    vertex: VertexId,
    twins: SmallVec<[HalfEdgeId; 1]>,  // 1 个时退化为单一 twin 语义
    next: Option<HalfEdgeId>,
    prev: Option<HalfEdgeId>,
    face: Option<FaceId>,
}
```

- `MeshStorage` 内部用 `enum EdgeRepr` 或泛型参数选择
- 公共 API（`halfedge_to`、`twin` 访问器）保持同名，内部 `match` 分派
- 默认（无 feature）构建的网格零开销；启用 feature 后获得非流形能力

## 3. 推荐路线

1. **Phase 1（feature 骨架）**：添加 `nonmanifold` feature flag；
   `HalfEdge` 在 feature 下用 `SmallVec<[HalfEdgeId; 1]>` 替代 `Option<HalfEdgeId>`；
   所有 `twin` 访问走统一访问器（如 `fn twins(&self) -> &[HalfEdgeId]`）。
2. **Phase 2（构建器适配）**：`io` 解析非流形 OBJ/PLY 时不强制 2-流形；
   `add_triangle` / builder 支持共享边多面。
3. **Phase 3（算法分层）**：
   - 兼容非流形的算法（遍历、属性、I/O）无条件支持；
   - 假设 2-流形的算法（细分、参数化、测地线、布尔）在检测到非流形时
     返回 `Err(NonManifold)` 或自动抽取 2-流形分量处理。
4. **Phase 4（validate 调整）**：`check_topology` 新增 `allow_nonmanifold` 参数；
   流形约束检查变为「若非流形 feature 关闭则强制」。

## 4. 风险与权衡

| 风险 | 缓解 |
|------|------|
| `SmallVec` 依赖引入 | 仅在 `nonmanifold` feature 下启用；默认构建无额外依赖 |
| 50+ 文件 `twin` 访问修改 | 统一访问器封装，调用点改动小 |
| 算法正确性（非流形下） | Phase 3 分层，非流形下明确错误而非静默错误 |
| 性能回归（默认构建） | 默认路径保持 `Option<HalfEdgeId>`，零开销 |

## 5. 结论

采用**方案 C（动态切换 + feature 门控）** 风险最低：

- 默认构建保持当前高性能、零额外依赖；
- 非流形作为可选能力，不影响现有 2-流形用户；
- 与 CGAL 的 `Non_manifold` 标签思路一致，便于用户理解迁移成本。

实现工作量集中于 `storage` / `io` / `topology_ops` 三处，预计 2-3 周。
