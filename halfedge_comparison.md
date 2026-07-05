## halfedge 项目与主流开源半边数据结构库对比分析

### 一、对比范围

本报告将 `halfedge`（Rust，本项目的半边网格库）与以下主流开源半边数据结构库进行系统性对比：

| 库名 | 语言 | 类型 | 维护状态 |
|---|---|---|---|
| **CGAL HalfedgeDS** | C++ | 老牌计算几何库 | 活跃（v6.0.1） |
| **CGAL Surface_mesh** | C++ | CGAL 新一代半边结构 | 活跃 |
| **OpenMesh** | C++ | 轻量网格处理库 | 活跃（v9.x） |
| **geometry-central** | C++ | 学术几何处理库 | 活跃（Nicholas Sharp 等） |
| **libigl** | C++ | 通用网格处理工具箱 | 活跃（ETH Zurich） |
| **plexus** | Rust | 图论风格网格库 | 停滞（v0.2.x，2021年停更） |
| **mesh-graph (Synphonyte)** | Rust | 高性能半边三角网格 | 活跃但小众 |
| **tri-mesh** | Rust | 三角网格库 | 活跃 |

---

### 二、核心数据结构设计

#### 2.1 存储策略

| 库 | 底层存储 | 寻址方式 | 删除策略 |
|---|---|---|---|
| **halfedge (本项目)** | `SlotMap<K, V>` ×3 | 索引+版本号 | 即时删除+版本号防ABA |
| **CGAL HalfedgeDS** | 双向链表 / `std::vector` | 指针+句柄包装 | 即时删除 |
| **CGAL Surface_mesh** | `std::vector` + 自由链表 | `uint32_t` 索引 | 标记删除+GC压缩 |
| **OpenMesh** | `std::vector` | 数组索引 | 标记删除+GC |
| **geometry-central** | 连续数组 + 数据容器绑定 | 索引 | 即时重编号 |
| **libigl** | 裸 `Eigen::Matrix` (V,F) | 矩阵行列索引 | 不适用（无拓扑结构） |
| **plexus** | `petgraph` 图 | 图节点索引 | 委托 petgraph |
| **mesh-graph** | 紧凑数组 | 索引 | 标记删除 |
| **tri-mesh** | `Vec` 数组 | 索引 | 即时删除 |

**分析：** 本项目的 `SlotMap` 设计在 Rust 生态中是独特且合理的选择。与 C++ 库常用的裸 `vector`+自由链表相比，SlotMap 自动管理版本号，彻底解决了悬垂句柄（dangling handle）问题——任何对已删除元素的操作会安全返回 `None` 而非 UB。CGAL Surface_mesh 和 OpenMesh 的 GC 策略在大规模删除后需要显式调用 `garbage_collection()`，增加了 API 复杂度。geometry-central 的即时重编号保证了数组紧凑性，但会使所有现有句柄失效。

本项目的代价是每个 SlotMap 条目携带额外开销（版本号 u32 + 占用标志），在百万级网格下内存开销略高于纯 Vec 方案。不过这一开销在实际场景中（几何数据本身占大头）通常可忽略。

#### 2.2 元素关联

| 库 | Vertex 存储 | HalfEdge 存储 | Face 存储 |
|---|---|---|---|
| **halfedge** | `pos + Option<he>` | `dst + Option<twin/next/prev/face>` | `Option<he>` |
| **CGAL HalfedgeDS** | 点 + 入射半边 | 对向/后继/前驱/面/点 | 半边 |
| **CGAL Surface_mesh** | 半边 | 目标/对向/后继/前驱/面 | 半边 |
| **OpenMesh** | 半边句柄 | 目标/对向/后继/前驱/面 | 半边句柄 |
| **geometry-central** | 位置数组 | 隐式（通过索引推算） | 起始半边+度数 |
| **libigl** | V矩阵行 | 无（仅 F 矩阵列） | F矩阵列 |

**分析：** 本项目与 CGAL/OpenMesh 的关联模式基本一致，都是经典的半边数据结构四向关联（vertex→twin→next→prev→face）。geometry-central 采用了更紧凑的隐式半边编码（通过 face index + local index 推算），牺牲了通用性换取内存效率。libigl 严格来说不是半边结构，仅存储顶点坐标和面片索引，通过运行时构建邻接表来辅助算法——它更像一个算法库而非数据结构库。

本项目的 `Option` 包裹设计体现了 Rust 的安全哲学：所有拓扑关系都可能不存在（边界边无 face、孤立顶点无 halfedge），这迫使调用者处理每个 None 情况，杜绝了 C++ 库中常见的空指针崩溃。

---

### 三、API 设计与人体工程学

#### 3.1 查询范式

| 库 | 查询风格 | 示例 |
|---|---|---|
| **halfedge** | 链式 DSL + 急/惰性双迭代器 | `v.halfedge_to(v2).cw_rotated().dst_vert().run(&mesh)` |
| **CGAL** | 句柄导航 + circulator | `he->next()->vertex()->point()` |
| **OpenMesh** | 句柄导航 + 迭代器/循环器 | `mesh.v_iter(vertex_handle)` |
| **geometry-central** | 数据容器 + 成员访问 | `v.getVertex()` + `VertexData<double>` |
| **libigl** | 矩阵索引 | `V.row(i)`, `F.row(f)` |
| **plexus** | 图遍历 + 迭代器 | `graph.vertices().map(\|v\| v.get())` |

**分析：** 本项目的 `MeshQuery` 链式 DSL 是一个有特色的设计——它将拓扑查询表达为一系列惰性 `Option` 组合操作，支持链式调用并在任何环节短路返回 `None`。这在 C++ 库中没有直接对应物，CGAL/OpenMesh 的导航是命令式的（逐条调用方法），不自动处理无效状态。geometry-central 的数据容器绑定模式（`VertexData<T>` 随 mesh 自动扩展）在算法编程中非常实用，本项目的 `PropertyHandle<T>` 实现了类似功能但采用 HashMap 而非 Vec 索引。

本项目提供的急/惰性双迭代器策略也是一个亮点：急迭代器（收集到 Vec）允许在迭代过程中可变借用 mesh，惰性迭代器（零堆分配）则保持不可变借用。这种设计在 C++ 库中不存在——C++ 的迭代器总是零分配但需要借用语义，无法同时支持可变操作。

#### 3.2 拓扑操作

| 操作 | halfedge | CGAL | OpenMesh | geometry-central | libigl |
|---|---|---|---|---|---|
| split_edge | 完整实现+验证 | Euler 操作子 | split_edge | intrinsic split | 无 |
| flip_edge | 完整实现+验证 | Euler 操作子 | 无内置 | intrinsic flip | flip_edge |
| collapse_edge | 完整+链接条件 | Euler 操作子 | collapse | collapse | collapse |
| extrude_face | 完整实现 | 无 | 无 | 无 | extrude |
| split_face | 完整实现 | 有 | split | 无 | 无 |
| 链接条件检查 | 自动 | 需手动 | 无 | 无 | 无 |

**分析：** 本项目的拓扑操作是同类 Rust 库中最完整的，与 CGAL 的 Euler operator 子系统定位接近。每个操作内置了操作后的 `validate_mesh` 验证调用——这在所有对比库中是独有的。CGAL 的 Euler operator 在数学上更严格（保证操作后仍是 2-流形），但 API 更复杂（需要理解 `join_face`/`split_face`/`join_vertex` 等 Euler 操作的组合）。OpenMesh 的拓扑操作相对简陋，geometry-central 侧重于内在几何（intrinsic geometry）而非显式拓扑修改。

---

### 四、功能完整性对比

#### 4.1 功能矩阵

| 功能 | halfedge | CGAL SM | OpenMesh | geo-central | libigl | plexus | tri-mesh |
|---|---|---|---|---|---|---|---|
| 半边遍历 | **完整** | 完整 | 完整 | 完整 | 部分 | 图遍历 | 基础 |
| 边界检测 | **完整** | 完整 | 完整 | 完整 | 部分 | 无 | 部分 |
| 连通分量 | **BFS** | BGL | 无内置 | 有 | 有 | 图连通 | 无 |
| 方向检测/修复 | **BFS** | 有 | 有 | 有 | 有 | 无 | 无 |
| Loop 细分 | **有** | 有 | 有 | 无 | 有 | 有 | 无 |
| Catmull-Clark | **有** | 有 | 无 | 无 | 有 | 无 | 无 |
| Sqrt(3) 细分 | **有** | 无 | 无 | 无 | 无 | 无 | 无 |
| QEM 简化 | **有** | 有 | 有 | 无 | 有 | 无 | 无 |
| Laplacian 平滑 | **均匀+cotan** | 有 | 有 | 有 | 有 | 无 | 无 |
| 特征边检测 | **有** | 有 | 无 | 有 | 有 | 无 | 无 |
| AABB | **有** | 有 | 无 | 无 | 有 | 无 | 无 |
| 顶点焊接 | **有** | 无 | 无 | 无 | 无 | 无 | 无 |
| 自定义属性 | **HashMap** | 运行时 | 运行时 | 数据容器 | 矩阵列 | 无 | 无 |
| OBJ I/O | **有** | 有 | 有 | 有 | 有 | 无 | 有 |
| PLY I/O | **有** | 有 | 有 | 有 | 有 | 无 | 有 |
| GPU 导出 | **wgpu** | 无 | 无 | 无 | 无 | 无 | 有 |
| 拓扑验证 | **8类检查** | 断言 | 无 | 无 | 无 | 无 | 无 |
| Euler 特征/亏格 | **有** | BGL | 无 | 有 | 有 | 无 | 无 |

**小结：** 本项目在功能覆盖面上已达到 OpenMesh 和 geometry-central 的水平，在多个方面（Sqrt(3) 细分、顶点焊接、8 类拓扑验证、方向修复、内蕴 Delaunay、N-RoSy 方向场、Marching Cubes）甚至超越了多数 C++ 库。参数化、测地线、共形映射、网格变形、布尔运算等高级算法已补齐。

#### 4.2 I/O 格式

| 库 | OBJ | PLY | STL | OFF | glTF | 其他 |
|---|---|---|---|---|---|---|
| **halfedge** | 读写 | ASCII读写 | 无 | 无 | 无 | — |
| **CGAL SM** | 读写 | 读写 | 读写 | 读写 | 有 | 20+ 格式 |
| **OpenMesh** | 读写 | 读写 | 读写 | 读写 | 有 | 10+ 格式 |
| **geometry-central** | 读写 | 读写 | 读写 | 有 | 无 | — |
| **libigl** | 读写 | 读写 | 读写 | 读写 | 无 | MESH, WRL... |
| **tri-mesh** | 读写 | 无 | 无 | 无 | 无 | — |

---

### 五、依赖与生态

| 库 | 运行时依赖数 | 核心依赖 | 编译复杂度 |
|---|---|---|---|
| **halfedge** | **1** | `slotmap` | 极低（`cargo build`） |
| **CGAL** | 5+ | Boost, Eigen, GMP, MPFR | 高（CMake + 系统库） |
| **OpenMesh** | 0 | 标准库 | 中（CMake） |
| **geometry-central** | 3+ | Eigen, STB, Nanoflann | 中（CMake + Eigen） |
| **libigl** | 2+ | Eigen, GLFW(可选) | 高（CMake + 大量可选依赖） |
| **plexus** | 5+ | petgraph, nalgebra, fnv | 中 |
| **tri-mesh** | 2+ | nalgebra, rand | 低 |

**分析：** 本项目仅 1 个运行时依赖（`slotmap`），在所有对比库中依赖最少。这使得编译速度极快、交叉编译友好、且几乎不会遇到依赖冲突。CGAL 的依赖链是出了名的复杂（Boost + GMP + MPFR 在 Windows 上尤其痛苦），libigl 虽有 header-only 模式但完整功能依赖大量子模块。

手写的 `[f64; 3]` 向量运算虽然不如 nalgebra/cgmath 功能全面，但避免了引入数学库的开销和 API 版本兼容性问题。对于几何处理库来说这是合理的取舍。

---

### 六、性能特征

#### 6.1 内存效率

| 库 | 每顶点开销(估计) | 每半边开销 | 每面开销 |
|---|---|---|---|
| **halfedge** | ~48B (SlotMap entry) | ~56B | ~24B |
| **CGAL Surface_mesh** | ~32B (index) | ~32B | ~16B |
| **OpenMesh** | ~32B | ~32B | ~16B |
| **geometry-central** | ~24B (紧凑) | ~24B (隐式) | ~12B |
| **libigl** | 24B (3×f64) | N/A | 24B (3×i32) |

SlotMap 的 per-entry 开销（version + occupied flag + Option 对齐填充）使得本项目在裸内存比较中略高于纯 Vec 方案。但考虑到几何数据本身（顶点坐标 24B + 法线等属性）远大于拓扑索引开销，实际差距不大。

#### 6.2 操作复杂度

| 操作 | halfedge | CGAL SM | OpenMesh |
|---|---|---|---|
| 插入元素 | O(1) amortized | O(1) amortized | O(1) |
| 删除元素 | O(1) | O(1)（标记） | O(1)（标记） |
| 句柄查找 | O(1) | O(1) | O(1) |
| 句柄失效安全 | **自动** | GC后失效 | GC后失效 |
| 邻域遍历 | O(degree) | O(degree) | O(degree) |
| GC/压缩 | **不需要** | O(n) | O(n) |

**分析：** 本项目不需要 GC 步骤是一个实际优势——在增量式网格编辑（如反复 split/collapse 的 remesh 算法）中，C++ 库需要定期调用 `garbage_collection()` 来回收标记删除的元素，否则数组会无限膨胀。SlotMap 的即时删除避免了这一问题。

---

### 七、安全与正确性

| 方面 | halfedge | CGAL | OpenMesh | geometry-central |
|---|---|---|---|---|
| 空指针安全 | **编译期保证** | 运行时崩溃 | 运行时崩溃 | 运行时崩溃 |
| 悬垂句柄 | **版本号自动拦截** | UB | UB | 数组越界 |
| 并发安全 | Rust 所有权 | 需手动 | 需手动 | 需手动 |
| 操作后验证 | **每次操作自动验证** | 可选断言 | 无 | 无 |
| 边界条件 | **Option 链处理** | 需手动检查 | 需手动检查 | 需手动检查 |

**分析：** 这是本项目相比所有 C++ 库最显著的优势。Rust 的类型系统在编译期消除了整个类别的 bug（空指针、悬垂引用、数据竞争），而本项目的 SlotMap 版本号进一步消除了逻辑层面的悬垂句柄问题。每次拓扑操作后的自动 `validate_mesh` 调用则提供了运行时的额外安全网。对于几何处理这类容易出现微妙拓扑错误的领域，这些保证有重大实际价值。

---

### 八、文档与测试

| 方面 | halfedge | CGAL | OpenMesh | geometry-central | libigl |
|---|---|---|---|---|---|
| 代码内文档 | **中文详尽** | 英文详尽 | 英文+教程 | 英文教程网站 | 英文教程网站 |
| 独立文档 | **29篇LaTeX PDF** | 完整手册 | 教程+论文 | 论文+文档 | 教程+论文 |
| 单元测试 | **477个** | 有 | 有 | 有限 | 有限 |
| 基准测试 | **4组Criterion** | 有 | 无公开 | 无公开 | 无公开 |
| 示例 | **15个** | 大量 | 有 | 有 | 大量 |

本项目在文档方面表现突出：29 篇 LaTeX 设计文档 + 15 个可运行示例 + 4 组 Criterion 基准测试的组合，在所有对比库中是最完备的。特别是每个模块的中文文档头注释，详细解释了算法思路、约定和设计决策。基准测试覆盖遍历迭代器（eager vs lazy）、拓扑操作、曲面细分、以及几何算法（测地线、参数化、QEM 简化、内蕴 Delaunay、方向场、Marching Cubes）。

---

### 九、定位与适用场景

| 库 | 定位 | 最适合 |
|---|---|---|
| **halfedge (本项目)** | Rust 原生半边网格库 | 需要安全保证的拓扑编辑、Remesh/细分管线、Rust 项目集成 |
| **CGAL** | 工业级计算几何 | 精确算术、复杂几何谓词、需要数学正确性保证 |
| **OpenMesh** | 轻量网格处理 | 快速原型、教学、C++ 项目中的网格数据容器 |
| **geometry-central** | 学术几何处理 | 离散微分几何、内在几何、研究方向场 |
| **libigl** | 算法工具箱 | 参数化、变形、物理仿真、大量算法实验 |
| **plexus** | 图论网格 | 理论研究、小规模网格（已停更） |
| **tri-mesh** | 简单三角网格 | 轻量级三角网格操作、渲染管线 |

---

### 十、总结

#### 本项目的核心优势

1. **内存安全 + 句柄安全：** 在所有对比库中唯一同时提供编译期空指针安全和运行时悬垂句柄防护的实现
2. **零依赖负担：** 仅 1 个运行时依赖，编译速度极快
3. **拓扑操作完整且自验证：** split/flip/collapse/extrude 全系列 + 操作后自动验证
4. **双模迭代器：** 急/惰性策略可按需选择，兼顾可变借用和零分配
5. **链式查询 DSL：** 人体工程学的拓扑导航 API
6. **文档质量：** 29 篇 LaTeX 设计文档 + 详尽代码注释

#### 可改进方向

1. **I/O 格式扩展：** 可增加 OFF、glTF 支持以扩大适用场景（OBJ/PLY/STL 已支持）
2. **SOA 存储优化：** 当前属性系统用 HashMap，大规模下可考虑 Vec 索引以提升性能
3. **并行化扩展：** 目前已有 rayon 并行的曲率/平滑/面积计算，可进一步并行化测地线、参数化等算法
4. **数值精度：** f64 对于精确几何谓词可能不够，已引入 Shewchuk 鲁棒谓词，可考虑进一步精确算术（类似 CGAL 的 EPEC kernel）
5. **非流形支持：** 当前严格限制为 2-流形，geometry-central 的 non-manifold Laplacian 思路值得参考
6. **教程网站：** 可基于现有 29 篇 LaTeX 文档构建 mdbook 教程网站，提升对外可发现性

#### 总体评价

本项目在 Rust 生态中填补了一个重要空缺——当前 Rust 社区缺少一个功能完整、设计严谨的半边数据结构库。plexus 已停更且偏向图论抽象，mesh-graph 和 tri-mesh 功能相对基础。本项目在拓扑操作完整性、安全保证和文档质量上已达到可投入实际使用的水平，在 Rust 网格处理生态中具有独特价值。
