use wgpu::util::DeviceExt;
use glam::Affine3A;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    pub params: [f32; 4], // x = point_size, y = line_width, zw = padding
    pub color: [f32; 4],
}

pub struct OverlayRenderer {
    /// MSAA point pipeline (renders into MSAA texture)
    pub point_pipeline: wgpu::RenderPipeline,
    /// Instanced triangle line pipeline (MSAA, adjustable width)
    pub line_pipeline: wgpu::RenderPipeline,
    /// Quad vertex buffer for point rendering
    pub point_quad_buffer: wgpu::Buffer,
    /// Quad vertex buffer for line rendering
    pub line_quad_buffer: wgpu::Buffer,
    pub uniform_bind_group_layout: wgpu::BindGroupLayout,
}

impl OverlayRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        scene_layout: &wgpu::BindGroupLayout,
        object_layout: &wgpu::BindGroupLayout,
        sample_count: u32,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::shader_source().into()),
        });

        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Overlay Uniform Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(std::num::NonZeroU64::new(std::mem::size_of::<OverlayUniforms>() as u64).unwrap()),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Overlay Pipeline Layout"),
            bind_group_layouts: &[scene_layout, object_layout, &uniform_layout],
            push_constant_ranges: &[],
        });

        // --- Point quad vertices: centered at origin, size controlled by uniform ---
        #[repr(C)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PointQuadVertex {
            x: f32,
            y: f32,
        }
        let point_quad_vertices: [PointQuadVertex; 6] = [
            PointQuadVertex { x: -0.5, y: -0.5 },
            PointQuadVertex { x:  0.5, y: -0.5 },
            PointQuadVertex { x: -0.5, y:  0.5 },
            PointQuadVertex { x: -0.5, y:  0.5 },
            PointQuadVertex { x:  0.5, y: -0.5 },
            PointQuadVertex { x:  0.5, y:  0.5 },
        ];
        let point_quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let point_quad_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PointQuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 4,
                },
            ],
        };

        // --- Line quad vertices: along=0..1 (endpoint A to B), perp=-0.5..0.5 (width) ---
        #[repr(C)]
        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct LineQuadVertex {
            along: f32,
            perp: f32,
        }
        let line_quad_vertices: [LineQuadVertex; 6] = [
            LineQuadVertex { along: 0.0, perp: -0.5 },
            LineQuadVertex { along: 1.0, perp: -0.5 },
            LineQuadVertex { along: 0.0, perp:  0.5 },
            LineQuadVertex { along: 0.0, perp:  0.5 },
            LineQuadVertex { along: 1.0, perp: -0.5 },
            LineQuadVertex { along: 1.0, perp:  0.5 },
        ];
        let line_quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Line Quad Vertex Buffer"),
            contents: bytemuck::cast_slice(&line_quad_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let line_quad_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineQuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 4,
                },
            ],
        };

        // --- Instance layouts ---
        let point_instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<engvis_core::MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
            ],
        };

        let line_endpoint_layout = wgpu::VertexBufferLayout {
            array_stride: 24,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        // --- Point pipeline ---
        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Point Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main_point"),
                buffers: &[point_instance_layout, point_quad_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main_point"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // --- Line pipeline (instanced triangle, adjustable width) ---
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Overlay Line Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main_line"),
                buffers: &[line_endpoint_layout, line_quad_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            point_pipeline,
            line_pipeline,
            point_quad_buffer,
            line_quad_buffer,
            uniform_bind_group_layout: uniform_layout,
        }
    }

    pub fn create_uniform_bind_group(
        &self,
        device: &wgpu::Device,
        color: [f32; 3],
        point_size: f32,
        line_width: f32,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let uniforms = OverlayUniforms {
            params: [point_size, line_width, 0.0, 0.0],
            color: [color[0], color[1], color[2], 1.0],
        };

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Overlay Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Overlay Uniform Bind Group"),
            layout: &self.uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        (buffer, bind_group)
    }

    fn shader_source() -> &'static str {
        r#"
struct SceneUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    viewport: vec4<f32>,
    global_opacity: vec4<f32>,
}

struct ObjectUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct OverlayUniforms {
    params: vec4<f32>,  // x = point_size, y = line_width, zw = padding
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<uniform> object: ObjectUniforms;
@group(2) @binding(0) var<uniform> overlay: OverlayUniforms;

// ============================================================
// Point rendering (instanced quad per vertex)
// ============================================================

struct PointVertexInput {
    @location(0) position: vec3<f32>,
    @location(4) quad_offset: vec2<f32>,
}

struct PointVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main_point(in: PointVertexInput) -> PointVertexOutput {
    var out: PointVertexOutput;
    let world_pos = (object.model * vec4<f32>(in.position, 1.0)).xyz;
    let clip_pos = scene.view_proj * vec4<f32>(world_pos, 1.0);

    let ndc_size = overlay.params.xx / scene.viewport.xy;
    out.clip_pos = vec4<f32>(
        clip_pos.x + in.quad_offset.x * ndc_size.x * clip_pos.w,
        clip_pos.y + in.quad_offset.y * ndc_size.y * clip_pos.w,
        clip_pos.z,
        clip_pos.w,
    );
    out.uv = in.quad_offset + 0.5;
    return out;
}

@fragment
fn fs_main_point(in: PointVertexOutput) -> @location(0) vec4<f32> {
    let dist = length(in.uv - 0.5);
    if (dist > 0.5) {
        discard;
    }
    let alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    return vec4<f32>(overlay.color.rgb, overlay.color.a * alpha);
}

// ============================================================
// Line rendering (instanced quad per edge, adjustable width)
// Same screen-space offset technique as points, but:
//   - along (0..1) interpolates between endpoints A and B
//   - perp (-0.5..0.5) expands perpendicular to edge direction
// ============================================================

struct LineVertexInput {
    @location(0) point_a: vec3<f32>,
    @location(1) point_b: vec3<f32>,
    @location(4) quad_offset: vec2<f32>,  // x = along (0..1), y = perp (-0.5..0.5)
}

struct LineVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) edge_dist: f32,
}

@vertex
fn vs_main_line(in: LineVertexInput) -> LineVertexOutput {
    var out: LineVertexOutput;

    // Transform both endpoints to clip space
    let world_a = (object.model * vec4<f32>(in.point_a, 1.0)).xyz;
    let world_b = (object.model * vec4<f32>(in.point_b, 1.0)).xyz;
    let clip_a = scene.view_proj * vec4<f32>(world_a, 1.0);
    let clip_b = scene.view_proj * vec4<f32>(world_b, 1.0);

    // Interpolate along edge: along=0 → A, along=1 → B
    let clip_pos = mix(clip_a, clip_b, in.quad_offset.x);

    // Compute edge direction in screen pixels
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;
    let dir_px = (ndc_b - ndc_a) * scene.viewport.xy;
    let dir_len = length(dir_px);

    // Perpendicular offset in NDC
    var perp_offset_ndc = vec2<f32>(0.0, 0.0);
    if (dir_len > 0.001) {
        let perp_px = normalize(vec2<f32>(-dir_px.y, dir_px.x));
        let half_w = overlay.params.y * 0.5;
        perp_offset_ndc = perp_px * in.quad_offset.y * half_w / scene.viewport.xy;
    }

    // Apply perpendicular offset with perspective correction
    out.clip_pos = vec4<f32>(
        clip_pos.x + perp_offset_ndc.x * clip_pos.w,
        clip_pos.y + perp_offset_ndc.y * clip_pos.w,
        clip_pos.z,
        clip_pos.w,
    );

    // Pass perp value to fragment for interpolation (NOT abs here!)
    out.edge_dist = in.quad_offset.y;
    return out;
}

@fragment
fn fs_main_line(in: LineVertexOutput) -> @location(0) vec4<f32> {
    let alpha = 1.0 - smoothstep(0.4, 0.5, abs(in.edge_dist));
    return vec4<f32>(overlay.color.rgb, overlay.color.a * alpha);
}
"#
    }
}

pub fn create_object_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    transform: Affine3A,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let model_mat = glam::Mat4::from(transform);
    let normal_mat = model_mat.inverse().transpose();

    #[repr(C)]
    #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ObjectUniforms {
        model: [[f32; 4]; 4],
        normal_matrix: [[f32; 4]; 4],
    }

    let uniforms = ObjectUniforms {
        model: model_mat.to_cols_array_2d(),
        normal_matrix: normal_mat.to_cols_array_2d(),
    };

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Overlay Object Uniform Buffer"),
        contents: bytemuck::cast_slice(&[uniforms]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Overlay Object Bind Group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (buffer, bind_group)
}
