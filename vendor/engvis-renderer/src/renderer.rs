use wgpu::util::DeviceExt;
use engvis_core::{OrbitCamera, Scene, RenderState};
use glam::Affine3A;

use crate::depth::DepthTexture;
use crate::grid_renderer::GridRenderer;
use crate::lighting::LightingBuffer;
use crate::material_pipeline::MaterialPipeline;
use crate::mesh_renderer::MeshRenderer;
use crate::overlay_renderer::OverlayRenderer;
use crate::texture_cache::TextureCache;
use crate::custom_material::CustomMaterial;
use crate::postprocess::PostProcessPipeline;





/// Scene uniform data (group 0)
///
/// `global_opacity`: `[opacity, env_intensity, _, _]`
///   - x: global surface opacity (0..1)
///   - y: environment / IBL intensity multiplier
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub viewport: [f32; 4],
    pub global_opacity: [f32; 4],
}

pub struct Renderer {
    pub depth: DepthTexture,
    pub msaa_texture: wgpu::Texture,
    pub msaa_view: wgpu::TextureView,
    pub surface_format: wgpu::TextureFormat,
    pub scene_uniform_buffer: wgpu::Buffer,
    pub scene_bind_group: wgpu::BindGroup,
    pub scene_bind_group_layout: wgpu::BindGroupLayout,
    pub lighting: LightingBuffer,
    pub material_pipeline: MaterialPipeline,
    pub mesh_renderer: MeshRenderer,
    pub grid_renderer: GridRenderer,
    pub overlay_renderer: OverlayRenderer,
    pub texture_cache: TextureCache,

    /// Internal render state. Set via `set_state()` or `set_state_from_camera()`.
    state: RenderState,
    /// MSAA sample count.
    sample_count: u32,
    /// Optional custom material override.
    custom_material: Option<Box<dyn CustomMaterial>>,
    /// FXAA post-processing pipeline.
    post_process: PostProcessPipeline,
    /// Intermediate texture for FXAA input (MSAA-resolved or direct scene output).
    fxaa_texture: wgpu::Texture,
    fxaa_view: wgpu::TextureView,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        scene: &Scene,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        let sc = sample_count.max(1);
        let depth = DepthTexture::new(device, width.max(1), height.max(1), sc);

        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let scene_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scene Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let initial_uniforms = SceneUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            viewport: [width as f32, height as f32, 0.0, 0.0],
            global_opacity: [1.0, 1.0, 0.0, 0.0],
        };

        let scene_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scene Uniform Buffer"),
                contents: bytemuck::cast_slice(&[initial_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scene Bind Group"),
            layout: &scene_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_uniform_buffer.as_entire_binding(),
            }],
        });

        let lighting = LightingBuffer::new(device, &scene.lighting);

        let scene_layout_for_grid = scene_bind_group_layout.clone();
        let grid_renderer = GridRenderer::new(device, surface_format, &scene_layout_for_grid, sample_count);

        let mesh_renderer = MeshRenderer::new(device);
        let material_pipeline = MaterialPipeline::new(
            device,
            surface_format,
            &scene_bind_group_layout,
            &lighting.bind_group_layout,
            &mesh_renderer.object_bind_group_layout,
            sample_count,
        );

        let scene_layout_for_overlay = scene_bind_group_layout.clone();
        let overlay_renderer = OverlayRenderer::new(
            device,
            surface_format,
            crate::depth::DepthTexture::FORMAT,
            &scene_layout_for_overlay,
            &mesh_renderer.object_bind_group_layout,
            sample_count,
        );

        let texture_cache = TextureCache::new(device, queue);

        let post_process = PostProcessPipeline::new(device, surface_format);
        let w = width.max(1);
        let h = height.max(1);
        let fxaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FXAA Intermediate Texture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let fxaa_view = fxaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut renderer = Self {
            depth,
            msaa_texture,
            msaa_view,
            surface_format,
            scene_uniform_buffer,
            scene_bind_group,
            scene_bind_group_layout,
            lighting,
            material_pipeline,
            mesh_renderer,
            grid_renderer,
            overlay_renderer,
            texture_cache,
            state: RenderState::default(),
            sample_count: sc,
            custom_material: None,
            post_process,
            fxaa_texture,
            fxaa_view,
        };

        renderer.upload_scene(device, queue, scene);
        renderer
    }

    // ── State management ─────────────────────────────────────

    /// Replace the entire render state at once.
    pub fn set_state(&mut self, state: &RenderState) {
        self.state = *state;
    }

    /// Update render state from camera clipping planes (deprecated — clipping planes
    /// are now managed directly via `FrameCtx::set_clip_planes()`).
    pub fn set_state_from_camera(&mut self, _camera: &OrbitCamera) {}

    /// Read-only access to current state.
    pub fn state(&self) -> &RenderState {
        &self.state
    }

    /// Mutable access to current state (for per-field changes).
    pub fn state_mut(&mut self) -> &mut RenderState {
        &mut self.state
    }

    /// Set a custom material override. Pass `None` to clear.
    pub fn set_custom_material(&mut self, material: Option<Box<dyn CustomMaterial>>) {
        self.custom_material = material;
    }

    /// Check if a custom material is active.
    pub fn has_custom_material(&self) -> bool {
        self.custom_material.is_some()
    }

    // ── Upload and resize ───────────────────────────────────

    pub fn upload_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
    ) {
        // Re-upload meshes
        self.mesh_renderer.mesh_buffers.clear();
        for mesh in &scene.meshes {
            self.mesh_renderer.upload_mesh(device, mesh);
        }

        // Create material bind groups
        self.mesh_renderer.material_bind_groups.clear();
        self.mesh_renderer.material_uniform_buffers.clear();
        for material in &scene.materials {
            let (bg, buf) = self
                .material_pipeline
                .create_material_bind_group(device, material, &self.texture_cache);
            self.mesh_renderer.material_bind_groups.push(bg);
            self.mesh_renderer.material_uniform_buffers.push(buf);
        }

        // Update lighting
        self.lighting.update(queue, &scene.lighting);
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        let sc = self.sample_count.max(1);
        self.depth = DepthTexture::new(device, w, h, sc);
        self.msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Color Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.msaa_view = self.msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate FXAA intermediate texture
        self.fxaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FXAA Intermediate Texture"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.fxaa_view = self.fxaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    // ── Render pass ─────────────────────────────────────────

   #[allow(clippy::too_many_arguments)]
   pub fn render_scene_pass(
       &self,
       device: &wgpu::Device,
       queue: &wgpu::Queue,
       _view: &wgpu::TextureView,
       encoder: &mut wgpu::CommandEncoder,
       scene: &Scene,
       camera: &OrbitCamera,
       width: u32,
       height: u32,
   ) {
        // Update uniforms
       let scene_uniforms = SceneUniforms {
           view_proj: camera.view_projection().to_cols_array_2d(),
           camera_pos: [camera.position().x, camera.position().y, camera.position().z, 1.0],
           viewport: [width as f32, height as f32, 0.0, 0.0],
           global_opacity: [self.state.opacity, self.state.env_intensity, 0.0, 0.0],
       };
        queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[scene_uniforms]),
        );

        // Update lighting
        self.lighting.update(queue, &scene.lighting);

        // Keep grid visibility adaptive to background brightness.
        self.grid_renderer.update_background_color(queue, self.state.background_color);

        // Update depth texture sizes if camera planes changed
        let s = &self.state;
        let depth_tex = &self.depth;

        // Scene always renders to intermediate texture (FXAA post-process reads from it)
        let resolve_target = if self.sample_count > 1 { Some(&self.fxaa_view) } else { None };
        let color_view = if self.sample_count > 1 { &self.msaa_view } else { &self.fxaa_view };
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: self.state.background_color[0] as f64,
                        g: self.state.background_color[1] as f64,
                        b: self.state.background_color[2] as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_tex.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });

       render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
       render_pass.set_bind_group(1, &self.lighting.bind_group, &[]);

      // --- Grid (opaque, drawn first so transparent surface can show through) ---
      if s.show_grid {
          self.grid_renderer.render(&mut render_pass);
          // Re-bind scene + lighting bind groups after grid renderer
          render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
          render_pass.set_bind_group(1, &self.lighting.bind_group, &[]);
      }

       // --- Surface rendering ---
       if s.show_surface {
           render_pass.set_pipeline(&self.material_pipeline.solid_pipeline);
           self.render_scene_nodes(&mut render_pass, scene, device, Affine3A::IDENTITY);
       }

       // --- Edge overlay ---
       if s.edge_opts.enabled {
           render_pass.set_pipeline(&self.overlay_renderer.line_pipeline);
           self.render_overlay_nodes_edges(
               &mut render_pass, scene, device,
               Affine3A::IDENTITY,
               s.edge_opts.color, s.edge_opts.line_width,
           );
       }

        // --- Vertex overlay ---
        if s.vertex_opts.enabled {
            let (_buf, overlay_bg) = self
                .overlay_renderer
                .create_uniform_bind_group(device, s.vertex_opts.color, s.vertex_opts.point_size, 0.0);
            render_pass.set_pipeline(&self.overlay_renderer.point_pipeline);
            self.render_overlay_nodes_vertices(&mut render_pass, scene, device, Affine3A::IDENTITY, &overlay_bg);
        }
    }

    /// Apply FXAA post-processing: read from the intermediate scene texture,
    /// write the anti-aliased result to `dest_view` (typically the surface texture).
    pub fn render_post_process(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dest_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.post_process.render(
            device,
            encoder,
            &self.fxaa_view,
            dest_view,
            width,
            height,
        );
    }

    fn render_scene_nodes(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        parent_transform: Affine3A,
    ) {
        for node in &scene.nodes {
            self.render_node(render_pass, scene, device, node, parent_transform);
        }
    }

    fn render_node(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        node: &engvis_core::SceneNode,
        parent_transform: Affine3A,
    ) {
        if !node.visible {
            return;
        }

        let world_transform = parent_transform * node.local_transform;

        if node.render_surface
            && let Some(mesh_idx) = node.mesh_index
            && mesh_idx < self.mesh_renderer.mesh_buffers.len()
            && let mesh_buf = &self.mesh_renderer.mesh_buffers[mesh_idx]
            && mesh_buf.vertex_count > 0 && mesh_buf.index_count > 0
            && mesh_idx < scene.meshes.len()
        {
                let mesh = &scene.meshes[mesh_idx];

                render_pass.set_vertex_buffer(0, mesh_buf.vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    mesh_buf.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                let (_obj_buf, obj_bg) = self
                    .mesh_renderer
                    .create_object_bind_group(device, world_transform);
                render_pass.set_bind_group(3, &obj_bg, &[]);

                for sub_mesh in &mesh.sub_meshes {
                    let mat_idx = sub_mesh.material_index;
                    if mat_idx < self.mesh_renderer.material_bind_groups.len() {
                        render_pass.set_bind_group(
                            2,
                            &self.mesh_renderer.material_bind_groups[mat_idx],
                            &[],
                        );
                    }
                    render_pass.draw_indexed(
                        sub_mesh.index_offset..sub_mesh.index_offset + sub_mesh.index_count,
                        0,
                        0..1,
                    );
                }
            }

        for child in &node.children {
            self.render_node(render_pass, scene, device, child, world_transform);
        }
    }

    fn render_overlay_nodes_vertices(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        parent_transform: Affine3A,
        overlay_bind_group: &wgpu::BindGroup,
    ) {
        for node in &scene.nodes {
            self.render_overlay_node_vertices(render_pass, device, node, parent_transform, overlay_bind_group);
        }
    }

    fn render_overlay_node_vertices(
        &self,
        render_pass: &mut wgpu::RenderPass,
        device: &wgpu::Device,
        node: &engvis_core::SceneNode,
        parent_transform: Affine3A,
        overlay_bind_group: &wgpu::BindGroup,
    ) {
        if !node.visible {
            return;
        }
        let world_transform = parent_transform * node.local_transform;

        if let Some(mesh_idx) = node.mesh_index
            && mesh_idx < self.mesh_renderer.mesh_buffers.len()
            && let mesh_buf = &self.mesh_renderer.mesh_buffers[mesh_idx]
            && mesh_buf.vertex_count > 0
        {
            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
            let (_obj_buf, obj_bg) = self
                .mesh_renderer
                .create_object_bind_group(device, world_transform);
            render_pass.set_bind_group(1, &obj_bg, &[]);
            render_pass.set_bind_group(2, overlay_bind_group, &[]);
            render_pass.set_vertex_buffer(0, mesh_buf.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.overlay_renderer.point_quad_buffer.slice(..));
            render_pass.draw(0..6, 0..mesh_buf.vertex_count);
        }

        for child in &node.children {
            self.render_overlay_node_vertices(render_pass, device, child, world_transform, overlay_bind_group);
        }
    }

    // ── Edge-overlay traversal with per-node color / width overrides ─
    //
    // Walks the scene graph the same way as `render_overlay_nodes`, but
    // for each node draws its edges with a uniform bind group built
    // from the (optionally-overridden) color and line width.  This lets
    // wireframe / annotation nodes have their own appearance independent
    // of the global `edge_opts` used for triangle-mesh edges.
    fn render_overlay_nodes_edges(
        &self,
        render_pass: &mut wgpu::RenderPass,
        scene: &Scene,
        device: &wgpu::Device,
        parent_transform: Affine3A,
        default_color: [f32; 3],
        default_width: f32,
    ) {
        for node in &scene.nodes {
            self.render_overlay_node_edges(
                render_pass, device, node, parent_transform,
                default_color, default_width,
            );
        }
    }

    fn render_overlay_node_edges(
        &self,
        render_pass: &mut wgpu::RenderPass,
        device: &wgpu::Device,
        node: &engvis_core::SceneNode,
        parent_transform: Affine3A,
        default_color: [f32; 3],
        default_width: f32,
    ) {
        if !node.visible {
            return;
        }
        let world_transform = parent_transform * node.local_transform;

        if node.render_edges
            && let Some(mesh_idx) = node.mesh_index
            && mesh_idx < self.mesh_renderer.mesh_buffers.len()
            && let mesh_buf = &self.mesh_renderer.mesh_buffers[mesh_idx]
            && mesh_buf.edge_instance_count > 0
        {
            let color = node.edge_color_override.unwrap_or(default_color);
            let width = node.edge_width_override.unwrap_or(default_width);
            let (_buf, overlay_bg) = self
                .overlay_renderer
                .create_uniform_bind_group(device, color, 0.0, width);

            render_pass.set_bind_group(0, &self.scene_bind_group, &[]);
            let (_obj_buf, obj_bg) = self
                .mesh_renderer
                .create_object_bind_group(device, world_transform);
            render_pass.set_bind_group(1, &obj_bg, &[]);
            render_pass.set_bind_group(2, &overlay_bg, &[]);
            if let Some(ref buf) = mesh_buf.edge_endpoint_buffer {
                render_pass.set_vertex_buffer(0, buf.slice(..));
                render_pass.set_vertex_buffer(1, self.overlay_renderer.line_quad_buffer.slice(..));
                render_pass.draw(0..6, 0..mesh_buf.edge_instance_count);
            }
        }

        for child in &node.children {
            self.render_overlay_node_edges(
                render_pass, device, child, world_transform,
                default_color, default_width,
            );
        }
    }
}
