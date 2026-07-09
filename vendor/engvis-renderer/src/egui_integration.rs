use std::sync::Arc;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use egui_winit::State;
use winit::window::Window;

pub struct EguiContext {
    pub context: egui::Context,
    pub state: State,
    pub renderer: egui_wgpu::Renderer,
}

impl EguiContext {
    pub fn new(window: &Arc<Window>, device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let context = egui::Context::default();
        let state = State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(device, surface_format, RendererOptions::default());

        Self {
            context,
            state,
            renderer,
        }
    }

    pub fn take_input(&mut self, window: &Window) -> egui::RawInput {
        self.state.take_egui_input(window)
    }

    pub fn handle_output(&mut self, window: &Window, output: egui::PlatformOutput) {
        self.state.handle_platform_output(window, output);
    }

    pub fn tessellate(
        &self,
        shapes: Vec<egui::epaint::ClippedShape>,
        pixels_per_point: f32,
    ) -> Vec<egui::ClippedPrimitive> {
        self.context.tessellate(shapes, pixels_per_point)
    }
}

/// Render both the 3D scene and the egui overlay onto the same surface texture.
///
/// Takes ownership of the `SurfaceTexture` so the view created from it lives
/// for the entire encoding scope (both 3D and egui passes).
///
/// `scene_render` is a closure that records the 3D render pass onto the shared
/// command encoder. It runs before the egui render pass, so the egui pass can
/// composite UI panels on top via `LoadOp::Load`.
pub fn render_egui<F>(
    egui_ctx: &mut EguiContext,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    window: &Window,
    surface_texture: wgpu::SurfaceTexture,
    full_output: egui::FullOutput,
    scene_render: F,
) -> (wgpu::CommandBuffer, wgpu::SurfaceTexture)
where
    F: FnOnce(&wgpu::TextureView, &mut wgpu::CommandEncoder),
{
    egui_ctx.handle_output(window, full_output.platform_output);

    let screen_descriptor = ScreenDescriptor {
        size_in_pixels: [window.inner_size().width, window.inner_size().height],
        pixels_per_point: window.scale_factor() as f32,
    };

    let tris = egui_ctx.tessellate(full_output.shapes, screen_descriptor.pixels_per_point);

    for (id, image_delta) in &full_output.textures_delta.set {
        egui_ctx
            .renderer
            .update_texture(device, queue, *id, image_delta);
    }

    // Create a single view from the surface texture.
    // This view lives for the entire function scope, covering both passes.
    let view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Frame Encoder"),
    });

    egui_ctx
        .renderer
        .update_buffers(device, queue, &mut encoder, &tris, &screen_descriptor);

    // 3D scene pass (recorded onto the same encoder, before egui)
    scene_render(&view, &mut encoder);

    // egui overlay pass (LoadOp::Load composites on top of the 3D content)
    {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        egui_ctx
            .renderer
            .render(&mut render_pass.forget_lifetime(), &tris, &screen_descriptor);
    }

    for id in &full_output.textures_delta.free {
        egui_ctx.renderer.free_texture(id);
    }

    (encoder.finish(), surface_texture)
}
