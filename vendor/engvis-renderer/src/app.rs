use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowId;

use engvis_core::{
    InputState, OrbitCamera, Scene,
    RenderState,
};

use crate::{
    create_window_and_gpu, render_egui,
    EguiContext,
    GpuResources, Renderer,
};

// ── RunConfig ─────────────────────────────────────────────────
/// Configuration for the viewer window.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Subdivision level for surface meshes (app-level hint — not used by runner itself).
    pub mesh_subdivision: u8,
    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    pub sample_count: u32,
    /// Logging / debug helpers.
    pub log_level: String,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            title: "engvis".into(),
            width: 1280,
            height: 800,
            mesh_subdivision: 6,
            sample_count: 4,
            log_level: "warn".into(),
        }
    }
}

// ── FrameCtx ──────────────────────────────────────────────────
/// Per-frame context exposed to the user's `ui()` and `on_frame()` callbacks.
pub struct FrameCtx<'a> {
    /// Mutable scene reference (replace meshes / materials / nodes each frame if needed).
    pub scene: &'a mut Scene,
    /// Mutable camera reference.
    pub camera: &'a mut OrbitCamera,
    /// Aggregated render state (surface, grid, opacity, overlays).
    /// Field access: `frame.render_state.show_surface = true`.
    /// Atomic update: `frame.set_render_state(RenderState { .. })`.
    pub render_state: &'a mut RenderState,
    /// Whether egui is currently interacting with a widget (e.g. dragging a
    /// resize handle, a slider, a text field). When true, the runner skips
    /// camera orbit/zoom/pan for this frame.
    pub egui_wants_pointer: bool,
    /// Current cursor position in screen coordinates (physical pixels).
    pub cursor_x: f64,
    pub cursor_y: f64,
    /// Current FPS counter value.
    pub fps: f32,
    /// The 3D viewport rectangle (populated by the runner's central panel).
    pub viewport: &'a mut engvis_core::ViewportRect,
    /// Set to `true` if the scene's meshes/materials/nodes have changed
    /// so the runner re-uploads GPU buffers before rendering.
    pub scene_dirty: &'a mut bool,
    /// Reference to the wgpu Device (for operations like texture uploads in ui/frame).
    pub device: &'a wgpu::Device,
    /// Reference to the wgpu Queue.
    pub queue: &'a wgpu::Queue,
    /// Surface texture format.
    pub surface_format: wgpu::TextureFormat,
    /// Reference to the renderer's texture cache (for loading textured models).
    pub texture_cache: &'a mut crate::TextureCache,
}

impl FrameCtx<'_> {
    /// Apply a complete `RenderState` atomically (opacity, display toggles, overlay opts).
    pub fn set_render_state(&mut self, state: RenderState) {
        *self.render_state = state;
    }

    /// Set clipping planes explicitly. This will update both the camera and the
    /// renderer on the next frame.
    pub fn set_clip_planes(&mut self, near: f32, far: f32) {
        self.camera.near = near;
        self.camera.far = far;
    }
}

// ── EventHandling ────────────────────────────────────────────
/// Describes how an intercepted window event should be processed by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventHandling {
    /// Process the event normally (default camera interaction allowed).
    Default,
    /// Event consumed; also prevents camera orbit/zoom/pan.
    Consumed,
    /// Event consumed by the app; camera still processes it.
    NoCamera,
}

// ── AppCtx ────────────────────────────────────────────────────
/// Context available during setup / init callbacks.
pub struct AppCtx<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,
    pub config: &'a RunConfig,
}

// ── EngvisApp trait ───────────────────────────────────────────
/// Implement this trait to build an interactive 3D viewer app.
///
/// # Lifecycle
///
/// ```text
/// app.config()                     ← called first
///     ↓
/// app.on_setup(&mut AppCtx)        ← after GPU ready, returns Scene
///     ↓
/// app.on_ready(scene, camera)      ← camera setup, e.g. fit_to_scene
///     ↓   ┌───────────────────────────┐
///         ↓  per frame:               │
///     app.ui(egui_ctx, frame)     ←  │  draw egui panels
///         ↓                           │
///     app.on_frame(frame)         ←  │  model load, mesh rebuild, etc.
///         ↓                           │
///     render + present            →  loop
/// ```
///
/// `on_frame()` runs after `ui()` but before the render pass.
pub trait EngvisApp {
    /// Return a run configuration. Called before any GPU work.
    fn config(&self) -> RunConfig {
        RunConfig::default()
    }

    /// Called once after GPU init, before the first frame.
    /// Return the initial scene.
    fn on_setup(&mut self, ctx: &mut AppCtx) -> Scene;

    /// Called after `on_setup` returns; lets you adjust the camera.
    fn on_ready(&mut self, _scene: &Scene, _camera: &mut OrbitCamera) {}

    /// Called every frame inside the egui context. Draw your UI here.
    fn ui(&mut self, egui_ctx: &egui::Context, frame: &mut FrameCtx);

    /// Called every frame right after `ui()`, before the render pass.
    fn on_frame(&mut self, _frame: &mut FrameCtx) {}

    /// Optional: intercept a raw winit window event before egui gets it.
    /// Return `EventHandling::Consumed` to suppress camera interaction.
    fn on_event(&mut self, _event: &WindowEvent) -> EventHandling {
        EventHandling::Default
    }
}

// ── Internal runner ───────────────────────────────────────────
struct AppRunner<A: EngvisApp> {
    app: A,
    window: Option<Arc<winit::window::Window>>,
    gpu: Option<GpuResources>,
    egui: Option<EguiContext>,
    renderer: Option<Renderer>,
    camera: OrbitCamera,
    input: InputState,
    scene: Scene,
    /// The render state we share with the user each frame.
    render_state: RenderState,
    fps_ct: u32,
    fps_ts: Instant,
    fps_v: f32,
    viewport_rect: engvis_core::ViewportRect,
    setup_done: bool,
    ready_done: bool,
    scene_dirty: bool,
}

impl<A: EngvisApp> AppRunner<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            window: None,
            gpu: None,
            egui: None,
            renderer: None,
            camera: OrbitCamera::default(),
            input: InputState::default(),
            scene: Scene::default(),
            render_state: RenderState::default(),
            fps_ct: 0,
            fps_ts: Instant::now(),
            fps_v: 0.0,
            viewport_rect: engvis_core::ViewportRect::default(),
            setup_done: false,
            ready_done: false,
            scene_dirty: false,
        }
    }

    fn run_setup(&mut self, event_loop: &ActiveEventLoop) {
        let config = self.app.config();

        pollster::block_on(async {
            let (window, gpu) = create_window_and_gpu(
                event_loop,
                &config.title,
                config.width,
                config.height,
            )
            .await;
            let size = window.inner_size();
            let egui = EguiContext::new(&window, &gpu.context.device, gpu.surface_format);

            let mut app_ctx = AppCtx {
                device: &gpu.context.device,
                queue: &gpu.context.queue,
                surface_format: gpu.surface_format,
                config: &config,
            };
            let scene = self.app.on_setup(&mut app_ctx);

            let renderer = Renderer::new(
                &gpu.context.device,
                &gpu.context.queue,
                gpu.surface_format,
                &scene,
                size.width,
                size.height,
                config.sample_count,
            );

            self.camera.aspect_ratio = size.width as f32 / size.height.max(1) as f32;
            self.window = Some(window);
            self.gpu = Some(gpu);
            self.egui = Some(egui);
            self.renderer = Some(renderer);
            self.scene = scene;
        });
        self.setup_done = true;
    }

    fn run_ready(&mut self) {
        self.app
            .on_ready(&self.scene, &mut self.camera);
        self.ready_done = true;
    }
}

impl<A: EngvisApp + 'static> ApplicationHandler for AppRunner<A> {
    fn new_events(&mut self, _el: &ActiveEventLoop, _cause: winit::event::StartCause) {}

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.setup_done {
            self.run_setup(event_loop);
        }
        if self.setup_done && !self.ready_done {
            self.run_ready();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else { return };

        // Check for close/exit first
        if matches!(event, WindowEvent::CloseRequested) {
            event_loop.exit();
            return;
        }

        // Let the app intercept the event
        match self.app.on_event(&event) {
            EventHandling::Consumed => {
                self.input.egui_wants_pointer = true;
            }
            EventHandling::NoCamera | EventHandling::Default => {}
        }

        // Process input for camera
        let _size = window.inner_size();
        let input = &mut self.input;
        match &event {
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                input.scroll_delta += scroll;
            }
            WindowEvent::MouseInput { button, state, .. } => {
                match (button, state) {
                    (winit::event::MouseButton::Left, winit::event::ElementState::Pressed) => input.left_mouse_down = true,
                    (winit::event::MouseButton::Left, winit::event::ElementState::Released) => input.left_mouse_down = false,
                    (winit::event::MouseButton::Right, winit::event::ElementState::Pressed) => input.right_mouse_down = true,
                    (winit::event::MouseButton::Right, winit::event::ElementState::Released) => input.right_mouse_down = false,
                    (winit::event::MouseButton::Middle, winit::event::ElementState::Pressed) => input.middle_mouse_down = true,
                    (winit::event::MouseButton::Middle, winit::event::ElementState::Released) => input.middle_mouse_down = false,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                input.cursor_x = position.x;
                input.cursor_y = position.y;
            }
            _ => {}
        }

        // Feed event to egui
        if let Some(egui) = &mut self.egui {
            let response = egui.state.on_window_event(window, &event);
            self.input.egui_wants_pointer = response.consumed;
        }

        if let WindowEvent::Resized(new_size) = event {
            if let Some(gpu) = &mut self.gpu {
                gpu.resize(new_size.width, new_size.height);
            }
            if let Some(renderer) = &mut self.renderer
                && let Some(gpu) = &self.gpu {
                    renderer.resize(&gpu.context.device, new_size.width, new_size.height);
                }
            self.camera.aspect_ratio = new_size.width as f32 / new_size.height.max(1) as f32;
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        let Some(window) = &self.window else { return };
        let Some(gpu) = &self.gpu else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(egui) = &mut self.egui else { return };
        let size = window.inner_size();

        // FPS counter
        self.fps_ct += 1;
        let dt = self.fps_ts.elapsed().as_secs_f32();
        if dt >= 1.0 {
            self.fps_v = self.fps_ct as f32 / dt;
            self.fps_ct = 0;
            self.fps_ts = Instant::now();
        }

        let raw_input = egui.take_input(window);
        self.scene_dirty = false;

        let device = &gpu.context.device;
        let queue = &gpu.context.queue;

        // Build FrameCtx
        let mut frame = FrameCtx {
            scene: &mut self.scene,
            camera: &mut self.camera,
            render_state: &mut self.render_state,
            egui_wants_pointer: false,
            cursor_x: self.input.cursor_x,
            cursor_y: self.input.cursor_y,
            fps: self.fps_v,
            viewport: &mut self.viewport_rect,
            scene_dirty: &mut self.scene_dirty,
            device,
            queue,
            surface_format: gpu.surface_format,
            texture_cache: &mut renderer.texture_cache,
        };

        // Phase 1: egui UI
        let full_output = egui.context.run(raw_input, |egui_ctx| {
            // User draws their panels (SidePanel, TopBottomPanel, etc.)
            self.app.ui(egui_ctx, &mut frame);

            // After user ui(): check if any egui widget (resize handle, slider,
            // text field, etc.) wants the pointer. Only reliable check in
            // non-winit-integrated mode since per-event consumed flags don't
            // cover ongoing drags (e.g. panel resize handles).
            frame.egui_wants_pointer = frame.egui_wants_pointer
                || egui_ctx.wants_pointer_input();

            // Auto-generated CentralPanel to capture the 3D viewport rect.
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(egui_ctx, |ui| {
                    let rect = ui.max_rect();
                    let ppp = ui.ctx().pixels_per_point();
                    frame.viewport.min_x = (rect.min.x * ppp) as f64;
                    frame.viewport.min_y = (rect.min.y * ppp) as f64;
                    frame.viewport.max_x = (rect.max.x * ppp) as f64;
                    frame.viewport.max_y = (rect.max.y * ppp) as f64;
                });
        });

        // Sync wants_pointer back to input state for camera handling
        self.input.egui_wants_pointer = self.input.egui_wants_pointer
            || frame.egui_wants_pointer;

        // Phase 2: deferred frame work
        self.app.on_frame(&mut frame);

        renderer.set_state(&self.render_state);

        // Re-upload scene if dirty
        if self.scene_dirty {
            renderer.upload_scene(device, queue, &self.scene);
        }

        // Apply input to camera
        self.input.viewport_rect = self.viewport_rect.clone();
        self.input
            .apply_to_camera(&mut self.camera, [size.width, size.height]);

        // Render
        let Some(output) = gpu.get_current_texture() else {
            return;
        };

        let scene = &self.scene;
        let camera = &self.camera;

       let (cmd, output) = render_egui(
           egui,
           device,
           queue,
           window,
           output,
           full_output,
           |view, encoder| {
               renderer.render_scene_pass(device, queue, view, encoder, scene, camera, size.width, size.height);
               renderer.render_post_process(device, encoder, view, size.width, size.height);
           },
       );

        queue.submit(std::iter::once(cmd));
        output.present();
        window.request_redraw();
    }
}

// ── Public API ────────────────────────────────────────────────
/// Run an interactive 3D viewer with a custom app.
///
/// Blocks until the window is closed.
pub fn run<A: EngvisApp + 'static>(app: A) {
    let event_loop = EventLoop::new().unwrap();
    let mut runner = AppRunner::new(app);
    event_loop.run_app(&mut runner).unwrap();
}
