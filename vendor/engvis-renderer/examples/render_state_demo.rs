//! # RenderState ownership pattern (reference example)
//!
//! This example shows the **recommended** way to manage display / overlay
//! options when building an `EngvisApp` on top of `engvis-renderer`:
//!
//! 1. The **App owns a single `RenderState`** as the source of truth.
//! 2. UI widgets mutate `self.render_state` in place (one place to look).
//! 3. Every frame, `on_frame` pushes the whole struct to the renderer with
//!    `frame.set_render_state(...)` — one atomic, copy-able update.
//!
//! ## Why this and not scattered fields?
//!
//! `RenderState` is a `Copy` aggregate covering surface visibility, grid,
//! opacity, background color, IBL intensity, and the point/edge overlay
//! options. Pushing it wholesale each frame avoids the classic bug of
//! "I changed `edge_color` but forgot to sync it into `frame.render_state`" —
//! there is only one copy, so it can never drift.
//!
//! ## What does NOT belong in `RenderState`?
//!
//! Per-*node* appearance that is baked into the scene graph —
//! `PbrMaterial` (albedo / metallic / roughness) and a node's
//! `render_edges` flag — is applied when you build the `Scene`. Changing
//! those requires rebuilding the scene (set `frame.scene_dirty = true` or
//! return a new scene), which is a heavier operation than the per-frame
//! `RenderState` push. Keep node-level look in your `AppBuildSnapshot` /
//! scene-building code, not in `RenderState`.
//!
//! Run with: `cargo run -p engvis-renderer --example render_state_demo`

use engvis_core::{
    material::{EdgeRenderOptions, RenderState},
    scene::Scene,
};
use engvis_renderer::{AppCtx, EngvisApp, FrameCtx, RunConfig};

struct DemoApp {
    /// Single source of truth for display / overlay options.
    render_state: RenderState,
}

impl EngvisApp for DemoApp {
    fn config(&self) -> RunConfig {
        RunConfig {
            title: "RenderState demo".into(),
            ..Default::default()
        }
    }

    // An empty scene is enough to demonstrate the render-state pattern.
    fn on_setup(&mut self, _ctx: &mut AppCtx) -> Scene {
        Scene::default()
    }

    fn ui(&mut self, egui_ctx: &egui::Context, _frame: &mut FrameCtx) {
        egui::Window::new("Display").show(egui_ctx, |ui| {
            // Every widget edits `self.render_state` directly — no second copy.
            ui.checkbox(&mut self.render_state.show_surface, "Surface");
            ui.checkbox(&mut self.render_state.show_grid, "Grid");
            ui.checkbox(&mut self.render_state.vertex_opts.enabled, "Points");
            if self.render_state.vertex_opts.enabled {
                ui.horizontal(|ui| {
                    ui.label("Point color");
                    ui.color_edit_button_rgb(&mut self.render_state.vertex_opts.color);
                });
                ui.add(
                    egui::Slider::new(&mut self.render_state.vertex_opts.point_size, 1.0..=12.0)
                        .text("Point size"),
                );
            }
            ui.horizontal(|ui| {
                ui.label("Background");
                ui.color_edit_button_rgb(&mut self.render_state.background_color);
            });
            ui.add(
                egui::Slider::new(&mut self.render_state.env_intensity, 0.0..=3.0)
                    .text("Env intensity"),
            );

            // Edge overlay options live in RenderState too.
            ui.checkbox(&mut self.render_state.edge_opts.enabled, "Edges");
            if self.render_state.edge_opts.enabled {
                ui.horizontal(|ui| {
                    ui.label("Edge color");
                    ui.color_edit_button_rgb(&mut self.render_state.edge_opts.color);
                });
                ui.add(
                    egui::Slider::new(&mut self.render_state.edge_opts.line_width, 0.5..=10.0)
                        .text("Edge width (px)"),
                );
            }
        });
    }

    fn on_frame(&mut self, frame: &mut FrameCtx) {
        // Push the owned RenderState atomically. `RenderState` is `Copy`,
        // so the App keeps its own copy for the next frame / next UI edit.
        frame.set_render_state(self.render_state);
    }
}

fn main() {
    // A sensible initial state: edges on, neutral gray, white background.
    let initial = RenderState {
        edge_opts: EdgeRenderOptions {
            enabled: true,
            color: [0.35, 0.35, 0.35],
            line_width: 2.0,
        },
        background_color: [1.0, 1.0, 1.0],
        ..Default::default()
    };
    engvis_renderer::run(DemoApp {
        render_state: initial,
    });
}
