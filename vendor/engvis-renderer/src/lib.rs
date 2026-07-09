#![doc = "GPU rendering for engineering visualization.\n\nBuilds on `engvis-core` to provide wgpu-based rendering pipelines, egui integration, and glTF loading."]

pub mod gpu;
pub mod depth;
pub mod egui_integration;
pub mod event;
pub mod renderer;
pub mod mesh_renderer;
pub mod grid_renderer;
pub mod material_pipeline;
pub mod overlay_renderer;
pub mod lighting;
pub mod gltf_loader;
pub mod texture_cache;
pub mod custom_material;
pub mod postprocess;
pub mod app;

pub use gpu::{GpuContext, GpuResources, create_window_and_gpu};
pub use depth::DepthTexture;
pub use egui_integration::{EguiContext, render_egui};
pub use event::{EventContext, EventResult, handle_window_event};
pub use renderer::Renderer;
pub use grid_renderer::GridRenderer;
pub use mesh_renderer::MeshRenderer;
pub use material_pipeline::MaterialPipeline;
pub use overlay_renderer::OverlayRenderer;
pub use lighting::LightingBuffer;
pub use texture_cache::TextureCache;
pub use gltf_loader::load_gltf;
pub use custom_material::{CustomMaterial, BasicMaterial};
pub use app::{EngvisApp, AppCtx, FrameCtx, RunConfig, EventHandling, run};
