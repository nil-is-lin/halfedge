use std::sync::Arc;
use winit::event_loop::ActiveEventLoop;
use winit::window::Window;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter: wgpu::Adapter,
    pub instance: wgpu::Instance,
}

pub struct GpuResources {
    pub context: GpuContext,
    pub surface_format: wgpu::TextureFormat,
}

impl GpuResources {
    pub async fn new(window: Arc<Window>) -> Self {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Request POLYGON_MODE_LINE for wireframe rendering
        let required_features = adapter
            .features()
            .intersection(wgpu::Features::POLYGON_MODE_LINE);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits: wgpu::Limits::default(),
                label: Some("engvis device"),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8UnormSrgb))
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let size = window.inner_size();

        // Prefer vsync modes to avoid burning CPU at uncapped framerate.
        // Mailbox (fast vsync) > Fifo (standard vsync) > fallback (first available).
        let present_mode = *surface_caps
            .present_modes
            .iter()
            .find(|m| matches!(m, wgpu::PresentMode::Mailbox))
            .or_else(|| {
                surface_caps
                    .present_modes
                    .iter()
                    .find(|m| matches!(m, wgpu::PresentMode::Fifo))
            })
            .unwrap_or(&surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let context = GpuContext {
            device,
            queue,
            surface,
            config,
            adapter,
            instance,
        };

        Self {
            context,
            surface_format,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.context.config.width = width;
        self.context.config.height = height;
        self.context
            .surface
            .configure(&self.context.device, &self.context.config);
    }

    pub fn get_current_texture(&self) -> Option<wgpu::SurfaceTexture> {
        self.context.surface.get_current_texture().ok()
    }
}

pub async fn create_window_and_gpu(
    event_loop: &ActiveEventLoop,
    title: &str,
    width: u32,
    height: u32,
) -> (Arc<Window>, GpuResources) {
    let window = event_loop
        .create_window(
            Window::default_attributes()
                .with_title(title)
                .with_visible(true)
                .with_inner_size(winit::dpi::LogicalSize::new(width, height)),
        )
        .unwrap();

    let window = Arc::new(window);
    let gpu = GpuResources::new(window.clone()).await;

    (window, gpu)
}
