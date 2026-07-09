use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;

use engvis_core::{InputState, OrbitCamera};

pub struct EventContext<'a> {
    pub input: &'a mut InputState,
    pub camera: &'a mut OrbitCamera,
    pub window_width: u32,
    pub window_height: u32,
}

pub enum EventResult {
    Consumed,
    NotConsumed,
    Exit,
}

pub fn handle_window_event(
    event_loop: &ActiveEventLoop,
    event: &WindowEvent,
    ctx: &mut EventContext,
) -> EventResult {
    match event {
        WindowEvent::CloseRequested => {
            event_loop.exit();
            EventResult::Exit
        }
        WindowEvent::Resized(size) => {
            ctx.window_width = size.width;
            ctx.window_height = size.height;
            EventResult::Consumed
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let scroll = match delta {
                MouseScrollDelta::LineDelta(_, y) => *y,
                MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
            };
            ctx.input.scroll_delta += scroll;
            EventResult::Consumed
        }
        WindowEvent::MouseInput { button, state, .. } => {
            match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => ctx.input.left_mouse_down = true,
                (MouseButton::Left, ElementState::Released) => ctx.input.left_mouse_down = false,
                (MouseButton::Right, ElementState::Pressed) => ctx.input.right_mouse_down = true,
                (MouseButton::Right, ElementState::Released) => ctx.input.right_mouse_down = false,
                (MouseButton::Middle, ElementState::Pressed) => ctx.input.middle_mouse_down = true,
                (MouseButton::Middle, ElementState::Released) => {
                    ctx.input.middle_mouse_down = false
                }
                _ => {}
            }
            EventResult::Consumed
        }
        WindowEvent::CursorMoved { position, .. } => {
            ctx.input.cursor_x = position.x;
            ctx.input.cursor_y = position.y;
            EventResult::NotConsumed
        }
        _ => EventResult::NotConsumed,
    }
}
