#![recursion_limit = "256"]

use image::{ImageBuffer, Luma};

pub mod batcher;
pub mod corruption_detector;
pub mod ffi;
pub mod frame_correlator;
pub mod loader;
pub mod models;
pub mod trainer;
pub mod utils;

pub type ImageData = ImageBuffer<Luma<u8>, Vec<u8>>;

#[derive(Debug, Clone, Default, Copy)]
pub struct ImageLabel {
    pub routine_pitch: f32,
    pub routine_yaw: f32,
    pub routine_distance: f32,
    pub routine_convergence: f32,
    pub fov_adjust_distance: f32,
    pub left_eye_pitch: f32,
    pub left_eye_yaw: f32,
    pub right_eye_pitch: f32,
    pub right_eye_yaw: f32,
    pub routine_left_lid: f32,
    pub routine_right_lid: f32,
    pub routine_brow_raise: f32,
    pub routine_brow_angry: f32,
    pub routine_widen: f32,
    pub routine_squint: f32,
    pub routine_dilate: f32,
    pub routine_state: i32,
}
