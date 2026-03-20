use std::{
    ffi::{CStr, CString, c_char},
    path::Path,
    sync::{LazyLock, Mutex},
};

use burn::{backend::Autodiff, optim::AdamConfig};
use circular_buffer::CircularBuffer;
use log::{error, info};

use crate::{
    ImageData, ImageLabel,
    batcher::EyeDataBatcher,
    ffi::{
        infer::{load_model, run_inference},
        train::train,
    },
    frame_correlator::align_frames,
    loader::FileReader,
    models::{MultiInputMergedMicroChad, MultiInputMergedMicroChadConfig},
    trainer::TrainingConfig,
};

pub mod infer;
pub mod train;

type B = burn::backend::Wgpu;

#[derive(Debug)]
struct InferenceState {
    model: MultiInputMergedMicroChad<B>,
    batcher: EyeDataBatcher,
}

static STATE: LazyLock<Mutex<Result<InferenceState, String>>> =
    LazyLock::new(|| Mutex::new(Err("Model is not yet initialised".into())));

static LEFT_EYE_FRAMES: LazyLock<Mutex<CircularBuffer<4, ImageData>>> =
    LazyLock::new(|| Mutex::new(CircularBuffer::new()));
static RIGHT_EYE_FRAMES: LazyLock<Mutex<CircularBuffer<4, ImageData>>> =
    LazyLock::new(|| Mutex::new(CircularBuffer::new()));

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct ModelOutput {
    pub pitch_l: f32,
    pub yaw_l: f32,
    pub blink_l: f32,
    pub eyebrow_l: f32,
    pub eyewide_l: f32,
    pub pitch_r: f32,
    pub yaw_r: f32,
    pub blink_r: f32,
    pub eyebrow_r: f32,
    pub eyewide_r: f32,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum CallbackType {
    Batch = 0,
    Epoch = 1,
    Finished = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TrainingDataCallback {
    callback_type: CallbackType,
    low: i32,
    high: i32,
    loss: f32,
}

#[repr(C)]
pub struct ModelOutputResult {
    is_error: bool,
    value: ModelOutputValue,
}

#[repr(C)]
pub union ModelOutputValue {
    model_output: ModelOutput,
    error_message: *mut c_char,
}

impl ModelOutputResult {
    pub fn from_error(message: String) -> Self {
        Self {
            is_error: true,
            value: ModelOutputValue {
                error_message: CString::new(message).unwrap().into_raw(),
            },
        }
    }

    pub fn from_output(output: ModelOutput) -> Self {
        Self {
            is_error: false,
            value: ModelOutputValue {
                model_output: output,
            },
        }
    }

    pub fn is_error(&self) -> bool {
        self.is_error
    }

    pub fn get_error_message(&self) -> Option<String> {
        if self.is_error() {
            unsafe {
                let c_str = CStr::from_ptr(self.value.error_message);
                Some(c_str.to_string_lossy().into_owned())
            }
        } else {
            None
        }
    }

    pub fn get_model_output(&self) -> Option<ModelOutput> {
        if !self.is_error() {
            unsafe { Some(self.value.model_output) }
        } else {
            None
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn loadModel(path_ptr: *const c_char) -> ModelOutputResult {
    let c_str = unsafe { CStr::from_ptr(path_ptr) };

    let c_str_str = c_str.to_str().unwrap_or_default();
    let decoded = urlencoding::decode(c_str_str).unwrap().to_string();
    let file_path = Path::new(&decoded);

    // Remove file extension for model loading
    let file_stem = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .split('.')
        .next()
        .unwrap_or_default();
    let model_path = file_path.with_file_name(file_stem);

    println!("Loading model from path: {:?} ({:?})", c_str, model_path);

    match load_model(model_path.to_str().unwrap_or_default()) {
        Ok(()) => ModelOutputResult::from_output(ModelOutput::default()),
        Err(err) => ModelOutputResult::from_error(err),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn infer(left: &[u8; 128 * 128], right: &[u8; 128 * 128]) -> ModelOutputResult {
    match run_inference(left, right) {
        Ok(output) => ModelOutputResult::from_output(output),
        Err(err) => ModelOutputResult::from_error(err),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn freeModelOutputResult(result: ModelOutputResult) {
    if result.is_error() {
        unsafe {
            let _ = CString::from_raw(result.value.error_message);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn trainModel(
    usercal_path: *const c_char,
    output_path: *const c_char,
    cb: extern "C" fn(epoch: TrainingDataCallback) -> (),
) {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let usercal_path_cstr = unsafe { CStr::from_ptr(usercal_path) };
    let output_path_cstr = unsafe { CStr::from_ptr(output_path) };

    type AutodiffBackend = Autodiff<B>;
    let device = Default::default();
    let mut reader = FileReader::new();

    info!(
        "Starting training with usercal path: {:?} and output path: {:?}",
        usercal_path_cstr, output_path_cstr
    );

    let aligned_frames =
        match reader.read_capture_file(usercal_path_cstr.to_str().unwrap(), true, true, 0, 0) {
            Ok(_) => {
                info!("Finished processing capture file");
                info!(
                    "Read {} left eye frames, {} right eye frames, and {} label frames",
                    reader.left_eye_frames.len(),
                    reader.right_eye_frames.len(),
                    reader.label_frames.len()
                );

                let mut left_frames: Vec<(u64, ImageData)> = reader
                    .left_eye_frames
                    .iter()
                    .map(|(ts, img)| (*ts, img.clone()))
                    .collect();
                let mut right_frames: Vec<(u64, ImageData)> = reader
                    .right_eye_frames
                    .iter()
                    .map(|(ts, img)| (*ts, img.clone()))
                    .collect();
                let mut label_frames: Vec<(u64, ImageLabel)> = reader
                    .label_frames
                    .iter()
                    .map(|(ts, label)| (*ts, label.clone()))
                    .collect();

                left_frames.sort_by_key(|(ts, _)| *ts);
                right_frames.sort_by_key(|(ts, _)| *ts);
                label_frames.sort_by_key(|(ts, _)| *ts);

                align_frames(left_frames, right_frames, label_frames)
            }
            Err(e) => {
                error!("Failed to read capture file: {e}");
                return;
            }
        };

    train::<AutodiffBackend>(
        output_path_cstr.to_str().unwrap(),
        TrainingConfig::new(MultiInputMergedMicroChadConfig::new(5), AdamConfig::new()),
        device,
        aligned_frames,
        cb,
    );
    cb(TrainingDataCallback {
        callback_type: CallbackType::Finished,
        low: 0,
        high: 0,
        loss: 0.0,
    });
}
