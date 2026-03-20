use std::{
    ffi::{CStr, CString, c_char},
    path::Path,
    sync::{LazyLock, Mutex},
};

use burn::{
    backend::Autodiff,
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::{
            Dataset, InMemDataset,
            transform::{MapperDataset, PartialDataset, ShuffledDataset, WindowsDataset},
        },
    },
    module::{AutodiffModule, Module},
    nn::loss::MseLoss,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::ToElement,
    record::{Recorder, SensitiveCompactRecorder},
    tensor::backend::AutodiffBackend,
};
use circular_buffer::CircularBuffer;
use histogram_equalization::hist_equal_hsv_rgb;
use image::{DynamicImage, GenericImageView, RgbImage};
use log::{error, info};

use crate::{
    ImageData, ImageLabel,
    batcher::{DatasetInfo, EyeDataBatcher, WindowedFrame},
    frame_correlator::{AlignedFrame, ImageToTensor, align_frames, create_dataset},
    loader::FileReader,
    models::{MultiInputMergedMicroChad, MultiInputMergedMicroChadConfig},
    trainer::TrainingConfig,
};

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

fn setup_inference(model_path: &str) -> Result<InferenceState, String> {
    let device = Default::default();

    let config = match TrainingConfig::load(format!("{model_path}_config.json")) {
        Ok(config) => config,
        Err(err) => {
            return Err(format!("Failed to load training config for model: {err}"));
        }
    };
    let record = match SensitiveCompactRecorder::new().load(model_path.into(), &device) {
        Ok(record) => record,
        Err(err) => {
            return Err(format!("Failed to load trained model: {err}"));
        }
    };

    let model = config.model.init::<B>(&device).load_record(record);

    let batcher = EyeDataBatcher {
        training: false,
        dataset_info: DatasetInfo::default(),
    };

    Ok(InferenceState { model, batcher })
}

fn equalize_histogram(data: &[u8; 128 * 128]) -> ImageData {
    let img = DynamicImage::ImageLuma8(
        ImageData::from_raw(128, 128, data.to_vec()).expect("Input image should be valid"),
    );
    let dimensions = img.dimensions();
    let channels = 3;
    let stride = dimensions.0 as usize * channels;
    let mut dst_bytes: Vec<u8> = vec![0; stride * dimensions.1 as usize];
    let src_bytes = img.into_rgb8().into_raw();
    hist_equal_hsv_rgb(
        &src_bytes,
        stride as u32,
        &mut dst_bytes,
        stride as u32,
        dimensions.0,
        dimensions.1,
        128,
    );

    DynamicImage::ImageRgb8(
        RgbImage::from_raw(dimensions.0, dimensions.1, dst_bytes)
            .expect("histogram equalized image was invalid"),
    )
    .into_luma8()
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

pub fn load_model(model_path: &str) -> Result<(), String> {
    match setup_inference(model_path) {
        Ok(inference_state) => {
            println!("Model loaded successfully from path: {model_path}");
            *STATE.lock().unwrap() = Ok(inference_state);
            return Ok(());
        }
        Err(err) => {
            println!("Failed to load model from path: {model_path}, error: {err}");
            *STATE.lock().unwrap() = Err(err.clone());
            return Err(err);
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

pub fn run_inference(
    left: &[u8; 128 * 128],
    right: &[u8; 128 * 128],
) -> Result<ModelOutput, String> {
    // let start_time = std::time::Instant::now();
    let device = Default::default();

    let state = STATE.lock().unwrap();
    if let Err(err) = state.as_ref() {
        return Err(err.clone());
    }

    let state = state.as_ref().unwrap();
    let mut left_frames = LEFT_EYE_FRAMES.lock().unwrap();
    let mut right_frames = RIGHT_EYE_FRAMES.lock().unwrap();

    left_frames.push_back(equalize_histogram(left));
    right_frames.push_back(equalize_histogram(right));

    let frame = WindowedFrame {
        label: ImageLabel::default(),
        left_eye: [
            left_frames
                .get(0)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(1)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(2)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            left_frames
                .get(3)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
        ],
        right_eye: [
            right_frames
                .get(0)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(1)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(2)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
            right_frames
                .get(3)
                .cloned()
                .unwrap_or_else(|| ImageData::new(128, 128)),
        ],
        timestamp: 0,
    };

    let batch = state.batcher.batch(vec![frame], &device);
    let output = state.model.forward(batch.images);

    let output = output.to_data().to_vec().unwrap();

    // println!(
    //     "Inference took {:.2?} ({} fps)",
    //     std::time::Instant::now() - start_time,
    //     1.0 / (std::time::Instant::now() - start_time).as_secs_f32()
    // );

    // println!(
    //     "Model output: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
    //     output[0],
    //     output[1],
    //     output[2],
    //     output[3],
    //     output[4],
    //     output[5],
    //     output[6],
    //     output[7],
    //     output[8],
    //     output[9]
    // );

    Ok(ModelOutput {
        pitch_l: output[0],
        yaw_l: output[1],
        blink_l: output[2],
        eyebrow_l: output[3],
        eyewide_l: output[4],
        pitch_r: output[5],
        yaw_r: output[6],
        blink_r: output[7],
        eyebrow_r: output[8],
        eyewide_r: output[9],
    })
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

pub fn train<B: AutodiffBackend>(
    model_name: &str,
    config: TrainingConfig,
    device: B::Device,
    frames: Vec<AlignedFrame>,
    cb: extern "C" fn(epoch: TrainingDataCallback) -> (),
) {
    config
        .save(format!("{model_name}_config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let dataset_info = create_dataset(&frames);

    let batcher = EyeDataBatcher {
        training: true,
        dataset_info,
    };
    let test_batcher = EyeDataBatcher {
        training: false,
        dataset_info,
    };

    let get_data = |frames, start, end| {
        let frames = ShuffledDataset::new(
            WindowsDataset::new(InMemDataset::new(frames), 4),
            config.seed,
        );
        let len = frames.len();

        MapperDataset::new(
            PartialDataset::new(frames, len * start / 10, len * end / 10),
            ImageToTensor,
        )
    };

    let dataset_train = get_data(frames.clone(), 0, 8);
    let train_size = dataset_train.len();
    let dataset_test = get_data(frames, 8, 10);
    let test_size = dataset_test.len();

    let dataloader_train = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(test_batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_test);

    let mut model = config.model.init::<B>(&device);
    let mut optim = config.optimizer.init();

    for epoch in 1..config.num_epochs + 1 {
        cb(TrainingDataCallback {
            callback_type: CallbackType::Epoch,
            low: epoch as i32,
            high: config.num_epochs as i32,
            loss: 0.0,
        });

        // Implement our training loop.
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let loss = MseLoss::new().forward(
                output.clone(),
                batch.targets.clone(),
                burn::nn::loss::Reduction::Auto,
            );

            info!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3}",
                epoch,
                iteration,
                loss.clone().into_scalar(),
            );
            cb(TrainingDataCallback {
                callback_type: CallbackType::Batch,
                low: (iteration * config.batch_size) as i32,
                high: (train_size + test_size) as i32,
                loss: loss.clone().into_scalar().to_f32(),
            });

            // Gradients for the current backward pass
            let grads = loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.learning_rate, model, grads);
        }

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for (iteration, batch) in dataloader_test.iter().enumerate() {
            let output = model_valid.forward(batch.images);
            let loss = MseLoss::new().forward(
                output.clone(),
                batch.targets.clone(),
                burn::nn::loss::Reduction::Auto,
            );

            info!(
                "[Valid - Epoch {} - Iteration {}] Loss {}",
                epoch,
                iteration,
                loss.clone().into_scalar()
            );
            cb(TrainingDataCallback {
                callback_type: CallbackType::Batch,
                low: (train_size + (iteration * config.batch_size)) as i32,
                high: (train_size + test_size) as i32,
                loss: loss.clone().into_scalar().to_f32(),
            });
        }
    }

    model
        .save_file(model_name, &SensitiveCompactRecorder::new())
        .expect("Trained model should be saved successfully");
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
