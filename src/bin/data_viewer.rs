use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, Sender, channel},
};

use babble_model::{
    ImageData, ImageLabel,
    ffi::{ModelOutput, infer::{load_model, run_inference}},
    loader::FileReader,
};
use eframe::egui::{self, Vec2};
use egui_notify::Toasts;
use log::{error, info};

fn load_file(path: &str) -> Result<FileData, String> {
    let mut reader = FileReader::new();

    info!("Loading file: {}", path);

    reader
        .read_capture_file(path, false, true, 0, 0)
        .map_err(|e| format!("Could not load file: {e}"))?;

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

    let data = left_frames
        .iter()
        .zip(right_frames.iter())
        .zip(label_frames.iter())
        .map(|(((ts_l, left), (ts_r, right)), (ts_label, label))| {
            (
                (*ts_l, left.clone()),
                (*ts_r, right.clone()),
                (*ts_label, label.clone()),
            )
        })
        .collect();

    Ok(data)
}

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 670.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Frame Viewer",
        options,
        Box::new(|cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(DataViewer {
                data: Vec::new(),
                text_channel: channel(),
                image_index: 0,
                last_infered: None,
                toasts: Arc::new(Mutex::new(Toasts::default())),
                model_inited: Arc::new(Mutex::new(false)),
            }))
        }),
    )
}

type FileData = Vec<((u64, ImageData), (u64, ImageData), (u64, ImageLabel))>;

struct DataViewer {
    text_channel: (Sender<FileData>, Receiver<FileData>),
    data: FileData,
    image_index: usize,
    last_infered: Option<(usize, ModelOutput)>,
    toasts: Arc<Mutex<Toasts>>,
    model_inited: Arc<Mutex<bool>>,
}

fn image_to_bytes(image: &ImageData) -> Vec<u8> {
    let mut buffer = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buffer, 80);

    let _ = image.write_with_encoder(encoder);

    buffer
}

impl eframe::App for DataViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Ok(data) = self.text_channel.1.try_recv() {
                self.data = data;
            }

            ui.horizontal(|ui| {
                if ui.button("📂 Open data file").clicked() {
                    let sender = self.text_channel.0.clone();
                    let task = rfd::AsyncFileDialog::new().pick_file();

                    let ctx = ui.ctx().clone();
                    std::thread::spawn(move || futures::executor::block_on(async move {
                        let file = task.await;
                        if let Some(Ok(file)) = file.map(|f| load_file(f.path().to_str().unwrap_or_default())) {
                            let _ = sender.send(file);
                            ctx.request_repaint();
                        }
                    }));
                }

                if ui.button("📂 Open model file").clicked() {
                    let task = rfd::AsyncFileDialog::new().pick_file();

                    let ctx = ui.ctx().clone();
                    let toasts = self.toasts.clone();
                    let model_inited = self.model_inited.clone();
                    std::thread::spawn(move || futures::executor::block_on(async move {
                        let file = task.await;
                        if let Some(file) = file {
                            let file_path = file.path();

                            // Remove file extension for model loading
                            let file_stem = file_path.file_name().and_then(|s| s.to_str()).unwrap_or_default().split('.').next().unwrap_or_default();
                            let model_path = file_path.with_file_name(file_stem);

                            if let Err(err) = load_model(model_path.to_str().unwrap_or_default()) {
                                toasts.lock().unwrap().error(
                                    format!("Failed to load model: {err}")
                                );
                                error!("Failed to load model: {err}");
                                *model_inited.lock().unwrap() = false;
                            } else {
                                toasts.lock().unwrap().success(
                                    "Model loaded successfully"
                                );
                                *model_inited.lock().unwrap() = true;
                            }
                            ctx.request_repaint();
                        }
                    }));
                }
            });
            self.toasts.lock().unwrap().show(&ctx);

            if self.data.is_empty() {
                ui.label("No data loaded");
                return;
            }

            if self.image_index >= self.data.len() {
                self.image_index = 0;
            }

            let left = &self.data[self.image_index].0;
            let right = &self.data[self.image_index].1;
            let label = &self.data[self.image_index].2;

            if let Some((last_index, _)) = self.last_infered {
                if last_index != self.image_index {
                    self.last_infered = None;
                }
            }

            if self.last_infered.is_none() && *self.model_inited.lock().unwrap() {
                let mut left_buf: [u8; 128*128] = [0; 128*128];
                let mut right_buf: [u8; 128*128] = [0; 128*128];

                left_buf.copy_from_slice(left.1.as_raw());
                right_buf.copy_from_slice(right.1.as_raw());

                match run_inference(&left_buf, &right_buf) {
                    Ok(inference) => {
                        self.last_infered = Some((self.image_index, ModelOutput {
                            pitch_l: inference.pitch_l * 90.0 - 45.0,
                            yaw_l: inference.yaw_l * 90.0 - 45.0,
                            blink_l: 1.0 - inference.blink_l,
                            eyebrow_l: inference.eyebrow_l,
                            eyewide_l: inference.eyewide_l,
                            pitch_r: inference.pitch_r * 90.0 - 45.0,
                            yaw_r: inference.yaw_r * 90.0 - 45.0,
                            blink_r: 1.0 - inference.blink_r,
                            eyebrow_r: inference.eyebrow_r,
                            eyewide_r: inference.eyewide_r,
                        }))
                    },
                    Err(err) => {
                        error!("Inference error: {err}");
                    }
                }
            }

            let inference = &self.last_infered.as_ref().map(|(_, inf)| *inf).unwrap_or_default();

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.add(
                        egui::Image::from_bytes(
                            format!("bytes://{}_l.png", self.image_index),
                            image_to_bytes(&left.1),
                        )
                        .fit_to_exact_size(Vec2::new(400.0, 400.0))
                        .corner_radius(10),
                    );
                    ui.label(format!("TS: {}", left.0));
                });
                ui.vertical(|ui| {
                    ui.add(
                        egui::Image::from_bytes(
                            format!("bytes://{}_r.png", self.image_index),
                            image_to_bytes(&right.1),
                        )
                        .fit_to_exact_size(Vec2::new(400.0, 400.0))
                        .corner_radius(10),
                    );
                    ui.label(format!("TS: {}", right.0));
                });
            });

            ui.separator();

            ui.spacing_mut().slider_width = ui.available_width() / 2.0;
            ui.add(
                    egui::Slider::new(&mut self.image_index, 0..=(self.data.len() - 1))
                        .clamping(egui::SliderClamping::Always)
                        .drag_value_speed(1.0)
                );

            ui.label(format!(
                "TS ({:.0}):\nleft_pitch={:.1} (Infered: {:.1})\nleft_yaw={:.1} (Infered: {:.1})\nright_pitch={:.1} (Infered: {:.1})\nright_yaw={:.1} (Infered: {:.1})\nleft_lid={:.1} (Infered: {:.1})\nright_lid={:.1} (Infered: {:.1})\nbrow_raise={:.1}\nbrow_angry={:.1} (Infered: {:.1}, {:.1})\nwiden={:.1} (Infered: {:.1}, {:.1})\nsquint={:.1}\ndilate={:.1}\nstate={:b}",
                label.0,
                label.1.left_eye_pitch,
                inference.pitch_l,
                label.1.left_eye_yaw,
                inference.yaw_l,
                label.1.right_eye_pitch,
                inference.pitch_r,
                label.1.right_eye_yaw,
                inference.yaw_r,
                label.1.routine_left_lid,
                inference.blink_l,
                label.1.routine_right_lid,
                inference.blink_r,
                label.1.routine_brow_raise,
                label.1.routine_brow_angry,
                inference.eyebrow_l,
                inference.eyebrow_r,
                label.1.routine_widen,
                inference.eyewide_l,
                inference.eyewide_r,
                label.1.routine_squint,
                label.1.routine_dilate,
                label.1.routine_state,
            ));
        });
    }
}
