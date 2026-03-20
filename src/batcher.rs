use burn::{Tensor, data::dataloader::batcher::Batcher, prelude::Backend};

use crate::{ImageData, ImageLabel};

#[derive(Clone, Debug, Default, Copy)]
pub struct DatasetInfo {
    pub pitch_min_l: f32,
    pub pitch_max_l: f32,
    pub yaw_min_l: f32,
    pub yaw_max_l: f32,

    pub pitch_min_r: f32,
    pub pitch_max_r: f32,
    pub yaw_min_r: f32,
    pub yaw_max_r: f32,

    pub label_count: usize,
}

#[derive(Clone, Debug)]
pub struct EyeDataBatcher {
    pub training: bool,
    pub dataset_info: DatasetInfo,
}

#[derive(Clone, Debug)]
pub struct EyeDataBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct WindowedFrame {
    pub label: ImageLabel,
    pub left_eye: [ImageData; 4],
    pub right_eye: [ImageData; 4],
    pub timestamp: u64,
}

fn apply_spatial_transform(
    image: [ImageData; 8],
    _max_shift: i32,
    _max_rotation: f32,
    _max_scale: f32,
) -> [ImageData; 8] {
    // Placeholder for actual spatial transformation logic
    image.clone()
}

fn apply_intensity_transformation(
    image: [ImageData; 8],
    _brightness_range: f32,
    _contrast_range: f32,
) -> [ImageData; 8] {
    // Placeholder for actual intensity transformation logic
    image.clone()
}

fn apply_blur(image: [ImageData; 8], _max_kernel_size: i32) -> [ImageData; 8] {
    // Placeholder for actual blur transformation logic
    image.clone()
}

impl<B: Backend> Batcher<B, WindowedFrame, EyeDataBatch<B>> for EyeDataBatcher {
    fn batch(&self, mut items: Vec<WindowedFrame>, device: &B::Device) -> EyeDataBatch<B> {
        fn handle_data(
            pitch: f32,
            yaw: f32,
            lid: f32,
            squint: f32,
            widen: f32,
            angry: f32,
        ) -> [f32; 5] {
            let lid_closed = lid < 0.5;

            let mut norm_pitch = ((pitch + 45.0) / 90.0).clamp(0.0, 1.0);
            let mut norm_yaw = ((yaw + 45.0) / 90.0).clamp(0.0, 1.0);

            if lid_closed || squint > 0.5 || widen > 0.5 || angry > 0.5 {
                norm_pitch = 0.5;
                norm_yaw = 0.5;
            }

            // Order is [pitch, yaw, lid_closed, eyebrow, eyewide]

            return [
                norm_pitch,
                norm_yaw,
                if lid_closed {
                    1.0
                } else if squint > 0.5 {
                    0.5
                } else {
                    0.0
                },
                if angry > 0.5 { 1.0 } else { ((1.0 - norm_pitch)-0.5).clamp(0.0, 1.0) },
                if widen > 0.5 { 1.0 } else { 0.0 },
            ];
        }

        // Pull these out first so that we can pull the image buffers out without copies
        let targets = items
            .iter()
            .map(|item| {
                let left_values = handle_data(
                    item.label.left_eye_pitch,
                    item.label.left_eye_yaw,
                    item.label.routine_left_lid,
                    item.label.routine_squint,
                    item.label.routine_widen,
                    item.label.routine_brow_angry,
                );
                let right_values = handle_data(
                    item.label.right_eye_pitch,
                    item.label.right_eye_yaw,
                    item.label.routine_right_lid,
                    item.label.routine_squint,
                    item.label.routine_widen,
                    item.label.routine_brow_angry,
                );

                return Tensor::<B, 1>::from_data(
                    [
                        left_values[0],
                        left_values[1],
                        left_values[2],
                        left_values[3],
                        left_values[4],
                        right_values[0],
                        right_values[1],
                        right_values[2],
                        right_values[3],
                        right_values[4],
                    ],
                    device,
                );
            })
            .map(|tensor| tensor.reshape([1, -1]))
            .collect();

        let images = items
            .drain(..)
            .map(|item| {
                let [l_0, l_1, l_2, l_3] = item.left_eye;
                let [r_0, r_1, r_2, r_3] = item.right_eye;

                // Intersperse Left and Right channels, with Left channels in even indices and Right channels in odd indices
                let mut image = [l_0, r_0, l_1, r_1, l_2, r_2, l_3, r_3];

                if self.training {
                    if rand::random::<f32>() < 0.2 {
                        image = apply_spatial_transform(image, 24, 10.0, 0.1);
                    }

                    if rand::random::<f32>() < 0.3 {
                        image = apply_intensity_transformation(image, 0.1, 0.5);
                    }

                    if rand::random::<f32>() < 0.2 {
                        image = apply_blur(image, 5);
                    }
                }

                Tensor::stack(
                    image
                        .iter()
                        .map(|img| {
                            let data = img.as_raw();
                            let height = img.height() as usize;
                            let width = img.width() as usize;

                            Tensor::<B, 1>::from_data(&data[..], device).reshape([width, height])
                        })
                        .collect(),
                    0,
                )
            })
            .map(|tensor| {
                let [channels, height, width] = tensor.dims();

                tensor.reshape([1, channels, height, width])
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        EyeDataBatch { images, targets }
    }
}
