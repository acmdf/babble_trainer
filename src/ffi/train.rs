use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::{Dataset, InMemDataset, transform::{MapperDataset, PartialDataset, ShuffledDataset, WindowsDataset}}}, module::{AutodiffModule, Module}, nn::loss::MseLoss, optim::{GradientsParams, Optimizer}, prelude::ToElement, record::SensitiveCompactRecorder, tensor::backend::AutodiffBackend};
use log::info;

use crate::{batcher::EyeDataBatcher, ffi::{CallbackType, TrainingDataCallback}, frame_correlator::{AlignedFrame, ImageToTensor, create_dataset}, trainer::TrainingConfig};

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
