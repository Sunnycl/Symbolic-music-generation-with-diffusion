from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 512
    eval_batch_size = 64  # how many images to sample during evaluation
    sample_batch_size = 16
    voice_num = 4
    num_epochs = 3
    eval_epoch = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    timestep = 1000
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ddpm_1000step"  # the model name locally and on the HF Hub
    data_dir = './dataset/bach_pianoroll_128_4.npy'

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    resolution=4  # default=24
    seed=0  
    deta=0.01

