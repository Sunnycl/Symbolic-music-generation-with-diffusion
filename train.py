import torch
import os
from music21 import *
from diffusers import UNet2DModel, DDPMPipeline
import torch.nn.functional as F
from diffusers.utils import make_image_grid
from config.train_config import TrainingConfig
from torch.utils.data import DataLoader
from utils.data_util import MidiDataset, create_logger, savemusic
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    image, image_array = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        num_inference_steps = 1000,
        return_dict=True
    )  # [batch, 128, 128, 1]
    # Make a grid out of the images
    image_grid = make_image_grid(image, rows=4, cols=4)  #将一个批次16的图片排在一起输出显示

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    # Save the music
    savemusic(image_array, test_dir, epoch)

def train(config, model, noise_scheduler, train_dataloader):
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_data_loader) * config.num_epochs),
    )

    print((f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"))
    os.makedirs(config.output_dir, exist_ok=True)

    global_step = 0
    logger = create_logger(os.path.join(config.output_dir, 'logs'))

    # Now you train the model
    for epoch in range(0, config.num_epochs):
        logger.info(f'==========================epoch: {epoch}===============================')
        for batch in train_dataloader:
            clean_images = batch["images"].to(device)
            # 增加维度
            clean_images = clean_images.unsqueeze(1)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            logs = {"loss": loss.detach().item(),  "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "total step": len(train_dataloader)}
            logger.info(logs)

            global_step += 1

        # 保存模型
        os.makedirs(os.path.join(config.output_dir, 'unet'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(config.output_dir, 'unet', f'unet_{epoch}'))

        # 采样
        pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        evaluate(config, epoch, pipeline)

    logger.info('Done')

# 加载配置文件
config = TrainingConfig()

# 加载数据集
dataset = MidiDataset(config.data_dir, compress=False, blend=True)  # [batch, 128, 128]

# 划分数据集
train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_data_loader = DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size=config.eval_batch_size,shuffle=True)

# 实例化模型
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    # block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    block_out_channels=(32, 32, 64, 64, 128, 128),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    dropout=0.2
).to(device)

# 添加随机噪声的调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start = 0.000008, beta_end = 0.01)

if __name__=="__main__":
    train(config, model, noise_scheduler, train_data_loader)
