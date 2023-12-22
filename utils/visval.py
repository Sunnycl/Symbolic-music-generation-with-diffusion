import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from torch.utils.data import DataLoader
from utils.data_util import MidiDataset


def plot_data(piano_representations, start=0):
    """可视化钢琴卷"""
    fig = plt.figure()
    new_my_pianoroll_rep1 =  np.rot90 (piano_representations[start])
    plt.imshow(new_my_pianoroll_rep1, cmap='gray')
    plt.title(str(start))
    plt.show()

def vistraindata(path):
    """可视化训练集"""
    dataset = MidiDataset(path)
    batch_size = 1

    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True)
    train_x = next(iter(data_loader))
    for i in range (1) :    
        plot_data(train_x['piano_rolls'], i) 
        train_x['piano_rolls'].shape


def visalize_split(train_dataloader):
    for step in range(0, 1):
        batch = next(iter(train_dataloader))
        clean_images = batch["images"][0]
        cat_image = torch.zeros(128, 128)
        for i in clean_images:
            cat_image += i
        cat_image = cat_image.unsqueeze(-1)
        # 增加维度
        clean_images = DDPMPipeline.numpy_to_pil(np.array(clean_images.unsqueeze(-1)))
        cat_image = DDPMPipeline.numpy_to_pil(np.array(cat_image))
        for i, image in enumerate(clean_images):
            image.save(f'clean_image_8_{i}.png')
        for i, image in enumerate(cat_image):
            image.save(f'blend_image_8_{i}.png')

def visalize(train_dataloader):
    for step in range(0, 4):
        batch = next(iter(train_dataloader))
        clean_images = batch["images"][:, step]
        # 增加维度
        clean_images = DDPMPipeline.numpy_to_pil(np.array(clean_images.unsqueeze(-1)))
        for i, image in enumerate(clean_images):
            image.save(f'clean_image_4_{step}_{i}.png')

def visalize_blend(train_dataloader):
    batch = next(iter(train_dataloader))
    clean_images = batch["images"][0]

    # 增加维度
    clean_images = clean_images.unsqueeze(-1)
    # Sample noise to add to the images
    noise = torch.randn(clean_images.shape).to(clean_images.device)
    bs = clean_images.shape[0]

    noise_litter = 0.0001*noise + (1-0.0001)*clean_images
    noise_litter = DDPMPipeline.numpy_to_pil(np.array(noise_litter))
    image = make_image_grid(noise_litter, rows=1, cols=1)

    # Make a grid out of the images
    clean_images = DDPMPipeline.numpy_to_pil(np.array(clean_images))
    image_grid = make_image_grid(clean_images, rows=1, cols=1)  #将一个批次的图片排在一起输出显示

    image_grid.save(f"clean_image.png")
    image.save('litterimage.png')

def show_plt(score, figname=None):
    """对输入的一维列表进行画图"""
    plt.figure()
    plt.style.use('seaborn-v0_8-paper')
    plt.title(figname)
    x = [i for i in range(len(score))]


    plt.plot(x, score, 'b')
    plt.legend(loc='upper right')

    filename1 = f'score_{figname}.png'
    savepath = os.path.join('./', filename1)
    
    plt.savefig(savepath)

    plt.show()

def show_loss_acc(reconstruction_loss, acc, path): 
    epoki = range(len(reconstruction_loss))

    fig,axs=plt.subplots(1, 2, (15, 30))

    axs[0].plot(epoki, reconstruction_loss, 'g', label='reconstruction_loss')
    axs[1].plot(epoki, acc, 'b', label='acc')

    axs[0].set_title('Training Loss')
    axs[1].set_title('Training Acc')
    fig.legend()

    epoch = len(reconstruction_loss)
    filename1 = 'Loss_and_Acc_%04d.png' % (epoch)
    savepath = os.path.join(path, filename1)
    
    fig.savefig(savepath)