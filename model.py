"""
new mean = sqrt(at) * xo
standard deviation = sqrt(1 - alpha t) * random noise
"""

import torch
import urllib
import PIL
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch.nn as nn
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Config:
    img_shape: Tuple[int, int]
    start_schedule: float
    end_schedule: float
    timestep: int
    
class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.width = config.img_shape[0]
        self.height = config.img_shape[1]
        self.start_schedule = config.start_schedule
        self.end_schedule = config.end_schedule
        self.timestep = config.timestep

        self.betas = torch.linspace(self.start_schedule, self.end_schedule, self.timestep)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, axis=0)
    
    def forward(self, x_0, t, device):
        """
        x_0: (B, C, H, W)
        t: (B, )
        """
        noise = torch.rand_like(x_0)
        sqrt_alpha_cumprod_t = self.get_index_from_list(self.alpha_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_miuns_alpha_prod = self.get_index_from_list(torch.sqrt(1. - self.alpha_cumprod), t, x_0.shape)

        mean = sqrt_alpha_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_miuns_alpha_prod.to(device) * noise.to(device)
        
        return mean + variance, noise.to(device)

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick values according from vals
        """
        result = values.gather(-1, t.cpu())
        """
        if shape of x -> (5, 3, 64, 64)
                -> len(x_shape) = 4
                -> len(x_shape) - 1 = 3
        """
        return result.view(batch_size, *([1] * (len(x_shape) ))).to(t.device)

def get_sample_image()-> PIL.Image.Image:
    url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZmJy3aSZ1Ix573d2MlJXQowLCLQyIUsPdniOJ7rBsgG4XJb04g9ZFA9MhxYvckeKkVmo&usqp=CAU'
    filename = 'racoon.jpg'
    urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)


config = Config(
    img_shape=(32, 32),
    start_schedule=0.0001,
    end_schedule=0.02,
    timestep=300
)

model = DiffusionModel(config).to(device)


def forward_diffusion(x0, t, betas = torch.linspace(0.0, 1.0, 5)):
    noise = torch.rand_like(x0)
    
    alphas = 1 - betas
    alpha_hat = torch.cumprod(alphas, axis=-1)
    result = alpha_hat.gather(-1, t)
    result = result.reshape(-1, 1, 1, 1)
    # print(alpha_hat.sqrt().size())
    # print(x0.size())
    mean = result.sqrt() * x0
    variance = torch.sqrt(1-result) * noise
    x_t = mean + variance
    return x_t, noise

betas = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25]) # betas = number noise added to ever diffusion step 

url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZmJy3aSZ1Ix573d2MlJXQowLCLQyIUsPdniOJ7rBsgG4XJb04g9ZFA9MhxYvckeKkVmo&usqp=CAU'
filename = "racoon.png"
urllib.request.urlretrieve(url, filename)

image = Image.open(filename)
image

transform = transforms.Compose([ #PIL -> torch
    transforms.Resize(config.img_shape[0]),
    transforms.ToTensor(), # 0 to 1
    #scale from -1 to 1, layer norm
    transforms.Lambda(lambda t: (t*2) - 1)
])

reverse_transform = transforms.Compose([ #torch -> PIL
    transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # B,C,H,W->B,H,W,C
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage()
])


pil_image = get_sample_image()
torch_image = transform(pil_image)

NO_DISPLAY_IMAGES = 5
torch_image_batch = torch.stack([torch_image] * NO_DISPLAY_IMAGES)
t = torch.linspace(0, model.timestep - 1, NO_DISPLAY_IMAGES).long()
noisy_image_batch, _ = model.forward(torch_image_batch, t, device)

plt.figure(figsize=(15,15))
f, ax = plt.subplots(1, NO_DISPLAY_IMAGES, figsize = (100,100))

for idx, image in enumerate(noisy_image_batch):
    ax[idx].imshow(reverse_transform(image))
    ax[idx].set_title(f"Iteration: {t[idx].item()}", fontsize = 100)
plt.show()


t = torch.tensor([0, 1, 2, 3, 4])
batch_images = torch.stack([torch_image] * 5)
noisy_image, _ = forward_diffusion(batch_images, t)
# plt.imshow(reverse_transform(noisy_image))