import numpy as np
import torch, os
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import Dataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
Inspired by and credit to Phil Wang's
implementation of the Google's Vision Transformer.

https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
'''

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)

"""
Adapted from the Vision Transformer implementation by Phil Wang 
(https://github.com/lucidrains/vit-pytorch/)
"""
class MHAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5 # "scaled" dot product

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # scaled dot product

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, MHAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return x

'''
Generator architecture inspired by TransGAN
from Jiang Y. et al., 2021
'''
class TGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(1024, 8*8*1024)
        self.stage1 = nn.ModuleList([
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024)
        ])

        self.ps1 = nn.PixelShuffle(2)

        self.stage2 = nn.ModuleList([
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256)
        ])

        self.ps2 = nn.PixelShuffle(4)

        self.stage3 = nn.ModuleList([
            Encoder(16*16*256, 6, 8, 32*32*64, 32*32*64),
            Encoder(16*16*256, 6, 8, 32*32*64, 32*32*64)
        ])

        self.out = nn.Linear(32*32*64, 32*32*3)

    def forward(self, noise):
        x = self.input_layer(noise)
        x = self.stage1(x)
        x = self.ps1(x)
        x = self.stage2(x)
        x = self.ps2(x)
        x = self.stage3(x)
        out = self.out(x)

        return out

'''
Discriminator architecture inspired by TransGAN
from Jiang Y. et al., 2021
'''
class TDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(32*32*3, (8*8+1)*384)
        self.stage1 = nn.ModuleList([
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384),
            Encoder((8*8+1)*384, 6, 8, (8*8+1)*384, (8*8+1)*384)
        ])
        self.fc1 = nn.Linear((8*8+1)*384, 1*384)
        self.out = nn.Linear(1*384, 1) # classification head

    def forward(self, img):
        x = self.input_layer(img)
        x = self.stage1(x)
        x = self.fc1(x)
        pred = self.out(1)

        return pred

class ImageDatasetLoader(Dataset):
    def __init__(self, root_x, root_y):
        super().__init__()
        self.root_x = root_x
        self.root_y = root_y
        self.transform = A.Compose(
            [
                A.Resize(width=256, height=256),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

        self.x_imgs = os.listdir(root_x)
        self.y_imgs = os.listdir(root_y)
        self.ds_len = max(len(self.x_imgs), len(self.y_imgs))
        self.x_len = len(self.x_imgs)
        self.y_len = len(self.y_imgs)

    def __len__(self):
        return self.ds_len
    
    def __getitem__(self, index):
        x_img = self.x_imgs[index % self.x_len]
        y_img = self.y_imgs[index % self.y_len]

        x_path = os.path.join(self.root_x, x_img)
        y_path = os.path.join(self.root_y, y_img)

        x_img = np.array(Image.open(x_path).convert("RGB"))
        y_img = np.array(Image.open(y_path).convert("RGB"))

        augmentations = self.transform(image=y_img, image0=x_img)
        x_img = augmentations["image"]
        y_img = augmentations["image0"]

        return x_img, y_img