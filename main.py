import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from cyclegansformer import CycleGANsformer 
from cyclegansformer import Discriminator, Generator

# cgf = CycleGANsformer()
# cgf.train()

# x = torch.randn((5, 3, 256, 256))
# model = Discriminator(in_channels=3)
# pred = model(x)
# print (model)
# print (pred.shape)

img_ch = 3
img_size = 256
x = torch.randn((2, img_ch, img_size, img_size))
gen = Generator(img_ch, n_res=9)
print (gen)
print (gen(x).shape)