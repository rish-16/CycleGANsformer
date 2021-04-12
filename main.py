import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cyclegansformer import TGenerator
# from cyclegansformer import Discriminator, Generator, CycleGAN, ImageDatasetLoader

# idl = ImageDatasetLoader("./datasets/horse2zebra/trainA/", "./datasets/horse2zebra/trainB/")

# cg = CycleGAN()
# cg.fit(idl)

# x = torch.randn((1, 3, 256, 256))
gen = TGenerator()
# pred = gen(x)
print (gen)
# print (pred.shape)