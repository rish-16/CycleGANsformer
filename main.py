import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from cyclegansformer import CycleGANsformer 
from cyclegansformer import Discriminator, Generator, CycleGAN, ImageDatasetLoader

idl = ImageDatasetLoader("./dataset/train/HORSES", "./dataset/train/ZEBRAS")

cg = CycleGAN()
cg.fit(idl)