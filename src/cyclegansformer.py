import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from models import Generator, Discriminator

CUTOUT_PROB = 0.3
COLOR_PROB = 1.0

class CycleGAN(nn.Module):
    def __init__(self):
        self.gen = Generator()
        self.disc = Discriminator()

        self.G_opt = Adam(self.gen.parameters, lr=2e-4, betas=(0.5, 0.9))
        self.D_opt = Adam(self.disc.parameters, lr=2e-4, betas=(0.5, 0.9))

    def forward(self, noise):
        img = self.gen(noise)
        pred = self.disc(img)