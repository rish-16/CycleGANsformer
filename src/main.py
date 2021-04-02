import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import Generator, Discriminator

CUTOUT_PROB = 0.3
COLOR_PROB = 1.0

class CycleGAN(nn.Module):
    def __init__(self):
        self.gen = Generator()
        self.disc = Discriminator()

    def forward(self, noise):
        img = self.gen(noise)
        pred = self.disc(img)