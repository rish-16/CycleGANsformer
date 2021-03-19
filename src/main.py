import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models import Generator, Discriminator

class CycleGAN(nn.Module):
    def __init__(self):
        pass