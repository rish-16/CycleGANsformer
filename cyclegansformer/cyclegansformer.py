import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import itertools

from cyclegansformer.models import Generator, Discriminator
from cyclegansformer.utils import LRDecay, ReplayBuffer, weights_init

CUTOUT_PROB = 0.3
COLOR_PROB = 1.0
LR = 2e-4
EPOCHS = 200
DECAY_EP = 100

class CycleGANsformer(nn.Module):
    def __init__(self):
        self.gen_X2Y = Generator()
        self.gen_Y2X = Generator()        
        self.disc_X = Discriminator()
        self.disc_Y = Discriminator()

        # custom initialise weights
        self.gen_X2Y.apply(weights_init)
        self.gen_Y2X.apply(weights_init)
        self.disc_X.apply(weights_init)
        self.disc_Y.apply(weights_init)

        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.L1Loss()

        self.optim_gen = Adam(itertools.chain(self.gen_X2Y.parameters(), self.gen_Y2X.parameters()), betas=(0.5, 0.999), lr=LR)
        self.optim_disc_X = Adam(self.disc_X.parameters(), betas=(0.5, 0.999), lr=LR)
        self.optim_disc_Y = Adam(self.disc_Y.parameters(), betas=(0.5, 0.999), lr=LR)

        self.lr_lambda = LRDecay(EPOCHS, 0, DECAY_EP)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optim_gen, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optim_disc_X, lr_lambda=self.lr_lambda)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optim_disc_Y, lr_lambda=self.lr_lambda)

        self.G_opt = Adam(self.gen.parameters, lr=2e-4, betas=(0.5, 0.9))
        self.D_opt = Adam(self.disc.parameters, lr=2e-4, betas=(0.5, 0.9))

        self.g_losses = []
        self.d_losses = []
        self.gan_losses = []
        self.cycle_losses = []

        self.fake_X = ReplayBuffer()
        self.fake_Y = ReplayBuffer()

    def train(self):
        print ("hello")