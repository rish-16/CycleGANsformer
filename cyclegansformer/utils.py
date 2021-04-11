import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import itertools

class LRDecay(nn.Module):
    def __init__(self, n_epochs, offset, decay_epoch):
        super().__init__()
        epoch_flag = n_epochs - decay_epoch
        assert (epoch_flag > 0), "Decay must begin before training ends"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epochs = decay_epoch

    def forward(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_epochs) / (self.n_epochs - self.decay_epochs)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert (max_size > 0), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)