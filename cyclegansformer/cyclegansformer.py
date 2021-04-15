import torch, os, sys
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from cyclegansformer.utils import TGenerator, TDiscriminator

LR = 2e-4
EPOCHS = 200
DECAY_EP = 100

class CycleGANsformer(nn.Module):
    def __init__(self):
        self.discX = TDiscriminator()
        self.discY = TDiscriminator()

        self.genX2Y = TGenerator()
        self.genY2X = TGenerator()

        self.opt_disc = Adam(
            list(self.discX.parameters()) + list(self.discY.parameters()),
            lr=LR,
            betas=(0.5, 0.999)
        )

        self.opt_gen = Adam(
            list(self.genX2Y.parameters()) + list(self.genY2X.parameters()),
            lr=LR,
            betas=(0.5, 0.999)
        )

        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def _train_fn(self, loader, gscaler, dscaler):
        loop = tqdm(loader, leave=True)

        for idx, (x_img, y_img) in enumerate(loop):
            with torch.cuda.amp.autocast():
                fake_x = self.genY2X(y_img)
                D_X_real = self.discX(x_img)
                D_X_fake = self.discX(fake_x.detach())
                D_X_real_loss = self.mse(D_X_real, torch.ones_like(D_X_real))
                D_X_fake_loss = self.mse(D_X_fake, torch.zeros_like(D_X_fake))
                D_X_loss = D_X_real_loss + D_X_fake_loss

                fake_y = self.genX2Y(x_img)
                D_Y_real = self.discY(fake_y)
                D_Y_fake = self.discYiscY(fake_y.detach())            
                D_Y_real_loss = self.mse(D_Y_real, torch.ones_like(D_Y_real))
                D_Y_fake_loss = self.mse(D_Y_fake, torch.zeros_like(D_Y_fake))
                D_Y_loss = D_Y_real_loss + D_Y_fake_loss

                # adversarial loss
                D_loss = (D_X_loss + D_Y_loss) / 2

            opt_disc.zero_grad()
            dscaler.scale(D_loss).backward(retain_graph=True)
            dscaler.step(opt_disc)
            dscaler.update()

            with torch.cuda.amp.autocast():
                # train generators
                D_X_fake = discX(fake_x)
                D_Y_fake = discY(fake_y)
                loss_G_X = self.mse(D_X_fake, torch.ones_like(D_X_fake))
                loss_G_Y = self.mse(D_Y_fake, torch.ones_like(D_Y_fake))

                # cycle loss
                cycle_Y = self.genX2Y(fake_x)
                cycle_X = self.genY2X(fake_y)
                cycle_Y_loss = self.L1(y_img, cycle_Y)
                cycle_X_loss = self.L1(x_img, cycle_X)

                # identity loss
                id_Y = self.genX2Y(y_img)
                id_X = self.genY2X(x_img)
                id_Y_loss = self.L1(y_img, id_Y)
                id_X_loss = self.L1(x_img, id_X)

                # combined loss
                LAMBDA_CYCLE = 10
                LAMBDA_ID = 0
                G_loss = loss_G_Y + loss_G_X + (cycle_Y_loss + cycle_X_loss)*LAMBDA_CYCLE + (id_Y_loss + id_X_loss)*LAMBDA_ID

            opt_gen.zero_grad()
            gscaler.scale(G_loss).backward(retain_graph=True)
            gscaler.step(opt_gen)
            gscaler.update()

    def fit(self, dataset, epochs=200):
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        G_scaler = torch.cuda.amp.GradScaler()
        D_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            self._train_fn(loader, G_scaler, D_scaler)

    def forward(self, x):
        return self.genX2Y(x) # get generated image