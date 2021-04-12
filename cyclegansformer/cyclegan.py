import torch, os, sys
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, bias=True, padding=1, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) 
            if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, n_features=64, n_res=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )

        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(n_features, n_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(n_features*2, n_features*4, kernel_size=3, stride=2, padding=1)
            ]
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_features*4) for _ in range(n_res)]
        )

        self.up_block = nn.ModuleList(
            [
                ConvBlock(n_features*4, n_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(n_features*2, n_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(n_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_block:
            x = layer(x)
        x = self.last(x)
        return torch.tanh(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        x = torch.sigmoid(x)
        return self.model(x)

class ImageDatasetLoader(Dataset):
    def __init__(self, root_x, root_y, transform=None):
        super().__init__()
        self.root_x = root_x
        self.root_y = root_y
        self.transform = transform

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

        if self.transform:
            augmentation = self.tranform(img=x_img, image0=x_img)
            x_img = augmentation["image"]
            y_img = augmentation["image0"]

        return x_img, y_img

class CycleGAN:
    def __init__(self):
        super().__init__()
        self.disc_X  = Discriminator(in_channels=3)
        self.disc_Y  = Discriminator(in_channels=3)
        
        self.gen_X2Y = Generator(img_channels=3, n_res=9)
        self.gen_Y2X = Generator(img_channels=3, n_res=9)

        self.opt_disc = Adam(
            list(self.disc_X.parameters() + self.disc_Y.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        self.opt_gen = Adam(
            list(self.gen_X2Y.parameters() + self.gen_X2Y.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        self.L1 = nn.L1Loss
        self.mse = nn.MSELoss()

    def _train_fn(self, discX, discY, genX2Y, genY2X, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
        loop = tqdm(loader, leave=True)

        for idx, (x_img, y_img) in enumerate(loop):
            # train discriminators
            with torch.cuda.amp.autocast():
                fake_x = genY2X(y_img)
                D_X_real = discX(fake_x)
                D_X_fake = discX(fake_x.detach())
                D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
                D_X_fake_loss = mse(D_X_fake, torch.zeros_like(D_X_fake))
                D_X_loss = D_X_real_loss + D_X_fake_loss

                fake_y = genX2Y(x_img)
                D_Y_real = discY(fake_y)
                D_Y_fake = discY(fake_y.detach())            
                D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
                D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
                D_Y_loss = D_Y_real_loss + D_Y_fake_loss

                # adversarial loss
                D_loss = (D_X_loss + D_Y_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            with torch.cuda.amp.autocast():
                # train generators
                D_X_fake = discX(fake_x)
                D_Y_fake = discY(fake_y)
                loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))
                loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))

                # cycle loss
                cycle_Y = genX2Y(fake_x)
                cycle_X = genY2X(fake_y)
                cycle_Y_loss = l1(y_img, cycle_Y)
                cycle_X_loss = l1(x_img, cycle_X)

                # identity loss
                id_Y = genX2Y(y_img)
                id_X = genY2X(x_img)
                id_Y_loss = l1(y_img, id_Y)
                id_X_loss = l1(x_img, id_X)

                # combined loss
                LAMBDA_CYCLE = 10
                LAMBDA_ID = 0
                G_loss = loss_G_Y + loss_G_X + (cycle_Y_loss + cycle_X_loss)*LAMBDA_CYCLE + (id_Y_loss + id_X_loss)*LAMBDA_ID

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

    def fit(self, dataset, epochs=200):
        loader = Dataset(dataset, batch_size=1, shuffle=True, n_workers=4, pin_memory=True)
        G_scaler = torch.cuda.amp.GradScaler()
        D_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            self._train_fn(
                self.disc_X, 
                self.disc_Y, 
                self.gen_X2Y, 
                self.gen_Y2X, 
                loader, 
                self.opt_disc, 
                self.opt_gen, 
                self.L1, 
                self.mse, 
                D_scaler, 
                G_scaler
            )

            
