import torch.nn as nn
import torch
import os
from typing import List

class Model:
    name: str
    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, f'{self.name}.bin')):
                path = os.path.join(ckpt_dir, f'{self.name}.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), f'{self.name}.bin')
        except:
            print(f"[{type(self).__name__}] No checkpoint")
            return
        print(f"[{type(self).__name__}] Restore from", path)
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), f'{self.name}.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def get_mlp_generator(arch, device):
    model = MLPGenerator(arch).to(device)
    model.apply(weights_init)
    return model

def get_mlp_discriminator(arch, device):
    model = MLPDiscriminator(arch).to(device)
    model.apply(weights_init)
    return model

def get_generator(num_channels, latent_dim, hidden_dim, device):
    model = Generator(num_channels, latent_dim, hidden_dim).to(device)
    model.apply(weights_init)
    return model

def get_discriminator(num_channels, hidden_dim, device):
    model = Discriminator(num_channels, hidden_dim).to(device)
    model.apply(weights_init)
    return model


class Generator(nn.Module, Model):
    def __init__(self, num_channels, latent_dim, hidden_dim):
        super(Generator, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.name = type(self).__name__.lower()

		# TODO START
        layer1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 4 * self.hidden_dim, 4, 1, 0),
            nn.BatchNorm2d(4 * self.hidden_dim),
            nn.ReLU(),
        )
        layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.hidden_dim, 2 * self.hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(2 * self.hidden_dim),
            nn.ReLU(),
        )
        layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, self.hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(),
        )
        layer4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, 1, 4, 2, 1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
        )
		# TODO END

    def forward(self, z):
        '''
        *   Arguments:
            *   z (torch.FloatTensor): [batch_size, latent_dim, 1, 1]
        '''
        z = z.to(next(self.parameters()).device)
        return self.decoder(z)

class MLPGenerator(nn.Module, Model):
    def __init__(self, arch: List[int]):
        super().__init__()
        modules = []
        for i in range(len(arch) - 1):
            modules.append(nn.Linear(arch[i], arch[i + 1]))
            if i != len(arch) - 2:
                modules.append(nn.BatchNorm1d(arch[i + 1]))
                modules.append(nn.ReLU())
        modules.append(nn.Tanh())
        self.decoder = nn.Sequential(*modules)
        self.name = type(self).__name__.lower()
        self.latent_dim = arch[0]

    def forward(self, z):
        z = z.view(*z.shape[:2])
        return self.decoder(z).view(z.size(0), 1, 32, 32)

class Discriminator(nn.Module, Model):
    def __init__(self, num_channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.clf = nn.Sequential(
            # input is (num_channels) x 32 x 32
            nn.Conv2d(num_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim) x 16 x 16
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*2) x 8 x 8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_dim*4) x 4 x 4
            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.name = type(self).__name__.lower()

    def forward(self, x):
        return self.clf(x).view(-1, 1).squeeze(1)


class MLPDiscriminator(nn.Module, Model):
    def __init__(self, arch: List[int]):
        super().__init__()
        modules = []
        for i in range(len(arch) - 1):
            modules.append(nn.Linear(arch[i], arch[i + 1]))
            if i != len(arch) - 2:
                # modules.append(nn.BatchNorm1d(arch[i + 1]))
                modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*modules)
        self.name = type(self).__name__.lower()

    def forward(self, z):
        z = z.view(z.size(0), -1)
        return self.encoder(z)
