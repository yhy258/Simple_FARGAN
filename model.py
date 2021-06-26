import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.module_list = nn.Sequential(
            *self.block(latent_dim, 256, bn=False),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, 28 * 28)
        )
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.module_list(x)
        x = x.view(-1, 1, 28, 28)
        x = self.act(x)
        return x

    def block(self, in_channel, out_channel, bn=True):
        layers = []
        layers.append(nn.Linear(in_channel, out_channel))
        if bn:
            layers.append(nn.BatchNorm1d(out_channel))
        layers.append(nn.LeakyReLU())

        return layers


class Critic(nn.Module):
    def __init__(self, ):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1)
        return self.model(x)

