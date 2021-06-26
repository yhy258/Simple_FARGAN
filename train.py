import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import *
from config import Config

"""
    Simple Fashion MNIST Train
"""

dataset = datasets.FashionMNIST(
    root = './.data',
    train = True,
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)]),
    download= True
)

train_dataloader = DataLoader(dataset =dataset, batch_size=Config.N_1, shuffle = True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


generator = Generator(Config.d_model).to(DEVICE)
critic = Critic().to(DEVICE)


gen_optim = torch.optim.Adam(params=generator.parameters(), lr=Config.lr, betas=[Config.beta_1, Config.beta_2])
crit_optim = torch.optim.Adam(params=critic.parameters(), lr=Config.lr, betas=[Config.beta_1, Config.beta_2])

"""
    Original GAN처럼 BCELoss를 사용해도 굉장히 stable하게 학습이 진행된다.
"""
criterion = nn.BCELoss()

for epoch in range(Config.EPOCHS):
    critic_losses = []
    generator_losses = []
    print("Epoch {}/{}".format(epoch + 1, Config.EPOCHS))
    for i, (img, _) in enumerate(tqdm(train_dataloader)):

        img = img.to(DEVICE)

        """
            Close Pair Choice!!!!
        """
        close_latent = torch.randn(Config.N_0 * Config.f, Config.d_model).to(DEVICE)
        close_generated = generator(close_latent).detach()

        output = critic(close_generated)
        sort_idx = torch.argsort(output, dim=0)  # 작은 것부터 앞으로. 오름차순. idx

        sort_idx = sort_idx[:Config.N_0]  # 앞부분 top N_0개 만큼 뽑아오기 (close pair 후보군)

        close_pair = torch.tensor([]).to(DEVICE)

        for idx in sort_idx:
            torch.cat([close_pair, close_generated[idx]], dim=0)

        close_pair = torch.cat([close_pair, img], dim=0)

        close_pair = close_pair.requires_grad_(True)

        """
            GP 적용 시 close pair에 대해서 sigmoid 전 feature map으로 적용해준다!!!
        """
        disc_true = critic(close_pair)
        disc_true_sig = F.sigmoid(disc_true)

        latent = torch.randn(Config.N, Config.d_model).to(DEVICE)
        generated = generator(latent).detach()
        disc_false = critic(generated)
        disc_false = F.sigmoid(disc_false)

        """
            GP 구하기
        """
        gradient = torch.autograd.grad(outputs=disc_true, inputs=close_pair, grad_outputs=torch.ones_like(disc_true),
                                       retain_graph=True, create_graph=True)[0]
        gp = torch.linalg.norm(gradient) ** 2

        """
            Discriminator Loss (FAR Loss 계산)
        """
        FARLOSS = criterion(disc_true_sig, torch.ones_like(disc_true_sig)) + criterion(disc_false, torch.zeros_like(
            disc_false)) + Config.lamb * torch.mean(gp)

        critic_losses.append(FARLOSS.item())

        crit_optim.zero_grad()
        FARLOSS.backward()
        crit_optim.step()

        """
            Generator Train
        """
        img = img.to(DEVICE)

        latent = torch.randn(Config.N, Config.d_model).to(DEVICE)

        generated = generator(latent)
        gen_true = critic(generated)
        gen_true = F.sigmoid(gen_true)

        gen_loss = criterion(gen_true, torch.ones_like(gen_true))
        generator_losses.append(gen_loss.item())

        gen_optim.zero_grad()
        gen_loss.backward()
        gen_optim.step()

    print("Critic Loss : {}\t Generator Loss : {}".format(np.mean(critic_losses), np.mean(generator_losses)))
