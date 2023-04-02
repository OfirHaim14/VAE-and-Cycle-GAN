import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets
import random
torch.manual_seed(0)

d1_weights = 'disc1_weights.pkl'
d2_weights = 'disc2_weights.pkl'
g1_weights = 'gen1_weights.pkl'
g2_weights = 'gen2_weights.pkl'

# Will be used to randomly colored every digit.
transform1 = [transforms.Lambda(lambda x: torch.cat((x, torch.zeros_like(x), torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), 0)),
              transforms.Lambda(lambda x: torch.cat((torch.zeros_like(x), torch.zeros_like(x), x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, x, 0*x), 0)),
              transforms.Lambda(lambda x: torch.cat((0*x, x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0*x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.8*x, 0.2*x, 0*x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.85*x, 0.2*x, 0.35*x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.64*x, 0.4*x, 0.66*x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.17*x, 0.5*x, 0.55*x), 0))]

transform2 = [transforms.Lambda(lambda x: torch.cat((0 * x, 0 * x, x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.5 * x, 0 * x, 1 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0.4 * x, 0.7 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0.6 * x, 0.3 * x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0 * x, 0.6 * x, 0.3 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0 * x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, 0.5 * x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((x, x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0 * x, x, 0 * x), 0)),
              transforms.Lambda(lambda x: torch.cat((0 * x, x, x), 0))]

lr = 0.0001
epochs = 100
gen_loss_weights = torch.tensor([1.0, 1.0, 3.0, 3.0, 1.0, 1.0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    """
    The Discriminator class.
    """
    def __init__(self):
        super().__init__()
        self.lay1 = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(0.2), nn.AvgPool2d(2))
        self.lay2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 2), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
                                  nn.AvgPool2d(2))
        self.lay3 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
                                  nn.AvgPool2d(2))
        self.lay4 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
                                  nn.AvgPool2d(2))
        self.lay5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1), nn.InstanceNorm2d(1024), nn.LeakyReLU(0.2),
                                  nn.AvgPool2d(2))
        self.lin_lay = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        x = self.lay5(x)
        x = x.reshape(-1, 1024)
        return torch.sigmoid(self.lin_lay(x))


class up_down_blocks(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs) if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class double_block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            up_down_blocks(channels, channels, kernel_size=3, padding=1),
            up_down_blocks(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList([
                up_down_blocks(64, 64 * 2, kernel_size=3, stride=2, padding=1),
                up_down_blocks(64 * 2, 64 * 4, kernel_size=3, stride=2)
        ])
        self.res_blocks = nn.Sequential(
            *[double_block(64 * 4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList([
            up_down_blocks(64 * 4, 128, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            up_down_blocks(128, 64, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            up_down_blocks(64, 32, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            up_down_blocks(32, 16, down=False, kernel_size=3, stride=2, padding=1, output_padding=0),
            ])

        self.last = nn.Conv2d(16, 3, kernel_size=6, stride=1, padding=0)

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))


def get_data_loaders():
    data_list1, data_list2 = [], []
    for i in range(10):
        data1 = torchvision.datasets.MNIST(root='./data2', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transform1[i], transforms.Normalize((0.5,), (0.5,))]), download=True)
        data2 = torchvision.datasets.MNIST(root='./data2', train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transform2[i], transforms.Normalize((0.5,), (0.5,))]), download=True)
        indices1 = [j for j, target in enumerate(data1.targets) if target in [i]]
        indices2 = [j for j, target in enumerate(data2.targets) if target in [i]]
        data_list1.append(torch.utils.data.Subset(data1, indices1))
        data_list2.append(torch.utils.data.Subset(data2, indices2))
    train_data = torch.utils.data.ConcatDataset(data_list1)
    train_data2 = torch.utils.data.ConcatDataset(data_list2)
    loader1 = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=128)
    loader2 = torch.utils.data.DataLoader(train_data2, shuffle=True, batch_size=128)
    return loader1, loader2


def train_eval_GAN(loader1, loader2):
    # creates the generators and discriminators
    disc1 = Discriminator().to(device)
    disc2 = Discriminator().to(device)
    gen1 = Generator().to(device)
    gen2 = Generator().to(device)
    opt_disc = torch.optim.Adam(list(disc1.parameters()) + list(disc2.parameters()), lr=lr)
    opt_gen = torch.optim.Adam(list(gen1.parameters()) + list(gen2.parameters()), lr=lr)
    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    for epoch in range(epochs):
        dataset = tqdm(zip(loader1, loader2), leave=True)
        for idx, data in enumerate(dataset):
            img1, _, img2, _ = data[0][0].to(device), data[0][1], data[1][0].to(device), data[1][1]
            with torch.cuda.amp.autocast():
                fake1 = gen1(img2)
                fake2 = gen2(img1)
                disc_real1 = disc1(img1)
                disc_fake1 = disc1(fake1.detach())
                disc_real2 = disc2(img2)
                disc_fake2 = disc2(fake2.detach())
                real_loss1 = MSE(disc_real1, torch.ones_like(disc_real1))
                real_loss2 = MSE(disc_real2, torch.ones_like(disc_real2))
                fake_loss1 = MSE(disc_fake1, torch.zeros_like(disc_fake1))
                fake_loss2 = MSE(disc_fake2, torch.zeros_like(disc_fake2))
                loss1 = real_loss1 + fake_loss1
                loss2 = real_loss2 + fake_loss2
                disc_loss = (loss1 + loss2) / 2
            opt_disc.zero_grad()

            with torch.cuda.amp.autocast():
                d_fake1 = disc1(fake1)
                d_fake2 = disc2(fake2)
                g_fake_loss1 = MSE(d_fake1, torch.ones_like(d_fake1))
                g_fake_loss2 = MSE(d_fake2, torch.ones_like(d_fake2))
                recon_x1 = gen1(fake2)
                recon_x2 = gen2(fake1)
                cycle_loss1 = L1(img1, recon_x1)
                cycle_loss2 = L1(img2, recon_x2)
                identity1 = gen1(img1)
                identity2 = gen2(img2)
                identity_loss1 = L1(img1, identity1)
                identity_loss2 = L1(img2, identity2)
                gen_loss_list = [g_fake_loss1, g_fake_loss2, cycle_loss1, cycle_loss2,
                                 identity_loss1, identity_loss2]
                gen_loss = sum([gen_loss_list[i] * gen_loss_weights[i] for i in range(len(gen_loss_list))])
            opt_gen.zero_grad()
            if idx % 200 == 0:
                save_image(fake1 * 0.5 + 0.5, f"saved_images/fake1_{idx}.png", nrow=12)
                save_image(img1 * 0.5 + 0.5, f"saved_images/real1_{idx}.png", nrow=12)
                save_image(fake2 * 0.5 + 0.5, f"saved_images/fake2_{idx}.png", nrow=12)
                save_image(img2 * 0.5 + 0.5, f"saved_images/real2_{idx}.png", nrow=12)
        print('Epoch {}: Discriminator Loss {}, Generator Loss {}'.format(epoch + 1, disc_loss, gen_loss))
    torch.save(disc1.state_dict(), d1_weights)
    torch.save(disc2.state_dict(), d2_weights)
    torch.save(gen1.state_dict(), g1_weights)
    torch.save(gen2.state_dict(), g2_weights)
    save_GAN_images(gen1, gen2, loader1, loader2)


def save_GAN_images(gen1, gen2, loader1, loader2):
    data = random.sample(list(loader1), 1)
    images1, _ = data[0]
    fake_gen2 = gen2(images1.to(device))
    data = random.sample(list(loader2), 1)
    images2, _ = data[0]
    fake_gen1 = gen1(images2.to(device))
    save_image(images1 * 0.5 + 0.5, f"images1.png", nrow=12)
    save_image(images2 * 0.5 + 0.5, f"images2.png", nrow=12)
    save_image(fake_gen1 * 0.5 + 0.5, f"fakeImages1.png", nrow=12)
    save_image(fake_gen2 * 0.5 + 0.5, f"fakeImages2.png", nrow=12)
