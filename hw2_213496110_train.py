import time
import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.utils
import torch.distributions
import torch.nn.functional as F
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200


device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.001
prob = 0.3
LATENT_DIM = 200
HIDDEN_SIZE = 16 * 49
epochs = 50


def re_parametrization_trick(mu, log_var):
    # Using re-parameterization trick to sample from a gaussian
    eps = torch.randn_like(log_var)
    return mu + torch.exp(log_var / 2) * eps


class Encoder(nn.Module):
    """
    The encoder class for the VAE object. The Encoder gets an argument kind to differ between the
    latent spaces, because each kind use different re-parameterization trick. All the types of the encoder have 2 blocks
    of conventional neural networks, and max pool regularization layers.
    After the conv layers each kind has different linear layers to get the mean and var of the distribution.
    """
    def __init__(self, kind):
        super(Encoder, self).__init__()
        self.kind = kind
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        if self.kind == 'Continuous':
            self.linear1 = nn.Linear(HIDDEN_SIZE, LATENT_DIM)
            self.linear2 = nn.Linear(HIDDEN_SIZE, LATENT_DIM)
        elif self.kind == 'discrete':
            self.linear = nn.Linear(HIDDEN_SIZE, int(LATENT_DIM))
        elif self.kind == 'both':
            self.linear1 = nn.Linear(HIDDEN_SIZE, int(LATENT_DIM / 2))
            self.linear2 = nn.Linear(HIDDEN_SIZE, int(LATENT_DIM / 2))
            self.linear = nn.Linear(HIDDEN_SIZE, int(LATENT_DIM / 2))
        self.dropout = nn.Dropout(p=prob)
        self.pool1_saver = None
        self.pool2_saver = None

    def forward(self, img):
        img = self.conv1(img)
        img, self.pool1_saver = self.pool(img)
        img = self.conv2(img)
        img, self.pool2_saver = self.pool(img)
        img = img.view(-1, HIDDEN_SIZE)
        img = self.dropout(img)
        if self.kind == 'Continuous':
            mu = self.linear1(img)
            logVar = self.linear2(img)
            return mu, logVar
        elif self.kind == 'discrete':
            img = self.linear(img)
            return img
        else:
            mu = self.linear1(img)
            logVar = self.linear2(img)
            img = self.linear(img)
            return img, mu, logVar


class Decoder(nn.Module):
    """
    Works the opposite to the Encoder, gets a sample from the encoder's distribution and pass it through linear
    layer and then through transpose conv and max unpool to recreate the image.
    Works the same for all the latent spaces.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(LATENT_DIM, HIDDEN_SIZE)
        self.ReLU = nn.ReLU()
        self.transpose_conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.transpose_conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.max_unPool = nn.MaxUnpool2d(2)
        self.softmax = nn.Softmax
        self.pool_saver1 = None
        self.pool_saver2 = None
        self.upSample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, img, encoder=True):
        img = self.ReLU(self.linear(img))
        img = img.view(-1, 32, 7, 7)
        if encoder:
            img = self.max_unPool(img, self.pool_saver2)
        else:
            img = self.upSample(img)
        img = self.transpose_conv1(img)
        if encoder:
            img = self.max_unPool(img, self.pool_saver1)
        else:
            img = self.upSample(img)
        img = self.transpose_conv2(img)
        img = self.ReLU(1 - self.ReLU(1 - img))
        return img


class VariationalAutoEncoder(nn.Module):
    """
    The VAE class, creates the encoder and the decoder and manage the flowing of the information between them according
    to the latent space kind.
    """
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.encoder = Encoder(kind)
        self.decoder = Decoder()

    def forward(self, img):
        if self.kind == 'Continuous':
            mu, logvar = self.encoder(img)
            z = re_parametrization_trick(mu, logvar)
            self.decoder.pool_saver1 = self.encoder.pool1_saver
            self.decoder.pool_saver2 = self.encoder.pool2_saver
            return self.decoder(z), mu, logvar
        elif self.kind == 'discrete':
            img = self.encoder(img)
            self.decoder.pool_saver1 = self.encoder.pool1_saver
            self.decoder.pool_saver2 = self.encoder.pool2_saver
            z_discrete = torch.distributions.OneHotCategorical(logits=img)
            z_sampled = z_discrete.sample()
            out = self.decoder(z_sampled)
            return out, z_discrete, z_sampled
        else:
            img, mu, logVar = self.encoder(img)
            self.decoder.pool_saver1 = self.encoder.pool1_saver
            self.decoder.pool_saver2 = self.encoder.pool2_saver
            y = re_parametrization_trick(mu, logVar)
            z_discrete = torch.distributions.OneHotCategorical(logits=img)
            z_sampled = z_discrete.sample()
            z = torch.cat((y, z_sampled), 1)
            out = self.decoder(z)
            return out, mu, logVar, z


def loss_function_discrete(recon_x, x, qy):
    """
    The loss for the discrete latent space.
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.shape[0]

    log_ratio = torch.log(qy * qy.size(-1) + 1e-20)
    KLD = torch.sum(qy * log_ratio, dim=-1).mean()
    return BCE + KLD


def train(vae, data_loader):
    """
    The training function.
    """
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    start_time = time.time()
    for epoch in range(epochs):
        for i, (x, label) in enumerate(data_loader):
            x = x.to(device)
            optimizer.zero_grad()
            if vae.kind == 'Continuous':
                recon_x, mu, logvar = vae(x)
                loss = loss_fn(recon_x, x, mu, logvar)
            elif vae.kind == 'discrete':
                out, z_discrete, z_sampled = vae(x)
                loss = MSE_loss(out.to(device), x.to(device))
            else:
                out, mu, logVar, z = vae(x)
                loss = loss_fn(out.to(device), x.to(device), mu.to(device), logVar.to(device)) + MSE_loss(
                    out.to(device), x.to(device))
            loss.backward()
            optimizer.step()
        print('Epoch {}: Loss {}'.format(epoch + 1, loss))
    end_time = time.time()
    print('the training took: ', start_time - end_time)
    torch.save(vae.state_dict(), vae.kind + '_weights.pkl')
    return vae


def divergence(mu1, std1, mu2, std2):
    """
    Calculates the KL divergence between the given distributions.
    """
    p = torch.distributions.Normal(mu1, std1)
    q = torch.distributions.Normal(mu2, std2)
    return torch.distributions.kl_divergence(p, q).mean()


def loss_fn(recon_x, x, mu, logvar):
    """
    The loss for the continuous latent space.
    """
    MSE = torch.nn.MSELoss(reduction='sum')
    loss = MSE(recon_x, x)
    std = torch.exp(0.5 * logvar)
    compMeans = torch.full(std.size(), 0.0).to(device)
    compStd = torch.full(std.size(), 1.0).to(device)
    dl = divergence(mu, std, compMeans, compStd)
    return dl + loss


def MSE_loss(output, img):
    MSE = torch.nn.functional.mse_loss
    MSE_loss = MSE(img, output)
    return MSE_loss
