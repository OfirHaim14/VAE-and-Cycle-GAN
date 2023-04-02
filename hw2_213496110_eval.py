import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import random as rnd
import time
from hw2_213496110_train import VariationalAutoEncoder, train, re_parametrization_trick
from torchvision.utils import make_grid
from skimage.metrics import structural_similarity as ssim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dictionary of rgb colors as values for every key digit
color_dict = {
    0: (0, 1, 1),
    1: (0, 0.5, 0.3),
    2: (1, 1, 0),
    3: (0, 1, 0),
    4: (1, 0, 0),
    5: (0.6, 0.3, 0),
    6: (0.5, 0, 1),
    7: (1, 0.4, 0.7),
    8: (0, 0, 1),
    9: (1, 0.5, 0),
}

# again but the colors are numpy arrays.
color_dict_np = {
    0: np.array([255, 0, 0]),
    1: np.array([255, 178, 0]),
    2: np.array([255, 255, 0]),
    3: np.array([0, 255, 0]),
    4: np.array([0, 255, 255]),
    5: np.array([0, 0, 255]),
    6: np.array([178, 0, 255]),
    7: np.array([255, 102, 178]),
    8: np.array([153, 76, 0]),
    9: np.array([0, 153, 67]),
}


class ColoredMNIST(Dataset):
    """
    custom dataset for the data set loader in order to transform the photos to rgb
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index], self.labels[index]
        return image, label


def color_white_pixels(img, label, is_train=True):
    """
    Color every digit in the fitting color.
    """
    color_index = rnd.randint(low=1, high=9)
    if is_train:
        new_color = color_dict_np[color_index]
    else:
        new_color = color_dict_np[label]
    new_img = torch.zeros(img.shape)
    for i in range(28):
        for j in range(28):
            for t in range(3):
                new_img[i][j][t] = new_color[t] * (img[i][j][t]/255) / 255
    return torch.permute(new_img, (2, 0, 1))


def eval_vae(vae, test_loader):
   """
    to evaluate the accuracy of the models we will give an input photo and check how much it
    similar to the network output.
    :param vae: the model
    :param test_loader: test images and labels loader
    :return:
   """
   mu_list, log_var_list = [], []
   label_list = []
   latent_list = []
   acc_list = []
   with torch.no_grad():
      for idx, (imgs, label) in enumerate(test_loader):
         imgs = imgs.to(device)
         if vae.kind == 'Continuous':
            out, mu, logVAR = vae(imgs)
            reParam = re_parametrization_trick(mu, logVAR).cpu().detach().numpy().tolist()
            mu = mu.cpu().detach().numpy().tolist()
            logVAR = logVAR.cpu().detach().numpy().tolist()
            mu_list.append(mu[0])
            log_var_list.append(logVAR[0])
            latent_list.append(reParam[0])
            label_list.append(label)
         elif vae.kind == 'discrete':
            out, z_discrete, z_sampled = vae(imgs)
            z_sampled = z_sampled.cpu().detach().numpy().tolist()
            latent_list.append(z_sampled[0])
         else:
            out, mu, logVAR, z = vae(imgs)
            z = z.cpu().detach().numpy().tolist()
            latent_list.append(z[0])
         acc = ssim_accuracy(imgs.cpu(), out.cpu()).cpu().detach().numpy()
         acc_list.append(acc)
      if vae.kind == 'Continuous':
         plot_latent_space(mu_list, label_list, 'ContLatentSpaceMu.png')
         plot_mu_var(mu_list, log_var_list, label_list, 'photos/ContLatentSpaceMuVar.png')
      if vae.kind != 'discrete':
         plot_latent_space(latent_list, label_list, vae.kind + 'LatentSpace.png')
      print(", SSIM Score: " + str(np.mean(np.array(acc_list))))


def plot_latent_space(latent_list, label_list, path):
   """
   plots the latent space the vae got
   """
   colors = []
   for k in label_list:
      colors.append(color_dict[k.item()])
   images = []
   labels = []
   for element in latent_list:
       images.append(element[1])
       labels.append(element[2])
   plt.scatter(images, labels, c=colors, s=15)
   plt.xlabel('mu')
   plt.ylabel('log variance')
   plt.title('Latent Space')
   plt.savefig(path)
   plt.clf()


def plot_mu_var(mu_list, var_list, label_list, path):
   """
   plots the mu and var
   """
   images = []
   labels = []
   colors = []
   for k in label_list:
      colors.append(color_dict[k.item()])
   for i in range(len(mu_list)):
      images.append(mu_list[i][1])
      labels.append(var_list[i][1])
   plt.scatter(images, labels, c=colors, s=15)
   plt.xlabel('mu')
   plt.ylabel('log variance')
   plt.title('Latent Space path')
   plt.savefig(path)
   plt.clf()


def plot_output(vae, test_loader, path):
   """
   plots the output of the vae.
   """
   photos = []
   for j in range(15):
      two_samples = random.sample(list(test_loader), 2)
      img1 = two_samples[0][0].to(device)
      img2 = two_samples[1][0].to(device)
      if vae.kind == 'Continuous':
         reckon_x1, mu1, log_var1 = vae(img1)
         reckon_x2, mu2, log_var2 = vae(img2)
         mu_minus = mu2 - mu1
         var_minus = log_var1 - log_var2
      elif vae.kind == 'discrete':
         reckon_x1, z_discrete1, z_sample1 = vae(img1)
         reckon_x2, z_discrete2, z_sample2 = vae(img2)
         z_minus = z_sample2 - z_sample1
      else:
         reckon_x1, mu1, log_var1, z1 = vae(img1)
         reckon_x2, mu2, log_var2, z2 = vae(img2)
         z_minus = z2 - z1
      for i in range(15):
         if vae.kind == 'Continuous':
            mu = mu1 + torch.mul(torch.div(mu_minus, 15), i)
            var = log_var1 + torch.mul(torch.div(var_minus, 15), i)
            reckon_x = re_parametrization_trick(mu, var).to(device)
         elif vae.kind == 'discrete':
            reckon_x = z_sample1 + torch.mul(torch.div(z_minus, 15), i).to(device)
         elif vae.kind == 'both':
            reckon_x = z1 + torch.mul(torch.div(z_minus, 15), i).to(device)
         x_hat = vae.decoder(reckon_x, encoder=False)
         out_img = torch.permute(x_hat.squeeze(), (1, 2, 0))
         photos.append(out_img)
   images = torch.stack(photos)
   images = images.permute(0, 3, 1, 2)
   plt.imshow(make_grid(images, nrow=15).permute(1, 2, 0).cpu().detach().numpy())
   plt.savefig(path)
   plt.clf()


def ssim_accuracy(real_images, generated_images):
   return torch.tensor([ssim(real_img.permute(1, 2, 0).numpy(), generated_img.permute(1, 2, 0).numpy(),
                       multichannel=True) for real_img, generated_img in zip(real_images, generated_images)]).mean()


if __name__ == "__main__":
    """
    creats the custom MNIST, train and test the model.
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Lambda(lambda x: torch.tensor(np.array(x))),
    ])

    start_time = time.time()

    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train, y_train = zip(*mnist_train)
    x_test, y_test = zip(*mnist_test)
    colored_mnist_train = [color_white_pixels(x_train[i], y_train[i]) for i in range(100)]
    colored_mnist_test = [color_white_pixels(x_test[i], y_test[i], is_train=False) for i in range(50)]
    colored_mnist_train_dataset = ColoredMNIST(colored_mnist_train, y_train[:100])
    colored_mnist_test_dataset = ColoredMNIST(colored_mnist_test, y_test[:50])
    end_of_coloring_time = time.time()
    print('time it took to color is: ', start_time - end_of_coloring_time)
    colored_mnist_train_loader = DataLoader(colored_mnist_train_dataset, batch_size=64, shuffle=True)
    colored_mnist_test_loader = DataLoader(colored_mnist_test_dataset, batch_size=1, shuffle=False)
    vae1 = VariationalAutoEncoder('Continuous').to(device)
    vae1 = train(vae1, colored_mnist_train_loader)
    eval_vae(vae1, colored_mnist_test_loader)
    plot_output(vae1, colored_mnist_test_loader, 'interpolation_Continuous.jpg')

    vae2 = VariationalAutoEncoder('discrete').to(device)
    vae2 = train(vae2, colored_mnist_train_loader)
    eval_vae(vae2, colored_mnist_test_loader)
    plot_output(vae2, colored_mnist_test_loader, 'interpolation_discrete.jpg')

    vae3 = VariationalAutoEncoder('both').to(device)
    vae3 = train(vae3, colored_mnist_train_loader)
    eval_vae(vae3, colored_mnist_test_loader)
    plot_output(vae3, colored_mnist_test_loader, 'interpolation_both.jpg')
