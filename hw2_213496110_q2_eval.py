import random
from torchvision.utils import save_image
import torch
from hw2_213496110_q2_train import get_data_loaders, train_eval_GAN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


trainLoader1, trainLoader2 = get_data_loaders()
train_eval_GAN(trainLoader1, trainLoader2)