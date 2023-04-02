# VAE-and-Cycle-GAN
## VAE:
### Preprocessing:
I worked on the MNISt data set.
I wanted to color every digit in a separate color so, I created a dictionary of digits as keys and sets of RGB colors as values. To color the MNIST dataset I created a custom dataset class and made a transformer to transfer the MNIST to RGB from greyscale and then applied the coloring function.
### Architecture:
I created three models each for every latent space kind (Continous, discrete, and both) Every model has an encoder and a decoder. Before starting the implantation I noticed that the VAE are different only in their loss function and in their re parameterization, so instead of creating 3 different classes for each model’s encoder, and decoder, I decided to create one Encoder and one Decoder class that will get the kind of the latent space and will build the model structure accordingly.
For the Encoder, each model got two CNN layers. Then, the model got linear layers. Every model got fitting linear layers according to its kind. For both discrete and continuous were 3 layers,  for continuous were 2, and 1 for discrete. To prevent overfitting I used the Pool function twice after the CNN layers and dropout for the linear layers.
The decoder structure is the same for all kinds. One linear layer passes the output to 2 transpose CNN layers with unpool layer between every transpose CNN.
The Vae class gets the kind and builds the matching encoder and decoder accordingly. For data flow, the Vae passes the encoder the photos get the output
from it, re parameterize and pass it to the decoder.

## GAN:
### Preprocessing:
I colored every image in a random color.
### Articheture:
To train the model, I first trained the discriminators to distinguish between real and fake photos. The discriminator got real and fake images. Then, calculated the loss to decide if the image was real or fake(1 or 0). For the generator, the loss function was more complex and contain 3 different losses. First was the adversarial loss, like the one in the discriminator. the purpose for it was that this loss was not only relevant to the discriminator output but also to the generator. Second, is cycle loss. By giving a fake image to the second generator to recreate the original photo and it calculated the loss with L1 loss. Third, identity loss with the 
generator output and its input image.
