import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


print("Beginning")
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, channels):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(num_classes + int(img_size**2) * channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        d_in = torch.cat((img_flat, self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


latent_dim = 100
img_size = 28
channels = 1
num_classes = 10
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 50


if __name__ == '__main__':

    print("Data downloading")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_data = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    generator = Generator(latent_dim, num_classes, img_size, channels)
    discriminator = Discriminator(num_classes, img_size, channels)

    adversarial_loss = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)


    generator.train()
    discriminator.train()
    print("Training")
    for epoch in range(n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batch_size = imgs.shape[0]
            real = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()

            z = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)

            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), real)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs, labels), real)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

        if epoch % 10 == 0:
            z = torch.randn(9, latent_dim, device=device)
            labels = torch.arange(9, device=device)
            gen_imgs = generator(z, labels)
            plt.imshow(gen_imgs.cpu().detach().numpy()[0][0], cmap="gray")
            plt.show()


    print("Saving the models")
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), '../MNISTGan/discriminator.pth')


    # Loading the model
    # Load the saved models
    """
    generator = Generator(latent_dim, num_classes, img_size, channels)
    discriminator = Discriminator(num_classes, img_size, channels)
    
    generator.load_state_dict(torch.load('generator.pth'))
    discriminator.load_state_dict(torch.load('discriminator.pth'))
    
    # Move models to device
    generator.to(device)
    discriminator.to(device)
    
    
    import matplotlib.pyplot as plt
    
    def generate_digit(generator, digit, latent_dim):
        generator.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            # Create random noise and generate an image for the specified digit
            noise = torch.randn(1, latent_dim, device=device)
            label = torch.tensor([digit], device=device)
            generated_img = generator(noise, label).cpu()
    
            # Convert tensor to NumPy and plot the image
            img = generated_img.squeeze().numpy()  # Remove batch and channel dimensions
            plt.imshow(img, cmap="gray")
            plt.title(f"Generated Digit: {digit}")
            plt.show()
    
    # Example: generate digit 1
    generate_digit(generator, 1, latent_dim)
    """
