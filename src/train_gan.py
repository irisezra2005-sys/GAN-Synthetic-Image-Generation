import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from gan_models import Generator, Discriminator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
epochs = 30
latent_dim = 100

# Create output folders
os.makedirs("output/generated_samples", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Models
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)

print("Training Started...\n")

for epoch in range(epochs):

    for i, (real_images, _) in enumerate(train_loader):

        batch_size_current = real_images.size(0)

        real_images = real_images.to(device)

        real_labels = torch.ones(batch_size_current, 1).to(device)
        fake_labels = torch.zeros(batch_size_current, 1).to(device)

        # Train Discriminator
      
        optimizer_D.zero_grad()

        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)

        noise = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(noise)

        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()

        optimizer_D.step()

        # Train Generator

        optimizer_G.zero_grad()

        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)

        g_loss.backward()

        optimizer_G.step()

        if (i + 1) % 200 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{i+1}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )

    # Save generated images every epoch
    save_image(
        fake_images[:25],
        f"output/generated_samples/epoch_{epoch+1}.png",
        nrow=5,
        normalize=True
    )

# Save trained generator
torch.save(generator.state_dict(), "output/generator.pth")

print("\nTraining Completed Successfully!")
print("Generated images saved in output/generated_samples/")
print("Generator model saved as output/generator.pth")
