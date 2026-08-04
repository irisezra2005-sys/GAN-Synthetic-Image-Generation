# GAN-Synthetic-Image-Generation

# Generative Adversarial Network (GAN) for Fashion-MNIST Image Generation

## Project Overview

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate synthetic images based on the Fashion-MNIST dataset. A GAN consists of two neural networks a **Generator** and a **Discriminator** that compete with each other during training. The Generator learns to create realistic images from random noise, while the Discriminator learns to distinguish between real and generated images.

The objective of the project is to train the Generator until it produces synthetic images that closely resemble real Fashion-MNIST samples.

# Dataset

- **Dataset:** Fashion-MNIST
- **Training Images:** 60,000
- **Testing Images:** 10,000
- **Image Size:** 28 × 28 pixels
- **Channels:** 1 (Grayscale)
- **Number of Classes:** 10

# GAN Architecture

## Generator Network

The Generator transforms a 100-dimensional random noise vector into a synthetic 28×28 grayscale image.

### Architecture

Input (100-dimensional latent vector)

↓

Linear (100 → 256)

↓

ReLU

↓

Linear (256 → 512)

↓

ReLU

↓

Linear (512 → 1024)

↓

ReLU

↓

Linear (1024 → 784)

↓

Tanh

↓

Reshape → (1 × 28 × 28)

### Generator Activation Functions

- ReLU
- Tanh (Output Layer)


## Discriminator Network

The Discriminator classifies an image as either **Real** or **Fake**.

### Architecture

Input Image (1 × 28 × 28)

↓

Flatten

↓

Linear (784 → 1024)

↓

LeakyReLU (0.2)

↓

Linear (1024 → 512)

↓

LeakyReLU (0.2)

↓

Linear (512 → 256)

↓

LeakyReLU (0.2)

↓

Linear (256 → 1)

↓

Sigmoid

### Discriminator Activation Functions

- LeakyReLU
- Sigmoid


# Training Process

The GAN is trained using an adversarial learning approach.

## Step 1

Load batches of real Fashion-MNIST images.

## Step 2

Generate random latent vectors.

## Step 3

Generate fake images using the Generator.

## Step 4

Train the Discriminator using both real and fake images.

## Step 5

Train the Generator to fool the Discriminator.

## Step 6

Repeat the above process for multiple epochs until the Generator produces realistic images.

## Step 7

Save generated image samples after every epoch.


# Loss Function

Binary Cross Entropy Loss (BCELoss)

The Discriminator minimizes the classification error between real and fake images, while the Generator minimizes the Discriminator's ability to correctly identify generated images.


# Optimizer

Adam Optimizer

Parameters:

- Learning Rate = 0.0002
- Beta1 = 0.5
- Beta2 = 0.999


# Hyperparameter Configuration

| Hyperparameter | Value |
|----------------|------:|
| Dataset | Fashion-MNIST |
| Batch Size | 64 |
| Epochs | 30 |
| Latent Dimension | 100 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Loss Function | BCELoss |
| Beta1 | 0.5 |
| Beta2 | 0.999 |

---

# Output

The project generates:

- Synthetic Fashion-MNIST images
- Trained Generator model
- Generated sample images after each epoch

Generated images are saved inside:

output/generated_samples/


# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib


# Applications of GANs

- Image Generation
- Data Augmentation
- Image Super-Resolution
- Face Generation
- Medical Image Synthesis
- Image-to-Image Translation


# Conclusion

This project demonstrates the implementation of a Generative Adversarial Network for synthetic image generation using the Fashion-MNIST dataset. Through adversarial training, the Generator progressively learns to produce realistic images while the Discriminator improves its ability to distinguish between real and generated samples. 
The project highlights the effectiveness of GANs in generative modeling and provides a foundation for more advanced architectures such as DCGAN, Conditional GAN, and StyleGAN.
