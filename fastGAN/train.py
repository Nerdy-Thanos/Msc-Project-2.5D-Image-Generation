import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# Root directory for dataset
dataroot = "left"
# Batch size during training
batch_size = 64
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 256
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0001
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

dataset = datasets.ImageFolder(root=dataroot,
							   transform=transforms.Compose([
							   transforms.Resize(image_size),
							   transforms.CenterCrop(image_size),
							   transforms.ToTensor(),
							   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
						   ]))

dataloader = DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=8)

print("y")