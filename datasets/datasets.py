import torch
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Lambda, ConvertImageDtype, Compose

import os

def get_mnist_dataset():
    tform = Compose([ToTensor(), ConvertImageDtype(torch.float32), Lambda(lambda x: x/torch.tensor(1.0, dtype=torch.float32))])
    return (MNIST(root=os.path.join(os.getcwd(), 'datasets'), train=True, download=True, transform=tform), 
            MNIST(root=os.path.join(os.getcwd(), 'datasets'), train=False, download=True, transform=tform))


def get_loaders(train, test, batch_size=32, shuffle=True):
    return DataLoader(train, batch_size=batch_size, shuffle=shuffle), DataLoader(test, batch_size=batch_size, shuffle=shuffle)