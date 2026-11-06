"""Original CIFAR-10 split; raw uint8 for attacks, normalized tensors for training."""

from torchvision import datasets, transforms

from .models import MEAN, STD


def cifar10(root, train=False, download=False, tensor=False):
    transform = (
        transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        if tensor
        else None
    )
    return datasets.CIFAR10(
        root=root, train=train, download=download, transform=transform
    )
