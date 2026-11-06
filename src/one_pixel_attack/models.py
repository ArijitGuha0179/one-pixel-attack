"""Notebook-compatible CIFAR-10 models; no downloads at import time."""

import torch
from torch import nn

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]
URLS = {
    "resnet18": "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
    "vgg16": "https://huggingface.co/edadaltocg/vgg16_bn_cifar10/resolve/main/pytorch_model.bin",
}


def build_model(name):
    if name == "small_cnn":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    import timm

    if name == "resnet18":
        model = timm.create_model("resnet18", pretrained=False, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif name == "vgg16":
        model = timm.create_model("vgg16_bn", pretrained=False)
        model.pre_logits = nn.Identity()
        model.head.fc = nn.Linear(512, 10)
    else:
        raise ValueError(f"unknown model: {name}")
    return model


def load_model(name, checkpoint=None):
    model = build_model(name)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["model"] != name or saved["mean"] != MEAN or saved["std"] != STD:
            raise ValueError(
                "checkpoint model/normalization differs from configuration"
            )
        state = saved["state_dict"]
    elif name in URLS:
        state = torch.hub.load_state_dict_from_url(
            URLS[name],
            map_location="cpu",
            weights_only=True,
            file_name=f"edadaltocg_{name}_cifar10.pth",
        )
    else:
        raise ValueError("small_cnn requires a trained --checkpoint")
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
