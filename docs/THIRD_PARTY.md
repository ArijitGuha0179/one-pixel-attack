# Sources, attribution and redistribution

- **Project notebook:** `notebooks/one-pixel-attack.ipynb`, supplied by the user and
  recovered from the referenced chat's attachment. Copied byte-for-byte. The /mnt/data
  path in the request was not present on this Windows host.
- **Paper:** Su, Vargas and Sakurai, *One Pixel Attack for Fooling Deep Neural Networks*,
  https://arxiv.org/abs/1710.08864. The unavailable local PDF was substituted with
  the public author manuscript for method verification. No paper PDF is redistributed.
- **CIFAR-10:** https://www.cs.toronto.edu/~kriz/cifar.html, downloaded through
  torchvision. Not bundled with the repository.
- **ResNet-18 weights:** https://huggingface.co/edadaltocg/resnet18_cifar10.
- **VGG-16 BN weights:** https://huggingface.co/edadaltocg/vgg16_bn_cifar10.
  These checkpoint sources and normalization constants are preserved from the
  notebook. Upstream URLs use `main`, which can change; run manifests hash the
  loaded model tensors. Weights are not bundled or relicensed by this project.
- **PyTorch / torchvision / timm:** model construction, training, inference and
  dataset loading. **NumPy:** handwritten search and array operations.
  **Matplotlib:** visualization. **unittest:** testing (Python standard library).

The MIT license applies to the project code and documentation to the extent the
contributors hold the rights. Upstream packages, data and weights retain their
own terms. The original paper's authors are credited for the method; this repository
does not claim to have invented One Pixel Attack or differential evolution.
