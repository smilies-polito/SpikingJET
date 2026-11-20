"""
VGG-9 for CIFAR-10.

Architecture notes:
    - 6 convolutional layers + 3 fully-connected layers = 9 weight layers total
    - Three conv blocks, each with 2 conv layers of the same channel count
      followed by MaxPool. Channels progress 64 → 128 → 256.
    - BatchNorm2d after every conv (not part of the original VGG, but standard
      practice for CIFAR variants and improves both training stability and
      final accuracy).
    - Classifier: 3 FC layers with BN and Dropout for regularization.

Total parameters: ~5.9M.
Expected clean test accuracy on CIFAR-10 with 200-epoch training: 91-93%.

References:
    Simonyan & Zisserman "Very Deep Convolutional Networks for Large-Scale
    Image Recognition" (2014), the original VGG paper.
    Cavagnero et al. (IOLTS 2022) use a similar VGG-like architecture for
    CIFAR-10 fault tolerance work.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init


__all__ = ['VGG9']


def _weights_init(m: nn.Module) -> None:
    """Kaiming initialization for Conv2d and Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if isinstance(m, nn.Linear) and m.bias is not None:
            init.zeros_(m.bias)


class VGG9(nn.Module):
    """
    VGG-9 for CIFAR-10.

    Layer structure:
        Features:
            Block 1: [Conv(3→64), BN, ReLU, Conv(64→64), BN, ReLU, MaxPool(2)]
            Block 2: [Conv(64→128), BN, ReLU, Conv(128→128), BN, ReLU, MaxPool(2)]
            Block 3: [Conv(128→256), BN, ReLU, Conv(256→256), BN, ReLU, MaxPool(2)]
            → spatial: 32 → 16 → 8 → 4
            → final feature map: (batch, 256, 4, 4)
        Classifier:
            Flatten → Linear(256*4*4=4096 → 512), BN, ReLU, Dropout(0.5)
            → Linear(512 → 512), BN, ReLU, Dropout(0.5)
            → Linear(512 → num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 → 64 → 64, then downsample
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 64 → 128 → 128, then downsample
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 128 → 256 → 256, then downsample
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # After 3 pooling stages, feature map is 4x4 with 256 channels
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Self-test: build the network, count params, run a forward pass
    net = VGG9(num_classes=10)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"VGG-9 parameter count: {n_params:,}")
    # Expected ~5.9M

    # Forward pass on a fake CIFAR-10 batch
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    if y.shape != (4, 10):
        raise RuntimeError(f"Expected (4, 10), got {tuple(y.shape)}")
    print("Self-test PASSED")
