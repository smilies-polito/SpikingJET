"""
ResNet-18 for CIFAR-10, CIFAR-adapted variant.

Architecture notes:
    - First conv is 3x3 stride 1 (not 7x7 stride 2 like the ImageNet variant).
    - No initial maxpool. This adapts the network to CIFAR-10's 32x32 inputs;
      the standard ImageNet ResNet-18 downsamples too aggressively for CIFAR.
    - Four stages of BasicBlocks with channels [64, 128, 256, 512] and 2
      blocks per stage.
    - Final AdaptiveAvgPool2d(1) → 512-dim → Linear(512, 10) classifier.

Total parameters: ~11.2M.
Expected clean test accuracy on CIFAR-10 with 200-epoch training: 93-94%.

References:
    He et al. "Deep Residual Learning for Image Recognition" (2015), arXiv:1512.03385
    FAT-RABBIT (Pourmehrani et al., ITC 2024) uses this exact CIFAR-adapted
    variant for its CIFAR-10 experiments.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['ResNet18', 'BasicBlock']


def _weights_init(m: nn.Module) -> None:
    """Kaiming initialization for Conv2d and Linear layers."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight)


class BasicBlock(nn.Module):
    """
    Standard ResNet BasicBlock (two 3x3 convs with a residual skip).

    Matches the torchvision-style BasicBlock but with the CIFAR-adaptation
    that the first Conv2d has stride matching `stride` (rather than 2 fixed).
    """

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            # 1x1 conv projection for dimension matching
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 CIFAR variant.

    Layer counts (matching the "18" in the name):
        1 (initial conv) + 2*2 (layer1) + 2*2 (layer2) + 2*2 (layer3) + 2*2 (layer4) + 1 (fc) = 18
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64

        # CIFAR-adapted stem: 3x3 stride-1 conv, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Four stages of BasicBlocks
        self.layer1 = self._make_layer(BasicBlock,  64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block: type, planes: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # Self-test: build the network, count params, run a forward pass
    net = ResNet18(num_classes=10)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-18 parameter count: {n_params:,}")
    # Expected ~11.2M

    # Forward pass on a fake CIFAR-10 batch
    x = torch.randn(4, 3, 32, 32)
    y = net(x)
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    if y.shape != (4, 10):
        raise RuntimeError(f"Expected (4, 10), got {tuple(y.shape)}")
    print("Self-test PASSED")
