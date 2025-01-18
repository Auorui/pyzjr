"""
`MnasNet: Platform-Aware Neural Architecture Search for Mobile
<https://arxiv.org/pdf/1807.11626.pdf>`_ paper.
"""
import torch
import torch.nn as nn
from functools import partial

from pyzjr.nn.models._utils import _make_divisible

__all__ = ["MNASNet", "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",]

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997
BatchNorm2d = partial(nn.BatchNorm2d, eps=0.001, momentum=_BN_MOMENTUM)

class _InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor, bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        if kernel_size not in [3, 5]:
            raise ValueError(f"kernel_size should be 3 or 5 instead of {kernel_size}")
        mid_ch = in_ch * expansion_factor
        self.apply_residual = in_ch == out_ch and stride == 1
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats, bn_momentum):
    """Creates a stack of inverted residuals."""
    if repeats < 1:
        raise ValueError(f"repeats should be >= 1, instead got {repeats}")
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor, bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(_InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor, bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _get_depths(alpha):
    """Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down."""
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_make_divisible(depth * alpha, 8) for depth in depths]


class MNASNet(nn.Module):
    def __init__(self, alpha, num_classes=1000, drop_rate=.2):
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha should be greater than 0.0 instead of {alpha}")
        self.alpha = alpha
        self.num_classes = num_classes
        depths = _get_depths(alpha)
        # First layer
        self.SepConv = nn.Sequential(
            # Conv3x3
            nn.Conv2d(3, depths[0], kernel_size=3, padding=1, stride=2, bias=False),   # torch.Size([2, 32, 112, 112])
            BatchNorm2d(depths[0]),
            nn.ReLU(inplace=True),
            # Depthwise separable
            nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False),  # torch.Size([2, 32, 112, 112])
            BatchNorm2d(depths[0]),
            nn.ReLU(inplace=True),
            # Conv1x1
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),  # torch.Size([2, 16, 112, 112])
            BatchNorm2d(depths[1]),
        )
        self.inverted_residuals = nn.Sequential(
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(1280, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.SepConv(x)
        x = self.inverted_residuals(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


def mnasnet0_5(num_classes, **kwargs):
    return MNASNet(0.5, num_classes=num_classes, **kwargs)


def mnasnet0_75(num_classes, **kwargs):
    return MNASNet(0.75, num_classes=num_classes, **kwargs)


def mnasnet1_0(num_classes, **kwargs):
    return MNASNet(1.0, num_classes=num_classes, **kwargs)


def mnasnet1_3(num_classes, **kwargs):
    return MNASNet(1.3, num_classes=num_classes, **kwargs)

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = mnasnet1_0(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # Total params: 5,006,380