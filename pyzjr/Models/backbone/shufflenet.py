"""
ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    <https://arxiv.org/pdf/1707.01083.pdf>
ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    <https://arxiv.org/pdf/1807.11164.pdf>
"""
import torch
import torch.nn as nn

__all__ = ["ShuffleNetV1", "shufflenet_v1_g1", "shufflenet_v1_g2", "shufflenet_v1_g3",
           "shufflenet_v1_g4", "shufflenet_v1_g8", "ShuffleNetV2", "shufflenet_v2_x0_5",
           "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",]

cfg_with_shufflenet = {
    'g1': [144, 288, 576],
    'g2': [200, 400, 800],
    'g3': [240, 480, 960],
    'g4': [272, 544, 1088],
    'g8': [384, 768, 1536],
    'x0_5': [24, 48, 96, 192, 1024],
    'x1_0': [24, 116, 232, 464, 1024],
    'x1_5': [24, 176, 352, 704, 1024],
    'x2_0': [24, 244, 488, 976, 2048],
}

def channel_shuffle(x, groups):
    """Shufflenet uses channel shuffling"""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)   # reshape
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, num_channels, height, width)  # flatten
    return x

class shufflenet_units(nn.Module):
    """可参考 <https://arxiv.org/pdf/1707.01083.pdf> Figure2 (b) and (c)"""
    def __init__(self, in_channels, out_channels, stride, groups):
        super(shufflenet_units, self).__init__()
        mid_channels = out_channels // 4
        self.stride = stride
        self.groups = 1 if in_channels == 24 else groups

        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.GConv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)
        if self.stride == 2:
            short_out = self.shortcut(x)
            out = torch.cat([out, short_out], dim=1)
        return self.relu(out)


class ShuffleNetV1(nn.Module):
    """
    参考 <https://arxiv.org/pdf/1707.01083.pdf> Table 1 实现
    根据论文所述 —— 组数越小表示性能越好
    """
    def __init__(self, groups, stages_out_channels, num_classes=1000, repeat_layers=(4, 8, 4)):
        super(ShuffleNetV1, self).__init__()
        self.groups = groups
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_layers(24, stages_out_channels[0], repeat_layers[0], groups)
        self.stage3 = self._make_layers(stages_out_channels[0], stages_out_channels[1], repeat_layers[1], groups)
        self.stage4 = self._make_layers(stages_out_channels[1], stages_out_channels[2], repeat_layers[2], groups)

        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(stages_out_channels[2], num_classes)

    def _make_layers(self, in_channels, out_channels, repeat_number, groups):
        layers = []
        # 不同的 stage 阶段在第一步都是 stride = 2
        layers.append(shufflenet_units(in_channels, out_channels - in_channels, 2, groups))
        in_channels = out_channels
        for i in range(repeat_number - 1):
            layers.append(shufflenet_units(in_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

################################### 以下为 ShuffleNetV2 ###############################################

class Improved_shufflenet_units(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        self.branch1 = nn.Sequential(
            # DWConv
            nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=inp),
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            # DWConv
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_out_channels,
                 num_classes=1000,
                 improved_units=Improved_shufflenet_units,
                 repeat_layers=(4, 8, 4)):
        super().__init__()
        if len(repeat_layers) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, repeat_layers, self._stage_out_channels[1:]):
            seq = [improved_units(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(improved_units(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

#----------------------------------------------------------------------------------------------------#
def shufflenet_v1_g1(num_classes):
    return ShuffleNetV1(groups=1,
                        stages_out_channels=cfg_with_shufflenet['g1'],
                        num_classes=num_classes)

def shufflenet_v1_g2(num_classes):
    return ShuffleNetV1(groups=2,
                        stages_out_channels=cfg_with_shufflenet['g2'],
                        num_classes=num_classes)

def shufflenet_v1_g3(num_classes):
    return ShuffleNetV1(groups=3,
                        stages_out_channels=cfg_with_shufflenet['g3'],
                        num_classes=num_classes)

def shufflenet_v1_g4(num_classes):
    return ShuffleNetV1(groups=4,
                        stages_out_channels=cfg_with_shufflenet['g4'],
                        num_classes=num_classes)

def shufflenet_v1_g8(num_classes):
    return ShuffleNetV1(groups=8,
                        stages_out_channels=cfg_with_shufflenet['g8'],
                        num_classes=num_classes)

def shufflenet_v2_x0_5(num_classes):
    return ShuffleNetV2(cfg_with_shufflenet['x0_5'],
                        num_classes=num_classes)

def shufflenet_v2_x1_0(num_classes):
    return ShuffleNetV2(cfg_with_shufflenet['x1_0'],
                        num_classes=num_classes)

def shufflenet_v2_x1_5(num_classes):
    return ShuffleNetV2(cfg_with_shufflenet['x1_5'],
                        num_classes=num_classes)

def shufflenet_v2_x2_0(num_classes):
    return ShuffleNetV2(cfg_with_shufflenet['x2_0'],
                        num_classes=num_classes)


if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = shufflenet_v2_x2_0(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # shufflenet_v1_g1 Total params: 1,880,296
    # shufflenet_v1_g2 Total params: 1,834,852
    # shufflenet_v1_g3 Total params: 1,781,032
    # shufflenet_v1_g4 Total params: 1,734,136
    # shufflenet_v1_g8 Total params: 1,797,304
    # shufflenet_v2_x0_5 Total params: 345,892
    # shufflenet_v2_x1_0 Total params: 1,257,704
    # shufflenet_v2_x1_5 Total params: 2,482,724
    # shufflenet_v2_x2_0 Total params: 5,353,192