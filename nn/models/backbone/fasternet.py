"""
Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks
    <https://arxiv.org/pdf/2303.03667.pdf>
Reference from: https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py
Blog records: https://blog.csdn.net/m0_62919535/article/details/136334105
"""
import torch
import torch.nn as nn
from pyzjr.nn.models.bricks.drop import DropPath

__all__=["FasterNet", "FasterNetBlock", "fasternet_t0", "fasternet_t1", "fasternet_t2",
         "fasternet_s", "fasternet_m", "fasternet_l"]

class PartialConv(nn.Module):
    def __init__(self, dim, n_div=4, kernel_size=3, forward='split_cat'):
        """
        PartialConv 模块

        Args:
            dim (int): 输入张量的通道数。
            n_div (int): 分割通道数的分母，用于确定部分卷积的通道数。
            forward (str): 使用的前向传播方法，可选 'slicing' 或 'split_cat'。
        """
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class FasterNetBlock(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0, forward='split_cat'):
        super().__init__()
        self.pconv = PartialConv(dim, forward=forward)
        self.conv1 = nn.Conv2d(dim, dim * expand_ratio, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim * expand_ratio)
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(dim * expand_ratio, dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x


class FasterNet(nn.Module):
    def __init__(self, in_channel=3, embed_dim=40, act_layer=None,
                 num_classes=1000, depths=None, drop_rate=0.0):
        super().__init__()
        # Embedding
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, embed_dim, 4, stride=4, bias=False),
            nn.BatchNorm2d(embed_dim),
            act_layer()
        )
        drop_path_list = [x.item() for x in torch.linspace(0, drop_rate, sum(depths))]
        self.feature = []
        embed_dim = embed_dim
        for idx, depth in enumerate(depths):
            self.feature.append(nn.Sequential(
                *[FasterNetBlock(embed_dim, act_layer=act_layer, drop_path_rate=drop_path_list[sum(depths[:idx]) + i]) for i in range(depth)]
            ))
            if idx < len(depths) - 1:
                # Merging
                self.feature.append(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim * 2, 2, stride=2, bias=False),
                    nn.BatchNorm2d(embed_dim * 2),
                    act_layer()
                ))
                embed_dim = embed_dim * 2

        self.feature = nn.Sequential(*self.feature)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(embed_dim, 1280, 1, bias=False)
        self.act_layer = act_layer()
        self.fc = nn.Linear(1280, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.feature(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.act_layer(x)
        x = self.fc(x.flatten(1))

        return x


def fasternet_t0(num_classes, drop_path_rate=0.0):
    return FasterNet(embed_dim=40,
                     act_layer=nn.GELU,
                     num_classes=num_classes,
                     depths=[1, 2, 8, 2],
                     drop_rate=drop_path_rate
                     )

def fasternet_t1(num_classes, drop_path_rate=0.02):
    return FasterNet(embed_dim=64,
                     act_layer=nn.GELU,
                     num_classes=num_classes,
                     depths=[1, 2, 8, 2],
                     drop_rate=drop_path_rate
                     )

def fasternet_t2(num_classes, drop_path_rate = 0.05):
    return FasterNet(embed_dim=96,
                     act_layer=nn.ReLU,
                     num_classes=num_classes,
                     depths=[1, 2, 8, 2],
                     drop_rate=drop_path_rate
                     )

def fasternet_s(num_classes, drop_path_rate = 0.03):
    return FasterNet(embed_dim=128,
                     act_layer=nn.ReLU,
                     num_classes=num_classes,
                     depths=[1, 2, 13, 2],
                     drop_rate=drop_path_rate
                     )

def fasternet_m(num_classes, drop_path_rate = 0.05):
    return FasterNet(embed_dim=144,
                     act_layer=nn.ReLU,
                     num_classes=num_classes,
                     depths=[3, 4, 18, 3],
                     drop_rate=drop_path_rate
                     )

def fasternet_l(num_classes, drop_path_rate = 0.05):
    return FasterNet(embed_dim=192,
                     act_layer=nn.ReLU,
                     num_classes=num_classes,
                     depths=[3, 4, 18, 3],
                     drop_rate=drop_path_rate
                     )

if __name__=="__main__":
    import torchsummary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.ones(2, 3, 224, 224).to(device)
    net = fasternet_t2(num_classes=4)
    net = net.to(device)
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # t0 Total params: 2,629,624 Estimated Total Size (MB): 57.70
    # t1 Total params: 6,319,492 Estimated Total Size (MB): 100.02
    # t2 Total params: 13,707,012 Estimated Total Size (MB): 165.87
    # s Total params: 29,905,156 Estimated Total Size (MB): 304.56
    # m Total params: 52,245,588 Estimated Total Size (MB): 568.01
    # l Total params: 92,189,316 Estimated Total Size (MB): 843.09