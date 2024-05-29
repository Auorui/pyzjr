"""
Visualizing and Understanding Convolutional Networks
    https://arxiv.org/pdf/1311.2901.pdf
"""
import torch
import torch.nn as nn

class ZFNet(nn.Module):
    """
    ZFNet网络结构和AlexNet保持一致，但是卷积核的大小和步长发生了变化。
    主要改进：
        ➢ 将第一个卷积层的卷积核大小改为了7×7
        ➢ 将第二、第三个卷积层的卷积步长都设置为2
        ➢ 增加了第三、第四个卷积层的卷积核个数
    """
    def __init__(self, num_classes=1000):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),   # [-1, 48, 110, 110]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # [-1, 48, 55, 55]
            nn.Conv2d(48, 128, kernel_size=5, stride=2),            # [-1, 128, 26, 26]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # [-1, 128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # [-1, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # [-1, 192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # [-1, 128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # [-1, 128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    import torchsummary
    input = torch.ones(2, 3, 224, 224).cpu()
    net = ZFNet(num_classes=4)
    net = net.cpu()
    out = net(input)
    print(out)
    print(out.shape)
    torchsummary.summary(net, input_size=(3, 224, 224))
    # Total params: 14,579,268