import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, vgg19_bn, vgg13_bn, vgg11_bn, \
                               resnet50, resnet101, resnet152
from pyzjr.nn.models.bricks.useful_block import DenseBlock
from pyzjr.nn.models.bricks.conv_norm_act import ConvBNReLU


class VggFCN32s(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = model
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        output = self.base_model(x)
        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, num_classes, x.H/1, x.W/1)

        return score  # size=(N, num_classes, x.H/1, x.W/1)

class VggFCN16s(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = model
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        output = self.base_model(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, num_classes, x.H/1, x.W/1)

        return score  # size=(N, num_classes, x.H/1, x.W/1)

class VggFCN8s(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = model
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        output = self.base_model(x)
        x5 = output["x5"]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output["x4"]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output["x3"]  # size=(N, 256, x.H/8,  x.W/8)
        score = self.relu(self.deconv1(x5))  # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))  # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, num_classes, x.H/1, x.W/1)

        return score  # size=(N, num_classes, x.H/1, x.W/1)


class VggFCNs(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = model
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.base_model(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, num_classes, x.H/1, x.W/1)

        return score  # size=(N, num_classes, x.H/1, x.W/1)


class VGGNet(nn.Module):
    def __init__(self, model='vgg16', pretrained=False, requires_grad=True, remove_fc=True, show_params=False):
        ranges = {
            'vgg11': ((0, 4), (4, 8), (8, 15), (15, 22), (22, 29)),
            'vgg13': ((0, 7), (7, 14), (14, 21), (21, 28), (28, 35)),
            'vgg16': ((0, 7), (7, 14), (14, 24), (24, 34), (34, 44)),
            'vgg19': ((0, 7), (7, 14), (14, 27), (27, 40), (40, 53))
        }
        super().__init__()
        if model == "vgg16":
            self.backbone = vgg16_bn(weights='IMAGENET1K_V1' if pretrained else None)
        elif model == "vgg19":
            self.backbone = vgg19_bn(weights='IMAGENET1K_V1' if pretrained else None)
        elif model == "vgg11":
            self.backbone = vgg11_bn(weights='IMAGENET1K_V1' if pretrained else None)
        elif model == "vgg13":
            self.backbone = vgg13_bn(weights='IMAGENET1K_V1' if pretrained else None)
        self.ranges = ranges[model]
        if not requires_grad:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.backbone.classifier

        if show_params:
            for name, param in self.backbone.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.backbone.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output

def _get_fun_model(backbone, fcntype, num_classes):
    model = None
    if fcntype == "FCN8s":
        model = VggFCN8s(model=backbone, num_classes=num_classes)
    elif fcntype == "FCN16s":
        model = VggFCN16s(model=backbone, num_classes=num_classes)
    elif fcntype == "FCN32s":
        model = VggFCN32s(model=backbone, num_classes=num_classes)
    elif fcntype == "FCNs":
        model = VggFCNs(model=backbone, num_classes=num_classes)
    return model

def VggFCN(backbone='vgg16', fcntype="FCN8s", num_classes=1000, pretrained=False):
    """这里仅支持以 vgg_bn 为主干的FCN网络"""
    backbones = VGGNet(model=backbone, pretrained=pretrained)
    model = _get_fun_model(backbones, fcntype, num_classes)
    return model


class ResFCN(nn.Module):
    """这里仅支持以 resnet(50, 101, 152) 为主干的FCN网络"""
    def __init__(self, backbone='resnet152', num_classes=21, pretrained=False, requires_grad=True, remove_fc=True):
        super(ResFCN, self).__init__()
        self.resnet = self._load_resnet(backbone, pretrained, requires_grad, remove_fc)
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        # Decoder
        self.decoder1 = nn.Sequential(ConvBNReLU(2048, num_classes, kernel_size=1, padding=0, bias=True),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1))
        self.layer3_conv = ConvBNReLU(1024, num_classes, kernel_size=1, padding=0, bias=True)
        self.decoder2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.layer2_conv = ConvBNReLU(512, num_classes, kernel_size=1, padding=0, bias=True)
        self.decoder3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)
        x = self.decoder1(x) + self.layer3_conv(x3)
        x = self.decoder2(x) + self.layer2_conv(x2)
        x = self.decoder3(x)

        return x

    def _load_resnet(self, model_type, pretrained, requires_grad, remove_fc):
        if model_type == "resnet50":
            self.resnet = resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_type == "resnet101":
            self.resnet = resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_type == "resnet152":
            self.resnet = resnet152(weights='IMAGENET1K_V1' if pretrained else None)

        if not requires_grad:
            for param in self.resnet.parameters():
                param.requires_grad = False

        if remove_fc:
            del self.resnet.fc

        return self.resnet


class DenseFCN(nn.Module):
    def __init__(self, num_classes=21):
        super(DenseFCN, self).__init__()
        # First convolution
        self.conv1 = ConvBNReLU(3, 64, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.DB1 = DenseBlock(6, 64, 4, 32, 0)
        self.conv2 = ConvBNReLU(256, 128, kernel_size=1, padding=0, bias=True)
        self.DB2 = DenseBlock(12, 128, 4, 32, 0)
        self.conv3 = ConvBNReLU(512, 256, kernel_size=1, padding=0, bias=True)
        self.DB3 = DenseBlock(24, 256, 4, 32, 0)
        self.conv4 = ConvBNReLU(1024, 512, kernel_size=1, padding=0, bias=True)
        self.DB4 = DenseBlock(16, 512, 4, 32, 0)
        self.conv5 = nn.Sequential(ConvBNReLU(1024, 1000, bias=True),
                                   ConvBNReLU(1000, num_classes, kernel_size=1, padding=0, bias=True))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                     ConvBNReLU(num_classes, num_classes, kernel_size=1, padding=0, bias=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                     ConvBNReLU(num_classes, num_classes, kernel_size=1, padding=0, bias=True))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                     ConvBNReLU(num_classes, num_classes, kernel_size=1, padding=0, bias=True))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                     ConvBNReLU(num_classes, num_classes, kernel_size=1, padding=0, bias=True))
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                     ConvBNReLU(num_classes, num_classes, kernel_size=1, padding=0, bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.DB1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.DB2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.DB3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.DB4(x)
        x = self.maxpool(x)
        x = self.conv5(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)

        return x


class MobileFCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(MobileFCN, self).__init__()

        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 深度卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 网络模型声明
        self.encoder1 = conv_bn(in_channels, 32, 2)                                                              # 240,240,32
        self.encoder2 = nn.Sequential(conv_dw(32, 64, 1), conv_dw(64, 128, 2))                             # 120,120,128
        self.encoder3 = nn.Sequential(conv_dw(128, 128, 1), conv_dw(128, 256, 2))                          # 60,60,256
        self.encoder4 = nn.Sequential(conv_dw(256, 256, 1), conv_dw(256, 512, 2))                          # 30,30,512
        self.encoder5 = nn.Sequential(conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1),
                                      conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2),
                                      conv_dw(1024, 1024, 1))                                              # 15,15,1024
        self.relu = nn.ReLU(inplace=True)
        self.decoder1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.decoder2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.decoder3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.decoder4 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.decoder5 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(2)
        self.classifier = nn.Conv2d(2, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x3 = self.encoder3(x)
        x4 = self.encoder4(x3)
        x = self.encoder5(x4)

        x = self.relu(self.decoder1(x))
        x = self.bn1(x + x4)
        x = self.relu(self.decoder2(x))
        x = self.bn2(x + x3)
        x = self.bn3(self.relu(self.decoder3(x)))
        x = self.bn4(self.relu(self.decoder4(x)))
        x = self.bn5(self.relu(self.decoder5(x)))
        x = self.classifier(x)

        return x


class MobileFCN1(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super(MobileFCN1, self).__init__()
        # 标准卷积
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        # 深度卷积
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))

        def up_samp(ch_in, ch_out):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_bn(in_channels, 32, 2)                                                              # 240,240,32
        self.encoder2 = nn.Sequential(conv_dw(32, 64, 1), conv_dw(64, 128, 2))                             # 120,120,128
        self.encoder3 = nn.Sequential(conv_dw(128, 128, 1), conv_dw(128, 256, 2))                          # 60,60,256
        self.encoder4 = nn.Sequential(conv_dw(256, 256, 1), conv_dw(256, 512, 2))                          # 30,30,512
        self.encoder5 = nn.Sequential(conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512, 1),
                                      conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2),
                                      conv_dw(1024, 1024, 1))                                              # 15,15,1024

        self.decoder1 = up_samp(1024, 512)
        self.decoder2 = up_samp(512, 256)
        self.decoder3 = up_samp(256, 128)
        self.decoder4 = up_samp(128, 32)
        self.decoder5 = up_samp(32, 2)
        self.classifier = nn.Conv2d(2, num_classes, kernel_size=(1,1))

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x3 = self.encoder3(x)
        x4 = self.encoder4(x3)
        x = self.encoder5(x4)

        x = self.decoder1(x)
        x = self.decoder2(x + x4)
        x = self.decoder3(x + x3)
        x = self.decoder4(x)
        x = self.decoder5(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    import torch.optim as optim
    batch_size, num_classes, h, w = 2, 21, 32, 32
    # fcn_model = FCN8s(num_classes=num_classes)
    fcn_model = DenseFCN(num_classes=num_classes)
    # fcn_model = VggFCN(backbone='vgg16', fcntype="FCN8s", num_classes=num_classes)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3)
    input = torch.randn(batch_size, 3, h, w, requires_grad=True)
    y = torch.randn(batch_size, num_classes, h, w, requires_grad=False)
    # 损失应当下降
    print("Start Train!")
    print(fcn_model(input).shape)
    for iter in range(10):
        optimizer.zero_grad()
        output = fcn_model(input)
        output = torch.sigmoid(output)
        loss = criterion(output, y)
        loss.backward()
        print(f"epoch {iter}, loss {loss.item()}")
        print(output.shape)
        optimizer.step()