import torch
import torch.nn as nn

class DilatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=2):
        super(DilatedConv, self).__init__(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)


if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    dilated_conv = DilatedConv(in_channels=3, out_channels=16)
    output = dilated_conv(x)
    print("Input size:", x.size())
    print("Output size:", output.size())
