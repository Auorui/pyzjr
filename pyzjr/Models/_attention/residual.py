"""
Original paper addresshttps: <https://arxiv.org/pdf/2108.02456.pdf>

"""
import torch
from torch import nn

class ResidualAttention(nn.Module):
    def __init__(self, channel=512, num_class=1000, la=0.2):
        super().__init__()
        self.la = la
        self.fc = nn.Conv2d(in_channels=channel,
                          out_channels=num_class,
                          kernel_size=1,
                          stride=1,
                          bias=False)

    def forward(self, x):
        y_raw = self.fc(x).flatten(2) #b,num_class,hxw
        y_avg = torch.mean(y_raw, dim=2) #b,num_class
        y_max = torch.max(y_raw, dim=2)[0] #b,num_class
        score = y_avg+self.la*y_max
        return score

if __name__ == '__main__':
    input = torch.randn(50,512,7,7)
    resatt = ResidualAttention(channel=512, num_class=1000, la=0.2)
    output = resatt(input)
    print(output.shape)