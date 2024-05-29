"""
Code originates from: https://github.com/mindspore-courses/External-Attention-MindSpore/blob/main/model/attention/UFOAttention.py
Refer to "UFO-ViT: High Performance Linear Vision Transformer without Softmax"
Original paper addresshttps: https://arxiv.org/pdf/2109.14382.pdf
"""
import torch
from torch import nn
from torch.nn import init

class UFOAttention(nn.Module):
    def __init__(self, d_model, d_k=None, d_v=None, h=1, dropout=0.1):
        super(UFOAttention, self).__init__()
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model

        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.gamma = nn.Parameter(torch.randn((1, h, 1, 1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys=None, values=None):
        if keys is None:
            keys = queries
        if values is None:
            values = keys

        B, N = queries.shape[:2]
        q = self.fc_q(queries).view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(B, N, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(B, N, self.h, self.d_v).permute(0, 2, 1, 3)

        kv = torch.matmul(k, v)
        kv_norm = self.XNorm(kv, self.gamma)
        q_norm = self.XNorm(q, self.gamma)
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).reshape(B, N, self.h * self.d_v)
        return self.fc_o(out)

    def XNorm(self, x, gamma):
        norm_tensor = torch.norm(x, 2, -1, keepdim=True)
        return x * gamma / norm_tensor

if __name__ == '__main__':
    dummy_input = torch.randn((25, 49, 512))
    ufo = UFOAttention(d_model=512, h=8)
    output = ufo(dummy_input)
    print(output.shape)

