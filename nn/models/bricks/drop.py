import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DropPath", "Dropout", "MultiSampleDropout", "DropConnect",
           "Standout", "GaussianDropout"]

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    I follow the implementation:
        https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    def __init__(self, drop_prob = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        DropPath类实现了"Drop Path"的正则化技术，用于深度神经网络中的主路径，特别是在残差块的主路径上。Drop Path类似于Dropout，
        但它是在网络的深度路径上随机丢弃整个路径（或称之为跳跃连接）。这有助于训练更加健壮和泛化性能更好的神经网络。
        """
        return self.drop_path(x, self.drop_prob, self.training)

    def __str__(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

    def drop_path(self, x, drop_prob=0., training=False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        output = x.div(keep_prob) * random_tensor.floor()
        return output


class Dropout(nn.Module):
    """This module provides a customizable dropout layer. Dropout is a regularization
    technique commonly used in neural networks during training to prevent overfitting.
    It randomly sets a fraction of input units to zero during each update, which helps
    prevent co-adaptation of hidden units.

    Reference from paper: https://arxiv.org/abs/1207.0580
    """
    def __init__(self, p=.5, training=True, inplace=True):
        super().__init__()
        self.p = p
        self.training = training
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)


class MultiSampleDropout(nn.Module):
    """The core idea is to apply Dropout multiple times to the same input in a single
    forward propagation process, generate multiple different masks, and aggregate the
    results.

    Reference from paper: https://arxiv.org/pdf/1905.09788.pdf"""
    def __init__(self, dropout_rate, num_samples):
        super(MultiSampleDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples

    def forward(self, x):
        outputs = []
        for _ in range(self.num_samples):
            output = F.dropout(x, self.dropout_rate, training=self.training)
            outputs.append(output)
        return torch.mean(torch.stack(outputs), dim=0)


class DropConnect(nn.Module):
    """Dropout works by randomly setting the activation output of neurons to zero,
    while DropConnect randomly sets the weights of the network to zero. This means
    that in DropConnect, the connection (i.e. weight) portion of the network is
    randomly "discarded" rather than outputted.

    Reference from paper: https://proceedings.mlr.press/v28/wan13.pdf"""
    def __init__(self, input_dim, output_dim, drop_prob=0.5, training=True):
        super(DropConnect, self).__init__()
        self.drop_prob = drop_prob
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
        self.training = training

    def forward(self, x):
        if self.training:       # 生成与权重相同形状的掩码, 用掩码乘以权重
            mask = torch.rand(self.weight.size()) > self.drop_prob
            drop_weight = self.weight * mask.float().to(self.weight.device)
        else:       # 在测试时不应用DropConnect，但要调整权重以反映丢弃率
            drop_weight = self.weight * (1 - self.drop_prob)
        return F.linear(x, drop_weight, self.bias)


class Standout(nn.Module):
    """Using the internal state of neural networks to determine which neurons are
    more likely to be retained, making the regularization process more dependent on
    the current behavior of the model. This adaptability enables Standout to achieve
    personalized regularization intensity at different training stages and data points.
    """
    def __init__(self, input_dim, output_dim, training=True):
        super(Standout, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.prob_fc = nn.Linear(input_dim, output_dim)   # 使用一个简单的全连接层来生成保留概率
        self.training = training

    def forward(self, x):
        #Note: Standout should not be applied in evaluation mode
        x_out = self.fc(x)
        if self.training:
            probs = torch.sigmoid(self.prob_fc(x))      # 计算每个神经元的保留概率
            mask = torch.bernoulli(probs).to(x.device)  # 生成与激活大小相同的二值掩码
            x_out = x_out * mask                        # 应用mask
        return x_out


class GaussianDropout(nn.Module):
    """
    When training neural networks, overfitting is prevented by introducing random
    noise that follows a Gaussian distribution. Unlike traditional Dropout, which
    sets activation to zero with a fixed probability, Gaussian Dropout is multiplied
    by a random variable (m), where (m) follows a Gaussian distribution.
    """
    def __init__(self, sigma=0.1, training=True):
        super(GaussianDropout, self).__init__()
        self.sigma = sigma
        self.training = training

    def forward(self, x):
        # Note: Dropout is not used during the testing phase
        if self.training:
            m_with_gaussian_noise = torch.normal(1.0, self.sigma, size=x.size()).to(x.device)
            return x * m_with_gaussian_noise
        else:
            return x