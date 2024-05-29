"""
Copyright (c) 2024, Auorui.
All rights reserved.

Activation function pytorch handwriting implementation

Time: 2024-01-02

After a day of hard work, I successfully implemented activation functions in PyTorch, some of which were in place operations.
During this process, I spent a lot of time researching materials and formulas to ensure that my implementation was the same as
the official one. I have not made any additional modifications to the parts that have already implemented official functions,
but there may still be room for improvement in terms of details.

At the end, there is a draft record of my experiment that you can test.

Second update: 2024-02-19

address: <https://arxiv.org/pdf/2402.09092.pdf>

The good news is that I recently saw a great paper that proposed a comprehensive survey involving 400 activation functions,
which is several times larger than previous surveys. So I have added some new implementation of activation functions here.
Since the following sections are all implemented according to formulas, the accuracy is still in the experimental stage.
However, those who are interested can study and research it themselves.

Most of them have been implemented here, but the parameters involved are not specified or have a certain range, or the
parameters that require training transformation have not been implemented.
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = ['Hardsigmoid', 'Hardtanh', 'Hardswish', 'Hardshrink', 'Threshold',  'Sigmoid', 'Tanh',
           'Softshrink', 'Softplus', 'Softmin', 'LogSoftmax', 'Softsign', 'Softmax',
           'ReLU', 'RReLU', 'ReLU6', 'LeakyReLU', "FReLU", 'PReLU', 'Mish',  'ELU', 'CELU', 'SELU', 'GLU', 'GELU',
           'SiLU', 'Swish', 'LogSigmoid', 'Tanhshrink', "AconC", "MetaAconC",
           "plot_activation_function",]

__all__.extend(["ShiftedScaledSigmoid", "VariantSigmoid", "ScaledHyperbolicTangent",
                "Arctan", "ArctanGR", "SigmoidAlgebraic", "TripleStateSigmoid",
                "ImprovedLogisticSigmoid", "SigLin", "PTanh", "SoftRootSign", "SoftClipping",
                "SmoothStep", "Elliott", "SincSigmoid", "SigmoidGumbel", "NewSigmoid",
                "Root2Sigmoid", "LogLog", "CLogLog", "ModifiedCLogLog", "SechSig", "TanhSig",
                "SymMSAF", "RootSig", "SGELU", "CaLU", "LaLU", "CoLU", "GeneralizedSwish",
                "ExponentialSwish", "DerivativeSigmoid", "Gish", "Logish", "LogLogish",
                "ExpExpish", "SelfArctan", "ParametricLogish", "Phish", "Suish", "TSReLU",
                "TBSReLU", "dSiLU", "DoubleSiLU", "MSiLU", "TSiLU", "ATSiLU", "RectifiedHyperbolicSecant",
                "LiSHT", "Smish", "TanhExp", "Serf", "SinSig", "SiELU", "ShiftedReLU", "SlReLU",
                "NReLU", "SineReLU", "Minsin", "SLU", "ReSP", "TRec", "mReLU", "SoftModulusQ",
                "SoftModulusT", "EPLAF", "DRLU", "DisReLU", "FlattedTSwish", "ReLUSwish", "OAF",
                "REU", "SigLU", "SaRa", "Maxsig", "ThLU", "DiffELU", "PolyLU", "PoLU", "PFLU",
                "FPFLU", "ELiSH", "SQNL", "SQLU", "Squish", "SqREU", "SqSoftplus", "LogSQNL",
                "ISRLU", "MEF", "SQRT", "BentIdentity", "Mishra", "SahaBora", "Logarithmic",
                "Symexp", "PUAF", "PSoftplus", "ArandaOrdaz", "PMAF", "PRBF", "MArcsinh",
                "HyperSinh", "Arctid", "Sine", "Cosine", "Cosid", "Sinp", "GCU", "ASU", "HcLSH",
                "Exponential", "NCU", "Triple", "SQU", "SCMish", "TBSReLUL", "Swim", "SquarePlus",
                "StepPlus", "BipolarPlus", "vReLUPlus", "BReLUPlus", "HardTanhPlus", "SwishPlus",
                "MishPlus", "LogishPlus", "SoftsignPlus", "SignReLUPlus"])


class PyZjrActivation(nn.Module):
    """继承自 nn.Module 的简单激活函数类"""
    def __init__(self):
        super().__init__()


class Sigmoid(PyZjrActivation):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x):
        return x.sigmoid_() if self.inplace else self._sigmoid(x)

class Tanh(PyZjrActivation):
    def __init__(self, inplace = False):
        super().__init__()
        self.inplace = inplace

    def _tanh(self, x):
        return (2 / (1 + torch.exp(-2 * x))) - 1

    def forward(self, x):
        return x.tanh_() if self.inplace else self._tanh(x)

class ReLU(PyZjrActivation):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def _relu(self, x):
        return torch.max(torch.tensor(0.0), x)

    def forward(self, x):
        return x.relu_() if self.inplace else self._relu(x)

class ReLU6(PyZjrActivation):
    def __init__(self, inplace=False):
        super(ReLU6, self).__init__()
        self.inplace = inplace

    def _relu6(self, x):
        return torch.clamp(x, min=0.0, max=6.0)

    def forward(self, x):
        if self.inplace:
            return x.clamp_(min=0.0, max=6.0)
        else:
            return self._relu6(x)

class FReLU(PyZjrActivation):
    def __init__(self, dim, kernel=3, init_weight=False):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel, 1, 1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        if init_weight:
            self.apply(self._init_weight)

    def _init_weight(self, m):
        init = nn.init.normal(mean=0, std=.02)
        zeros = nn.init.constant(0.)
        ones = nn.init.constant(1.)
        if isinstance(m, nn.Conv2d):
            init(m.weight)
            zeros(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            ones(m.weight)
            zeros(m.bias)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))

class LeakyReLU(PyZjrActivation):
    def __init__(self, negative_slope=0.01, inplace=False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def _leakyrelu(self, x):
        if self.inplace:
            return x.mul_(torch.where(x > 0, torch.tensor(1.0), torch.tensor(self.negative_slope)))
        else:
            return torch.where(x > 0, x, x * self.negative_slope)

    def forward(self, x):
        return self._leakyrelu(x)


class RReLU(PyZjrActivation):
    def __init__(self, lower=1.0 / 8, upper=1.0 / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def _rrelu(self, x):
        noise = torch.empty_like(x).uniform_(self.lower, self.upper)
        if self.inplace:
            return x.mul_(torch.where(x < 0, noise, torch.tensor(1.0)))
        else:
            return torch.where(x < 0, x * noise, x)

    def forward(self, x):
        return self._rrelu(x)

class PReLU(PyZjrActivation):
    def __init__(self, num_parameters=1, init=0.25):
        super(PReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.full((num_parameters,), init))

    def _prelu(self, x):
        return torch.where(x >= 0, x, self.weight * x)

    def forward(self, x):
        return self._prelu(x)

class Threshold(PyZjrActivation):
    def __init__(self, threshold=0.5, value=0.0, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def _threshold(self, x):
        if self.inplace:
            return x.threshold_(self.threshold, self.value)
        else:
            return F.threshold(x, self.threshold, self.value, self.inplace)

    def forward(self, x):
        return self._threshold(x)

class Softsign(PyZjrActivation):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def _soft_sign(self, x):
        if self.inplace:
            return x.div_(1 + torch.abs(x))
        else:
            return x / (1 + torch.abs(x))

    def forward(self, x):
        return self._soft_sign(x)

class Tanhshrink(Tanh):
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)
        self.inplace = inplace

    def _tanh_shrink(self, x):
        return x.sub_(self._tanh(x)) if self.inplace else x - self._tanh(x)

    def forward(self, x):
        return self._tanh_shrink(x)

class Softmin(PyZjrActivation):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def _softmin(self, x):
        exp_x = torch.exp(-x)
        softmax = exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
        return softmax

    def forward(self, x):
        return self._softmin(x)

class Softmax(PyZjrActivation):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def _softmax(self, x):
        exp_x = torch.exp(x)
        softmax = exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)
        return softmax

    def forward(self, x):
        return self._softmax(x)

class Mish(PyZjrActivation):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def _mish(self, x):
        if self.inplace:
            return x.mul_(torch.tanh(F.softplus(x)))
        else:
            return x * torch.tanh(F.softplus(x))

    def forward(self, x):
        return self._mish(x)

# Swish, also known as SiLU.
class SiLU(PyZjrActivation):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def _silu(self, x):
        return x.mul_(torch.sigmoid(x)) if self.inplace else x * torch.sigmoid(x)

    def forward(self, x):
        return self._silu(x)

class Swish(SiLU):
    def __init__(self, inplace=False):
        super(Swish, self).__init__(inplace=inplace)

    def forward(self, x):
        return self._silu(x)

class Hardswish(PyZjrActivation):
    def __init__(self, inplace=False):
        super(Hardswish, self).__init__()
        self.inplace = inplace

    def _hardswish(self, x):
        inner = F.relu6(x + 3.).div_(6.)
        return x.mul_(inner) if self.inplace else x.mul(inner)

    def forward(self, x):
        return self._hardswish(x)

class ELU(PyZjrActivation):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
        self.alpha = alpha

    def _elu(self, x):
        return torch.where(x < 0, self.alpha * (torch.exp(x) - 1), x)

    def forward(self, x):
        return self._elu(x)

class CELU(PyZjrActivation):
    def __init__(self, alpha=1.0, inplace=False):
        super(CELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def _celu(self, x, alpha):
        if self.inplace:
            x[x < 0] = alpha * (torch.exp(x[x < 0] / alpha) - 1)
            return x
        else:
            return torch.where(x < 0, alpha * (torch.exp(x / alpha) - 1), x)

    def forward(self, x):
        return self._celu(x, self.alpha)


class SELU(PyZjrActivation):
    def __init__(self):
        super(SELU, self).__init__()
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717

    def _selu(self, x):
        return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

    def forward(self, x):
        return self._selu(x)

class GLU(PyZjrActivation):
    def __init__(self, dim=-1):
        super(GLU, self).__init__()
        self.dim = dim

    def _glu(self, x):
        mid = x.size(self.dim) // 2
        return x.narrow(self.dim, 0, mid) * torch.sigmoid(x.narrow(self.dim, mid, mid))

    def forward(self, x):
        return self._glu(x)

class GELU(PyZjrActivation):
    def __init__(self, inplace=False):
        super(GELU, self).__init__()
        self.inplace = inplace

    def _gelu(self, x):
        if self.inplace:
            return x.mul_(0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))))
        else:
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        return self._gelu(x)

class Hardshrink(PyZjrActivation):
    def __init__(self, lambd=0.5):
        super(Hardshrink, self).__init__()
        self.lambd = lambd

    def _hardshrink(self, x):
        return torch.where(x < -self.lambd, x, torch.where(x > self.lambd, x, torch.tensor(0.0)))


    def forward(self, x):
        return self._hardshrink(x)


class Hardsigmoid(PyZjrActivation):
    def __init__(self, inplace=False):
        super(Hardsigmoid, self).__init__()
        self.inplace = inplace

    def _hardsigmoid(self, x):
        if self.inplace:
            return x.add_(3.).clamp_(0., 6.).div_(6.)
        else:
            return F.relu6(x + 3.0) / 6.0

    def forward(self, x):
        return self._hardsigmoid(x)


class Hardtanh(PyZjrActivation):
    def __init__(self, inplace=False):
        super(Hardtanh, self).__init__()
        self.inplace = inplace

    def _hardtanh(self, x):
        return x.clamp_(-1.0, 1.0) if self.inplace else torch.clamp(x, min=-1.0, max=1.0)

    def forward(self, x):
        return self._hardtanh(x)

class LogSoftmax(PyZjrActivation):
    def __init__(self, dim=-1, inplace=False):
        super(LogSoftmax, self).__init__()
        self.inplace = inplace
        self.dim = dim

    def _logsoftmax(self, x):
        max_vals, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_exp = torch.exp(x - max_vals)
        x_softmax = x_exp / torch.sum(x_exp, dim=self.dim, keepdim=True)
        if self.inplace:
            x.copy_(torch.log(x_softmax))
            return x
        else:
            return torch.log(x_softmax)

    def forward(self, x):
        return self._logsoftmax(x)

class LogSigmoid(PyZjrActivation):
    def __init__(self, inplace=False):
        super(LogSigmoid, self).__init__()
        self.inplace = inplace

    def _logsigmoid(self, x):
        if self.inplace:
            return x.sigmoid_().log_()
        else:
            return torch.log(torch.sigmoid(x))

    def forward(self, x):
        return self._logsigmoid(x)

class Softplus(PyZjrActivation):
    def __init__(self, beta = 1, threshold = 20, inplace=False):
        super(Softplus, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.inplace = inplace

    def _softplus(self, x):
        if self.inplace:
            return x.add_(1 / self.beta).log_()
        else:
            return torch.where(x > self.threshold, x, 1 / self.beta * torch.log(1 + torch.exp(self.beta * x)))

    def forward(self, x):
        return self._softplus(x)

class Softshrink(PyZjrActivation):
    def __init__(self, lambd=0.5):
        super(Softshrink, self).__init__()
        self.lambd = lambd

    def _softshrink(self, x):
        return torch.where(x < -self.lambd, x + self.lambd, torch.where(x > self.lambd, x - self.lambd, torch.tensor(0., device=x.device, dtype=x.dtype)))

    def forward(self, x):
        return self._softshrink(x)


# AconC与MetaAconC是从YOLOv5中复制过来的，暂时没有研究
class AconC(nn.Module):
    r""" ACON activation (activate or not)
    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x


class MetaAconC(nn.Module):
    r""" ACON activation (activate or not)
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is generated by a small network
    according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, c1, k=1, s=1, r=16):  # ch_in, kernel, stride, r
        super().__init__()
        c2 = max(r, c1 // r)
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
        self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
        # self.bn1 = nn.BatchNorm2d(c2)
        # self.bn2 = nn.BatchNorm2d(c1)

    def forward(self, x):
        y = x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True)
        # batch-size 1 bug/instabilities https://github.com/ultralytics/yolov5/issues/2891
        # beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(y)))))  # bug/unstable
        beta = torch.sigmoid(self.fc2(self.fc1(y)))  # bug patch BN layers removed
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(beta * dpx) + self.p2 * x

def plot_activation_function(activation_class, input_range=(-20, 20), num_points=10000, save_dir=None, format="png"):
    """matplotlib 绘制以上激活函数图像, 小部分无法绘制
    经过测试，FReLU与GLU,以及AconC与MetaAconC无法绘制
    """
    x_values = np.linspace(input_range[0], input_range[1], num_points)
    activation_func = activation_class()

    with torch.no_grad():
        y_values = activation_func(torch.tensor(x_values)).numpy()

    # Handle extreme values for better visualization
    y_values[np.isinf(y_values)] = np.nan
    y_values[np.isnan(y_values)] = np.max(np.abs(y_values[~np.isnan(y_values)]))

    plt.plot(x_values, y_values, label=activation_class.__name__)
    title = f"{activation_class.__name__} Activation Function"
    plt.title(title)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, activation_class.__name__) + f".{format}")
        print(f"Plot saved at: {save_dir}")
    plt.show()



# Second update---------------------------------------------------------------------------------

class ShiftedScaledSigmoid(PyZjrActivation):
    """Shifted and scaled sigmoid (SSS)
    where a and b are predetermined parameters; Arai and Imamura used a = 0.02 and b = 600.
    """
    def __init__(self, a=0.02, b=600):
        super(ShiftedScaledSigmoid, self).__init__()
        self.a = a
        self.b = b

    def shifted_scaled_sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.a * (x - self.b)))

    def forward(self, x):
        return self.shifted_scaled_sigmoid(x)

class VariantSigmoid(PyZjrActivation):
    """Variant Sigmoid Function (VSF)
    'VSF' is an older parametric variant of the logistic sigmoid
    a,b,c have not provided specific suggestions and need to be tested based on actual situations
    """
    def __init__(self, a=1., b=.5, c=.1):
        super(VariantSigmoid, self).__init__()
        self.a = a
        self.b = b
        self.c = c

    def variant_sigmoid_function(self, x):
        return self.a / (1 + torch.exp(-self.b * x)) - self.c

    def forward(self, x):
        return self.variant_sigmoid_function(x)

class ScaledHyperbolicTangent(PyZjrActivation):
    """Scaled Hyperbolic Tangent (stanh)
    where a and b are fixed hyperparameters that control the scaling of the function.
    Lecun et al. proposed using a = 1.7159 and b = 2 / 3
    """
    def __init__(self, a=1.7159, b=2/3):
        super(ScaledHyperbolicTangent, self).__init__()
        self.a = a
        self.b = b

    def scaled_hyperbolic_tangent(self, x):
        return self.a * torch.tanh(self.b * x)

    def forward(self, x):
        return self.scaled_hyperbolic_tangent(x)

class Arctan(PyZjrActivation):
    """Arctan Activation Function"""
    def __init__(self):
        super(Arctan, self).__init__()

    def forward(self, x):
        return torch.atan(x)

class ArctanGR(PyZjrActivation):
    """ArctanGR Activation Function (Scaled Arctan)

    Args:
        sqrt2: 1 + torch.sqrt(torch.tensor(2)) / 2
        pi: torch.pi
        sqrt5: 1 + torch.sqrt(torch.tensor(5)) / 2
        euler: torch.exp(torch.tensor(1))
    """
    def __init__(self, type='sqrt2'):
        super(ArctanGR, self).__init__()
        self.type = type

    def arctan_gr(self, x):
        if self.type == "sqrt2":
            return torch.atan(x) / (1 + torch.sqrt(torch.tensor(2)) / 2)
        elif self.type == "pi":
            return torch.atan(x) / torch.pi
        elif self.type == "sqrt5":
            return torch.atan(x) / (1 + torch.sqrt(torch.tensor(5)) / 2)
        elif self.type == "euler":
            return torch.atan(x) / torch.exp(torch.tensor(1))
        else:
            raise ValueError("Unsupported scaling factor")

    def forward(self, x):
        return self.arctan_gr(x)

class SigmoidAlgebraic(PyZjrActivation):
    """Sigmoid Algebraic Activation Function
    where a ≥ 0 is a parameter
    """
    def __init__(self, a=.1):
        super(SigmoidAlgebraic, self).__init__()
        if a < 0:
            raise ValueError("Parameter 'a' must be greater than or equal to 0.")
        self.a = a

    def sigmoid_algebraic(self, x):
        return 1 / (1 + torch.exp(
            -(x * (1 + self.a * torch.abs(x))) / (1 + torch.abs(x) * (1 + self.a * torch.abs(x)))
                    ))

    def forward(self, x):
        return self.sigmoid_algebraic(x)

class TripleStateSigmoid(PyZjrActivation):
    """Triple-state Sigmoid Activation Function
    Where a and b are fixed parameters, but not provided in the text
    """
    def __init__(self, a=.1, b=.1):
        super(TripleStateSigmoid, self).__init__()
        self.a = a
        self.b = b

    def triple_state_sigmoid(self, x):
        base = 1 + torch.exp(-x)
        a_base = 1 + torch.exp(-x + self.a)
        b_base = 1 + torch.exp(-x + self.b)
        return (1 / base) * ((1 / base) + (1 / a_base) + (1 / b_base))

    def forward(self, x):
        return self.triple_state_sigmoid(x)

class ImprovedLogisticSigmoid(PyZjrActivation):
    """Improved Logistic Sigmoid Activation Function
    where a and b are fixed parameters; a controls the slope and b is a thresholding parameter
    """
    def __init__(self, a=None, b=.2):
        super(ImprovedLogisticSigmoid, self).__init__()
        self.b = torch.tensor(b)
        amin = torch.exp(-self.b) / ((1 + torch.exp(-self.b))**2)
        if a is None:
            a = amin + 1e-2
        assert a > amin, "Parameter 'a' must be greater than amin."
        self.a = a
        self.amin = amin

    def improved_logistic_sigmoid(self, x):
        _sigmoid = torch.where(x >= self.b, self.a * (x - self.b) + torch.sigmoid(self.b),
                               torch.where((x > -self.b) & (x < self.b), torch.sigmoid(x),
                                           self.a * (x + self.b) + torch.sigmoid(self.b)))
        return _sigmoid

    def forward(self, x):
        return self.improved_logistic_sigmoid(x)

class SigLin(PyZjrActivation):
    """Combination of the sigmoid and linear activation (SigLin)
    Roodschild, Gotay Sardiñas, and Will experimented with a ∈ {0, 0.05, 0.1, 0.15}
    """
    def __init__(self, a=.15):
        super(SigLin, self).__init__()
        self.a = a

    def _siglin(self, x):
        return torch.sigmoid(x) + self.a * x

    def forward(self, x):
        return self._siglin(x)

class PTanh(PyZjrActivation):
    """Penalized Hyperbolic Tangent (ptanh) Activation Function
    where a∈(1，∞).
    """
    def __init__(self, a=1.2):
        super(PTanh, self).__init__()
        assert a > 1, "Parameter 'a' must be greater than 1."
        self.a = a

    def _ptanh(self, x):
        return torch.where(x >= 0, torch.tanh(x), torch.tanh(x) / self.a)

    def forward(self, x):
        return self._ptanh(x)

class SoftRootSign(PyZjrActivation):
    """Soft-root-sign (SRS)
    the authors Li and Zhou propose using a = 2 and b = 3
    """
    def __init__(self, a=2, b=3):
        super(SoftRootSign, self).__init__()
        self.a = a
        self.b = b

    def srs(self, x):
        return x / (x / self.a + torch.exp(-x / self.b))

    def forward(self, x):
        return self.srs(x)

class SoftClipping(PyZjrActivation):
    """Soft Clipping (SC)"""
    def __init__(self, a=1.1):
        super(SoftClipping, self).__init__()
        self.a = torch.tensor(a)

    def soft_clipping(self, x):
        return (1 / self.a) * torch.log(
            (1 + torch.exp(self.a * x)) / (1 + torch.exp(self.a * (x - 1)))
        )

    def forward(self, x):
        return self.soft_clipping(x)


class SmoothStep(PyZjrActivation):
    """Smooth Step
    where a is a fixed hyperparameter
    """
    def __init__(self, a=1):
        super(SmoothStep, self).__init__()
        self.a = a
        self.half_a = a / 2

    def smooth_step(self, x):
        return torch.where(x >= self.half_a, torch.tensor(1.0),
                           torch.where((-self.half_a <= x) & (x <= self.half_a),
                                       -2 * self.a**3 * x**3 + (3/2) * self.a**2 * x + (1/2), torch.tensor(0.0)))

    def forward(self, x):
        return self.smooth_step(x)

class Elliott(PyZjrActivation):
    """Elliott Activation Function
    Elliott activation function is one of the earliest proposed activation functions to
    replace the logistic sigmoid or tanh activation functions.
    """
    def __init__(self):
        super(Elliott, self).__init__()

    def _elliott(self, x):
        return 0.5 * x / (1 + torch.abs(x)) + 0.5

    def forward(self, x):
        return self._elliott(x)

class SincSigmoid(PyZjrActivation):
    """Sinc-Sigmoid Activation Function
    where sinc(x) is the unnormalized sinc function
    """
    def __init__(self):
        super(SincSigmoid, self).__init__()

    def sinc_sigmoid(self, x):
        return torch.sinc(torch.sigmoid(x))

    def forward(self, x):
        return self.sinc_sigmoid(x)


class SigmoidGumbel(PyZjrActivation):
    """Sigmoid-Gumbel Activation Function"""
    def __init__(self):
        super(SigmoidGumbel, self).__init__()

    def sigmoid_gumbel(self, x):
        return (1 / (1 + torch.exp(-x))) * torch.exp(-torch.exp(-x))

    def forward(self, x):
        return self.sigmoid_gumbel(x)

class NewSigmoid(PyZjrActivation):
    """NewSigmoid Activation Function"""
    def __init__(self):
        super(NewSigmoid, self).__init__()

    def new_sigmoid(self, x):
        return (torch.exp(x) - torch.exp(-x)) / torch.sqrt(
            (2 * (torch.exp(2*x) + torch.exp(-2*x)))
        )

    def forward(self, x):
        return self.new_sigmoid(x)

class Root2Sigmoid(PyZjrActivation):
    """Root2Sigmoid Activation Function"""
    def __init__(self):
        super(Root2Sigmoid, self).__init__()

    def root2_sigmoid(self, x):
        sqrt_2 = torch.sqrt(torch.tensor(2))
        sqrt_2_x = sqrt_2 ** x
        sqrt_2_negative_x = sqrt_2 ** (-x)
        sqrt_2_2x = sqrt_2 ** (2*x)
        sqrt_2_negative_2x = sqrt_2 ** (-2*x)
        return (sqrt_2_x - sqrt_2_negative_x) / \
               (2 * sqrt_2 * torch.sqrt(2 * sqrt_2 * torch.sqrt(2 * (sqrt_2_2x + sqrt_2_negative_2x))))

    def forward(self, x):
        return self.root2_sigmoid(x)

class LogLog(PyZjrActivation):
    """LogLog Activation Function"""
    def __init__(self):
        super(LogLog, self).__init__()

    def _loglog(self, x):
        return torch.exp(-torch.exp(-x))

    def forward(self, x):
        return self._loglog(x)

class CLogLog(PyZjrActivation):
    """Complementary Log-Log (cLogLog) Activation Function"""
    def __init__(self):
        super(CLogLog, self).__init__()

    def _cloglog(self, x):
        return 1 - torch.exp(-torch.exp(-x))

    def forward(self, x):
        return self._cloglog(x)

class ModifiedCLogLog(PyZjrActivation):
    """Modified Complementary Log-Log (cLogLogm) Activation Function"""
    def __init__(self):
        super(ModifiedCLogLog, self).__init__()

    def m_cloglog(self, x):
        return 1 - 2 * torch.exp(-0.7 * torch.exp(-x))

    def forward(self, x):
        return self.m_cloglog(x)

class SechSig(PyZjrActivation):
    """SechSig Activation Function
    Közkurt et al. also proposed a parameter version, which we refer to as the parameter
    SechSig (pSechSig), and also supports parameter free versions below
    """
    def __init__(self, a=None):
        super(SechSig, self).__init__()
        if a is None:
            a = 1.0
        self.a = a

    def _sechsig(self, x):
        sech_x = 1 / torch.cosh(x)
        sigmoid_x = torch.sigmoid(x)
        sech_x_add_a = 1 / torch.cosh(x + self.a)
        if self.a is None:
            return (x + sech_x) * sigmoid_x
        else:
            return (x + self.a * sech_x_add_a) * sigmoid_x

    def forward(self, x):
        return self._sechsig(x)

class TanhSig(PyZjrActivation):
    """TanhSig Activation Function
    The TanhSig is an AF similar to SechSig
    """
    def __init__(self, a=None):
        super(TanhSig, self).__init__()
        if a is None:
            a = 1.0
        self.a = a

    def _tanhsig(self, x):
        tanh_x = torch.tanh(x)
        sigmoid_x = torch.sigmoid(x)
        tanh_x_add_a = torch.tanh(x + self.a)
        if self.a is None:
            return (x + tanh_x) * sigmoid_x
        else:
            return (x + self.a * tanh_x_add_a) * sigmoid_x

    def forward(self, x):
        return self._tanhsig(x)


class SymMSAF(PyZjrActivation):
    """Symmetrical Multistate Activation Function (SymMSAF)
    where a is required to be significantly smaller than 0
    """
    def __init__(self, a=-0.1):
        super(SymMSAF, self).__init__()
        assert a < 0, "Parameter 'a' must be significantly smaller than 0"
        self.a = a

    def _symmsaf(self, x):
        sigmoid_positive = 1 / (1 + torch.exp(-x))
        sigmoid_negative = 1 / (1 + torch.exp(-x - self.a))
        return -1 + sigmoid_positive + sigmoid_negative

    def forward(self, x):
        return self._symmsaf(x)

class RootSig(PyZjrActivation):
    """RootSig Activation Function
    It has some unnamed variants, so we won't implement them here
    """
    def __init__(self, a=1):
        super(RootSig, self).__init__()
        self.a = a

    def _rootsig(self, x):
        return self.a * x / (1 + torch.sqrt(1 + self.a**2 * x**2))

    def forward(self, x):
        return self._rootsig(x)

class SGELU(PyZjrActivation):
    """Symmetrical Gaussian Error Linear Unit (SGELU) Activation Function"""
    def __init__(self, a=1.702):
        super(SGELU, self).__init__()
        self.a = torch.tensor(a)

    def _sgelu(self, x):
        return self.a * x * torch.erf(x / torch.sqrt(torch.tensor(2)))

    def forward(self, x):
        return self._sgelu(x)

class CaLU(PyZjrActivation):
    """Cauchy Linear Unit (CaLU) Activation Function"""
    def __init__(self):
        super(CaLU, self).__init__()

    def _calu(self, x):
        return x * (torch.atan(x) / torch.pi + 0.5)

    def forward(self, x):
        return self._calu(x)

class LaLU(PyZjrActivation):
    """Laplace Linear Unit (LaLU) Activation Function"""
    def __init__(self):
        super(LaLU, self).__init__()

    def _lalu(self, x):
        return torch.where(x >= 0, x * (1 - 0.5 * torch.exp(-x)), 0.5 * x * torch.exp(x))

    def forward(self, x):
        return self._lalu(x)

class CoLU(PyZjrActivation):
    """Collapsing Linear Unit (CoLU) Activation Function"""
    def __init__(self):
        super(CoLU, self).__init__()

    def _colu(self, x):
        return x * (1 / (1 - x * torch.exp(-(x + torch.exp(x)))))

    def forward(self, x):
        return self._colu(x)

class GeneralizedSwish(PyZjrActivation):
    """Generalized Swish Activation Function"""
    def __init__(self):
        super(GeneralizedSwish, self).__init__()

    def generalized_swish(self, x):
        return x * torch.sigmoid(torch.exp(-x))

    def forward(self, x):
        return self.generalized_swish(x)

class ExponentialSwish(PyZjrActivation):
    """Exponential Swish Activation Function"""
    def __init__(self):
        super(ExponentialSwish, self).__init__()

    def exponential_swish(self, x):
        return torch.exp(-x) * torch.sigmoid(x)

    def forward(self, x):
        return self.exponential_swish(x)


class DerivativeSigmoid(PyZjrActivation):
    """Derivative of sigmoid function"""
    def __init__(self):
        super(DerivativeSigmoid, self).__init__()

    def derivative_sigmoid(self, x):
        return torch.exp(-x) * (torch.sigmoid(x)**2)

    def forward(self, x):
        return self.derivative_sigmoid(x)

class Gish(PyZjrActivation):
    """Gish Activation Function"""
    def __init__(self):
        super(Gish, self).__init__()

    def _gish(self, x):
        return x * torch.log(2 - torch.exp(-torch.exp(x)))

    def forward(self, x):
        return self._gish(x)

class Logish(PyZjrActivation):
    """Logish Activation Function"""
    def __init__(self):
        super(Logish, self).__init__()

    def _logish(self, x):
        return x * torch.log(1 + torch.sigmoid(x))

    def forward(self, x):
        return self._logish(x)

class LogLogish(PyZjrActivation):
    """LogLogish Activation Function"""
    def __init__(self):
        super(LogLogish, self).__init__()

    def _loglogish(self, x):
        return x * (1 - torch.exp(-torch.exp(x)))

    def forward(self, x):
        return self._loglogish(x)

class ExpExpish(PyZjrActivation):
    """ExpExpish Activation Function"""
    def __init__(self):
        super(ExpExpish, self).__init__()

    def _expexpish(self, x):
        return x * torch.exp(-torch.exp(-x))

    def forward(self, x):
        return self._expexpish(x)

class SelfArctan(PyZjrActivation):
    """Self Arctan Activation Function"""
    def __init__(self):
        super(SelfArctan, self).__init__()

    def self_arctan(self, x):
        return x * torch.atan(x)

    def forward(self, x):
        return self.self_arctan(x)

class ParametricLogish(PyZjrActivation):
    """Parametric Logish Activation Function"""
    def __init__(self, a=1, b=10):
        super(ParametricLogish, self).__init__()
        self.a = a
        self.b = b

    def parametric_logish(self, x):
        return self.a * x * torch.log(1 + torch.sigmoid(self.b * x))

    def forward(self, x):
        return self.parametric_logish(x)

class Phish(PyZjrActivation):
    """Phish Activation Function"""
    def __init__(self):
        super(Phish, self).__init__()

    def phish(self, x):
        return x * torch.tanh(F.gelu(x))

    def forward(self, x):
        return self.phish(x)

class Suish(PyZjrActivation):
    """Suish Activation Function"""
    def __init__(self):
        super(Suish, self).__init__()

    def suish(self, x):
        return torch.max(x, x * torch.exp(-torch.abs(x)))

    def forward(self, x):
        return self.suish(x)

class TSReLU(PyZjrActivation):
    """Tangent-sigmoid ReLU (TSReLU) Activation Function"""
    def __init__(self):
        super(TSReLU, self).__init__()

    def _tsrelu(self, x):
        return x * torch.tanh(torch.sigmoid(x))

    def forward(self, x):
        return self._tsrelu(x)


class TBSReLU(PyZjrActivation):
    """Tangent-bipolar-sigmoid ReLU (TBSReLU) Activation Function"""
    def __init__(self):
        super(TBSReLU, self).__init__()

    def _tbsrelu(self, x):
        return x * torch.tanh((1 - torch.exp(-x)) / (1 + torch.exp(-x)))

    def forward(self, x):
        return self._tbsrelu(x)

class dSiLU(PyZjrActivation):
    """Derivative of sigmoid-weighted linear unit (dSiLU) Activation Function"""
    def __init__(self):
        super(dSiLU, self).__init__()

    def _dsilu(self, x):
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1 + x * (1 - sigmoid))

    def forward(self, x):
        return self._dsilu(x)

class DoubleSiLU(PyZjrActivation):
    """Double sigmoid-weighted linear unit (DoubleSiLU) Activation Function"""
    def __init__(self):
        super(DoubleSiLU, self).__init__()

    def double_silu(self, x):
        sigmoid1 = 1 / (1 + torch.exp(-x))
        sigmoid2 = 1 / (1 + torch.exp(-x * sigmoid1))
        return x * sigmoid2

    def forward(self, x):
        return self.double_silu(x)

class MSiLU(PyZjrActivation):
    """Modified sigmoid-weighted linear unit (MSiLU) Activation Function"""
    def __init__(self):
        super(MSiLU, self).__init__()

    def m_silu(self, x):
        return x * torch.sigmoid(x) + (torch.exp(-x**2 - 1)) / 4

    def forward(self, x):
        return self.m_silu(x)

class TSiLU(PyZjrActivation):
    """Hyperbolic tangent sigmoid-weighted linear unit (TSiLU)"""
    def __init__(self):
        super(TSiLU, self).__init__()

    def _tsilu(self, x):
        exp_term = torch.exp(x / (1 + torch.exp(-x)))
        return (exp_term - torch.exp(-x / (1 + torch.exp(-x)))) / (exp_term + exp_term)

    def forward(self, x):
        return self._tsilu(x)

class ATSiLU(PyZjrActivation):
    """Arctan sigmoid-weighted linear unit (ATSiLU) Activation Function"""
    def __init__(self):
        super(ATSiLU, self).__init__()

    def _atsilu(self, x):
        return torch.atan(x / (1 + torch.exp(-x)))

    def forward(self, x):
        return self._atsilu(x)

class RectifiedHyperbolicSecant(PyZjrActivation):
    """Rectified Hyperbolic Secant Activation Function"""
    def __init__(self):
        super(RectifiedHyperbolicSecant, self).__init__()

    def rectified_hyperbolic_secant(self, x):
        sech_x = 1 / torch.cosh(x)
        return x * sech_x

    def forward(self, x):
        return self.rectified_hyperbolic_secant(x)

class LiSHT(PyZjrActivation):
    """Linearly Scaled Hyperbolic Tangent (LiSHT) Activation Function"""
    def __init__(self):
        super(LiSHT, self).__init__()

    def _lisht(self, x):
        return x * torch.tanh(x)

    def forward(self, x):
        return self._lisht(x)

class Smish(PyZjrActivation):
    """Smish Activation Function
    Wang, Ren, and Wang recommend a = 1 and b = 1 based on a small
    parameter search
    """
    def __init__(self, a=1, b=1):
        super(Smish, self).__init__()
        self.a = a
        self.b = b

    def _smish(self, x):
        return self.a * x * torch.tanh(torch.log(1 + torch.sigmoid(self.b * x)))

    def forward(self, x):
        return self._smish(x)

class TanhExp(PyZjrActivation):
    """TanhExp Activation Function
    Similarly as the mish is the combination of tanh and softplus
    """
    def __init__(self):
        super(TanhExp, self).__init__()

    def tanh_exp(self, x):
        return x * torch.tanh(torch.exp(x))

    def forward(self, x):
        return self.tanh_exp(x)


class Serf(PyZjrActivation):
    """Serf Activation Function"""
    def __init__(self):
        super(Serf, self).__init__()

    def serf(self, x):
        return x * torch.erf(torch.log(1 + torch.exp(x)))

    def forward(self, x):
        return self.serf(x)

class EANAF(PyZjrActivation):
    """Exponential and Adaptive Non-Asymptotic Function (EANAF) Activation Function
    The EANAF is continuously differentiable. The EANAF is very similar to swish with similar amount of computation
    but Chai et al. found that it performs better than swish and several other activation functions in RetinaNet and
    YOLOv4 architectures on object detection tasks
    """
    def __init__(self):
        super(EANAF, self).__init__()

    def eanaf(self, x):
        return x * torch.exp(x) / (torch.exp(x) + 2)

    def forward(self, x):
        return self.eanaf(x)

class SinSig(PyZjrActivation):
    """Sinusoidal Sigmoid (SinSig) Activation Function"""
    def __init__(self):
        super(SinSig, self).__init__()

    def _sinsig(self, x):
        return x * torch.sin(((torch.pi / 2) * torch.sigmoid(x)))

    def forward(self, x):
        return self._sinsig(x)

class SiELU(PyZjrActivation):
    """Gaussian Error Linear Unit with Sigmoid (SiELU) Activation Function"""
    def __init__(self):
        super(SiELU, self).__init__()

    def _sielu(self, x):
        return x * torch.sigmoid(2 * torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * x**3))

    def forward(self, x):
        return self._sielu(x)

class ShiftedReLU(PyZjrActivation):
    """Shifted ReLU"""
    def __init__(self):
        super(ShiftedReLU, self).__init__()

    def _shiftedreLU(self, x):
        return torch.max(torch.tensor(-1.0), x)

    def forward(self, x):
        return self._shiftedreLU(x)


class SlReLU(PyZjrActivation):
    """A Sloped ReLU (SlReLU) is similar to the LReLU
    Seo, Lee, and Kim recommended a ∈ [1, 10]
    """
    def __init__(self, slope=2.0):
        super(SlReLU, self).__init__()
        self.slope = slope

    def _slrelu(self, x):
        return torch.where(x >= 0, self.slope * x, torch.tensor(0.0))

    def forward(self, x):
        return self._slrelu(x)

class NReLU(PyZjrActivation):
    """A stochastic variant of the ReLU called noisy ReLU (NReLU)"""
    def __init__(self):
        super(NReLU, self).__init__()

    def _nrelu(self, x):
        noise = torch.randn_like(x)
        sigma = x.std(dim=-1, keepdim=True)
        return F.relu(x + sigma * noise)

    def forward(self, x):
        return self._nrelu(x)

class SineReLU(PyZjrActivation):
    """The SineReLU is a ReLU based activation that uses trigonometric functions for negative inputs."""
    def __init__(self, a=1.0):
        super(SineReLU, self).__init__()
        self.a = a

    def _sinerelu(self, x):
        return torch.where(x >= 0, x, self.a * (torch.sin(x) - torch.cos(x)))

    def forward(self, x):
        return self._sinerelu(x)


class Minsin(PyZjrActivation):
    def __init__(self):
        super(Minsin, self).__init__()

    def forward(self, x):
        return torch.minimum(x, torch.sin(x))

class SLU(PyZjrActivation):
    """Softplus Linear Unit (SLU) Activation Function"""
    def __init__(self):
        super(SLU, self).__init__()

    def _slu(self, x):
        softplus_part = 2 * torch.log((torch.exp(x) + 1) / 2)
        return torch.where(x >= 0, x, softplus_part)

    def forward(self, x):
        return self._slu(x)

class ReSP(PyZjrActivation):
    """Rectified Softplus (ReSP) Activation Function
    a: Larger values of a between 1.4 and 2.0 were found to work well
    """
    def __init__(self, a=1.5):
        super(ReSP, self).__init__()
        self.a = a

    def _resp(self, x):
        relu_part = self.a * x + torch.log(torch.tensor(2))
        softplus_part = torch.log(1 + torch.exp(x))
        return torch.where(x >= 0, relu_part, softplus_part)

    def forward(self, x):
        return self._resp(x)

class TRec(PyZjrActivation):
    """Truncated Rectified (TRec) Activation Function"""
    def __init__(self, threshold=1.0):
        super(TRec, self).__init__()
        self.threshold = threshold

    def _trec(self, x):
        return torch.where(x > self.threshold, x, 0)

    def forward(self, x):
        return self._trec(x)

class mReLU(PyZjrActivation):
    """Mirrored Rectified Linear Unit (mReLU) Activation Function"""
    def __init__(self):
        super(mReLU, self).__init__()

    def _mrelu(self, x):
        return torch.where((x >= -1) & (x <= 0), 1 + x, torch.where((x > 0) & (x <= 1), 1 - x, 0))

    def forward(self, x):
        return self._mrelu(x)

class SoftModulusQ(PyZjrActivation):
    """SoftModulusQ Activation Function"""
    def __init__(self):
        super(SoftModulusQ, self).__init__()

    def _softmodulusq(self, x):
        return torch.where(torch.abs(x) >= 1, x**2 * (2 - torch.abs(x)), torch.abs(x))

    def forward(self, x):
        return self._softmodulusq(x)

class SoftModulusT(PyZjrActivation):
    """SoftModulusT Activation Function
    When a = 1, the SoftModulusT becames the LiSHT activation function.
    """
    def __init__(self, a=0.01):
        super(SoftModulusT, self).__init__()
        self.a = a

    def _softmodulust(self, x):
        return x * torch.tanh(x / self.a)

    def forward(self, x):
        return self._softmodulust(x)

class EPLAF(PyZjrActivation):
    """Even Power Linear Activation Function (EPLAF)
    Nasiri and Ghiasi-Shirazi focused on the EPLAF in their work and showed
    that EPLAF with d = 2 performed similarly as the ReLU for some of the tasks
    """
    def __init__(self, d=2):
        super(EPLAF, self).__init__()
        self.d = d

    def _eplaf(self, x):
        return torch.where(x >= 1, x - (1 - 1/self.d)**self.d,
                           torch.where(x < -1, -(1 - 1/self.d)**self.d - x,
                                       (1/self.d) * torch.abs(x)**self.d))

    def forward(self, x):
        return self._eplaf(x)


class DRLU(PyZjrActivation):
    """Delay ReLU Activation Function (DRLU)
    Shan, Li, and Chen also add a constraint a > 0 and they used
    a ∈ {0.06, 0.08, 0.10} in their experiments
    """
    def __init__(self, a=0.06):
        super(DRLU, self).__init__()
        self.a = a

    def _drlu(self, x):
        return torch.where(x - self.a >= 0, x - self.a, torch.tensor(0.0))

    def forward(self, x):
        return self._drlu(x)

class DisReLU(PyZjrActivation):
    """Displaced ReLU Activation Function (DisReLU)
    A Shifted ReLU (see section 3.6.1) is a special case of DisReLU with
    a = 1. The VGG-19 with DisReLUs outperform the ReLU, LReLU, PReLU, and ELU activation functions
    with a statistically significant difference in performance on the CIFAR-10 and CIFAR-100 datasets
    """
    def __init__(self, a=1):
        super(DisReLU, self).__init__()
        self.a = a

    def _disrelu(self, x):
        return torch.where(x + self.a >= 0, x, torch.tensor(-self.a, dtype=x.dtype))

    def forward(self, x):
        return self._disrelu(x)

class FlattedTSwish(PyZjrActivation):
    """Flatted-T Swish Activation Function (FTS)
    the recommended value is T = −0.20. The FTS is identical to a shifted swish for the
    positive x. The FTS was shown to outperform ReLU, LReLU, swish, ELU, and FReLU activation
    functions
    """
    def __init__(self, T=-0.20):
        super(FlattedTSwish, self).__init__()
        self.T = T

    def _flattedtswish(self, x):
        return F.relu(x) * torch.sigmoid(x) + self.T

    def forward(self, x):
        return self._flattedtswish(x)

class ReLUSwish(FlattedTSwish):
    """The special case with T = 0 was proposed independently under the name of ReLU-Swish"""
    def __init__(self):
        super(ReLUSwish, self).__init__(T=0)

class OAF(PyZjrActivation):
    """Optimal Activation Function (OAF)"""
    def __init__(self):
        super(OAF, self).__init__()

    def _oaf(self, x):
        return F.relu(x) + x * torch.sigmoid(x)

    def forward(self, x):
        return self._oaf(x)

class REU(PyZjrActivation):
    """Rectified Exponential Unit (REU) Activation Function"""
    def __init__(self):
        super(REU, self).__init__()

    def _reu(self, x):
        return torch.where(x >= 0, x, x * torch.exp(x))

    def forward(self, x):
        return self._reu(x)

class SigLU(PyZjrActivation):
    """Sigmoid Linear Unit (SigLU) Activation Function"""
    def __init__(self):
        super(SigLU, self).__init__()

    def _siglu(self, x):
        return torch.where(x >= 0, x, (1 - torch.exp(-2 * x)) / (1 + torch.exp(-2 * x)))

    def forward(self, x):
        return self._siglu(x)

class SaRa(PyZjrActivation):
    """Swish and ReLU Activation (SaRa) Function"""
    def __init__(self, a=0.5, b=0.7):
        super(SaRa, self).__init__()
        self.a = a
        self.b = b

    def _sara(self, x):
        return torch.where(x >= 0, x, x / (1 + self.a * torch.exp(-self.b * x)))

    def forward(self, x):
        return self._sara(x)

class Maxsig(PyZjrActivation):
    """Maxsig Activation Function"""
    def __init__(self):
        super(Maxsig, self).__init__()

    def _axsig(self, x):
        return torch.max(x, torch.sigmoid(x))

    def forward(self, x):
        return self._axsig(x)

class ThLU(PyZjrActivation):
    """Tanh linear unit (ThLU)"""
    def __init__(self):
        super(ThLU, self).__init__()

    def _thlu(self, x):
        return torch.where(x >= 0, x,
                           2 / (1 + torch.exp(-x)) - 1)

    def forward(self, x):
        return self._thlu(x)

class DiffELU(PyZjrActivation):
    """Difference ELU (DiffELU)
    The recommended setting is a = 0.3 and b = 0.1
    """
    def __init__(self, a=0.3, b=0.1):
        super(DiffELU, self).__init__()

        if not (0 < a < 1) or not (0 < b < 1):
            raise ValueError("Both 'a' and 'b' should be in the range (0, 1).")

        self.a = a
        self.b = b

    def _diffelu(self, x):
        return torch.where(x >= 0, x,
                           self.a * (x * torch.exp(x) - self.b * torch.exp(self.b * x)))

    def forward(self, x):
        return self._diffelu(x)

class PolyLU(PyZjrActivation):
    """Polynomial linear unit (PolyLU)"""
    def __init__(self):
        super(PolyLU, self).__init__()

    def _polylu(self, x):
        return torch.where(x >= 0, x, 1 / (1 - x) - 1)

    def forward(self, x):
        return self._polylu(x)

class PoLU(PyZjrActivation):
    """Power linear unit (PoLU)
    Li, Ding, and Li used a ∈ {1, 1.5, 2} in their experiments
    """
    def __init__(self, a=1.5):
        super(PoLU, self).__init__()
        self.a = a

    def _poly(self, x):
        return torch.where(x >= 0, x, (1 - x)**(-self.a) - 1)

    def forward(self, x):
        return self._poly(x)

class PFLU(PyZjrActivation):
    """Power function linear unit (PFLU)"""
    def __init__(self):
        super(PFLU, self).__init__()

    def _pflu(self, x):
        return x * (1 + x / torch.sqrt(1 + x**2)) / 2

    def forward(self, x):
        return self._pflu(x)


class FPFLU(PyZjrActivation):
    """Faster power function linear unit (FPFLU)"""
    def __init__(self):
        super(FPFLU, self).__init__()

    def _fplu(self, x):
        return torch.where(x >= 0, x, x + x / (2 * torch.sqrt(1 + x**2)))

    def forward(self, x):
        return self._fplu(x)

class ELiSH(PyZjrActivation):
    """Exponential linear sigmoid squashing (ELiSH)"""
    def __init__(self):
        super(ELiSH, self).__init__()

    def _elish(self, x):
        return torch.where(x >= 0, x / (1 + torch.exp(-x)), (torch.exp(x) - 1) / (1 + torch.exp(-x)))

    def forward(self, x):
        return self._elish(x)

class SQNL(PyZjrActivation):
    def __init__(self):
        super(SQNL, self).__init__()

    def _sqnl(self, x):
        return torch.where(x > 2, 1.0,
                           torch.where((x >= 0) & (x <= 2), x - 0.25 * x**2,
                                       torch.where((x >= -2) & (x < 0), x + 0.25 * x**2, -1.0)))

    def forward(self, x):
        return self._sqnl(x)


class SQLU(PyZjrActivation):
    """Square linear unit (SQLU)"""
    def __init__(self):
        super(SQLU, self).__init__()

    def _sqlu(self, x):
        return torch.where(x > 0, x,
                           torch.where((x >= -2) & (x <= 0), x + 0.25 * x**2, -1.0))

    def forward(self, x):
        return self._sqlu(x)

class Squish(PyZjrActivation):
    """Square swish (squish)"""
    def __init__(self):
        super(Squish, self).__init__()

    def _squish(self, x):
        return torch.where(x > 0, x + (x**2) / 32, torch.where((x >= -2) & (x <= 0), x + 0.5 * x**2, 0))

    def forward(self, x):
        return self._squish(x)

class SqREU(PyZjrActivation):
    """Square REU (SqREU)"""
    def __init__(self):
        super(SqREU, self).__init__()

    def _sqreu(self, x):
        return torch.where(x > 0, x, torch.where((x >= -2) & (x <= 0), x + 0.5 * x**2, 0))

    def forward(self, x):
        return self._sqreu(x)

class SqSoftplus(PyZjrActivation):
    """Square softplus (SqSoftplus)"""
    def __init__(self):
        super(SqSoftplus, self).__init__()

    def _sqsoftplus(self, x):
        return torch.where(x > 0.5, x, torch.where((x >= -0.5) & (x <= 0.5), x + 0.5 * (x + 0.5)**2, 0))

    def forward(self, x):
        return self._sqsoftplus(x)


class LogSQNL(PyZjrActivation):
    """Square logistic sigmoid (LogSQNL)"""
    def __init__(self):
        super(LogSQNL, self).__init__()

    def _logsqnl(self, x):
        return torch.where(x > 2, 1.0,
                           torch.where((x >= 0) & (x <= 2), 0.5 * (x - 0.25 * x**2) + 0.5,
                           torch.where((x >= -2) & (x < 0), 0.5 * (x + 0.25 * x**2) + 0.5, 0)))

    def forward(self, x):
        return self._logsqnl(x)

class ISRLU(PyZjrActivation):
    """Inverse square root linear unit (ISRLU)
    Carlile et al. analysed ISRLU with a = 1 and a = 3
    """
    def __init__(self, a=1):
        super(ISRLU, self).__init__()
        self.a = a

    def _isrlu(self, x):
        return torch.where(x >= 0, x, x / torch.sqrt(1 + self.a * x**2))

    def forward(self, x):
        return self._isrlu(x)

class MEF(PyZjrActivation):
    """Modified Elliott function (MEF)"""
    def __init__(self):
        super(MEF, self).__init__()

    def _mef(self, x):
        return x * (torch.rsqrt(1 + x**2)) + 0.5

    def forward(self, x):
        return self._mef(x)

class SQRT(PyZjrActivation):
    """Square-root-based activation function (SQRT)"""
    def __init__(self):
        super(SQRT, self).__init__()

    def _sqrt(self, x):
        return torch.where(x >= 0, torch.sqrt(x), -torch.sqrt(-x))

    def forward(self, x):
        return self._sqrt(x)

class BentIdentity(PyZjrActivation):
    """Bent identity"""
    def __init__(self):
        super(BentIdentity, self).__init__()

    def _bent_identity(self, x):
        return (torch.sqrt(x**2 + 1) - 1) / 2 + x

    def forward(self, x):
        return self._bent_identity(x)

class Mishra(PyZjrActivation):
    """Mishra activation function"""
    def __init__(self):
        super(Mishra, self).__init__()

    def _mishra(self, x):
        return 0.5 * ((x / (1 + torch.abs(x)))**2) + 0.5 * (x / (1 + torch.abs(x)))

    def forward(self, x):
        return self._mishra(x)

class SahaBora(PyZjrActivation):
    """Saha-Bora activation function (SBAF)
    which were set to k = 0.98 and α = 0.5, where authors
    determined a stable fixed point.
    """
    def __init__(self, alpha=0.5, k=0.98):
        super(SahaBora, self).__init__()
        self.alpha = alpha
        self.k = k

    def _sahabora(self, x):
        return 1 / (1 + self.k * (x ** self.alpha) * ((1 - x) ** (1 - self.alpha)))

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        return self._sahabora(x)

class Logarithmic(PyZjrActivation):
    """Logarithmic activation function """
    def __init__(self):
        super(Logarithmic, self).__init__()

    def _logarithmic(self, x):
        return torch.where(x >= 0, torch.log(x + 1) , -torch.log(-x + 1))

    def forward(self, x):
        return self._logarithmic(x)

class Symexp(PyZjrActivation):
    def __init__(self):
        super(Symexp, self).__init__()

    def _symexp(self, x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def forward(self, x):
        return self._symexp(x)

class PUAF(PyZjrActivation):
    """Polynomial universal activation function (PUAF)
    The PUAF becomes the ReLU with a = 1, b = 0, and c = 0; the logistic
    sigmoid is approximated with a = 0, b = 5, and c = 10; finally, the swish is approximated using a = 1, b = 5, and
    c = 10
    """
    def __init__(self, a=None, b=None, c=None, type='relu'):
        super(PUAF, self).__init__()
        self.a = a
        self.b = b
        self.c = c

        if self.a is None and self.b is None and self.c is None:
            if type == 'relu':
                self.a, self.b, self.c = 1, 0, 0
            elif type == 'sigmoid':
                self.a, self.b, self.c = 0, 5, 10
            elif type == 'swish':
                self.a, self.b, self.c = 1, 5, 10

    def forward(self, x):
        return torch.where(x > self.c, x**self.a, torch.where(x < -self.c, 0, x**self.a *
                                                              ((self.c+x)**self.b / ((self.c+x)**self.b + (self.c-x)**self.b))
                                                              ))

class PSoftplus(PyZjrActivation):
    """Parametric softplus (PSoftplus)"""
    def __init__(self, a=1.5, b=torch.log(torch.tensor(2.0))):
        super(PSoftplus, self).__init__()
        self.a = a
        self.b = b

    def _psoftplus(self, x):
        return self.a * (F.softplus(x) - self.b)

    def forward(self, x):
        return self._psoftplus(x)

class ArandaOrdaz(PyZjrActivation):
    """where a > 0 is a fixed parameter. Essai Ali, Abdel-Raman, and Badry used a = 2 in their work"""
    def __init__(self, a=2):
        super(ArandaOrdaz, self).__init__()
        self.a = a

    def _arandaordaz(self, x):
        return 1 - (1+self.a*torch.exp(x)) ** (-1 / self.a)

    def forward(self, x):
        return self._arandaordaz(x)

class PMAF(PyZjrActivation):
    """Piecewise Mexican-hat activation function (PMAF)
    where a is a fixed parameter — Liu, Zeng, and Wang used a = 4
    """
    def __init__(self, a=4):
        super(PMAF, self).__init__()
        self.a = a

    def _pmaf(self, x):
        head = 2 * torch.pi ** (-1 / 4) / torch.sqrt(torch.tensor(3))
        return torch.where(x < 0, head*(1-(x+self.a))*torch.exp(-(x+self.a)**2 / 2),
                           head*(1-(x-self.a))*torch.exp(-(x-self.a)**2 / 2))

    def forward(self, x):
        return self._pmaf(x)


class PRBF(PyZjrActivation):
    """Piecewise radial basis function (PRBF)
    where a and b are fixed parameters — Liu, Zeng, and Wang used a = 3 and b = 1
    """
    def __init__(self, a=3, b=1):
        super(PRBF, self).__init__()
        self.a = a
        self.b = b

    def _prbf(self, x):
        return torch.where(x >= self.a, torch.exp(-(x-2*self.a)**2 / self.b**2),
                           torch.where((-self.a < x) & (x < self.a), torch.exp(-x**2/self.b**2),
                                       torch.exp(-(x+2*self.a)**2 / self.b**2)))

    def forward(self, x):
        return self._prbf(x)


class MArcsinh(PyZjrActivation):
    """The modified arcsinh (m-arcsinh) AF"""
    def __init__(self):
        super(MArcsinh, self).__init__()

    def _marcsinh(self, z):
        return (1 / 12) * torch.asinh(z) * torch.sqrt(torch.abs(z))

    def forward(self, x):
        return self._marcsinh(x)


class HyperSinh(PyZjrActivation):
    """The hyper-sinh is an AF that uses the sinh and cubic functions"""
    def __init__(self):
        super(HyperSinh, self).__init__()

    def _hypersinh(self, x):
        return torch.where(x > 0, torch.sinh(x) / 3, x.pow(3) / 4)

    def forward(self, x):
        return self._hypersinh(x)

class Arctid(PyZjrActivation):
    """The arctid is an arctan-based AF """
    def __init__(self):
        super(Arctid, self).__init__()

    def _arctid(self, x):
        return torch.atan(x) ** 2 - x

    def forward(self, x):
        return self._arctid(x)

class Sine(PyZjrActivation):
    """Scaled sine with vertical shif
    where a is a fixed parameter; a ∈ {0.2, 0.8, 1.2, 1.8, 4}
    """
    def __init__(self, a=.2):
        super(Sine, self).__init__()
        self.a = a

    def _sine(self, z):
        return 0.5 * torch.sin(self.a * z) + 0.3

    def forward(self, x):
        return self._sine(x)


class Cosine(PyZjrActivation):
    """A cosine activation"""
    def __init__(self):
        super(Cosine, self).__init__()

    def _cosine(self, z):
        return 1 - torch.cos(z)

    def forward(self, x):
        return self._cosine(x)


class Cosid(PyZjrActivation):
    """The cosid is one of the AFs"""
    def __init__(self):
        super(Cosid, self).__init__()

    def _cosid(self, x):
        return torch.cos(x) - x

    def forward(self, x):
        return self._cosid(x)

class Sinp(PyZjrActivation):
    """A parametric AF similar to the cosid
    Chan et al. used a ∈ {1, 1.5, 2}
    """
    def __init__(self, a=1.5):
        super(Sinp, self).__init__()
        self.a = a

    def _sinp(self, x):
        return torch.sin(x) - self.a * x

    def forward(self, x):
        return self._sinp(x)

class GCU(PyZjrActivation):
    """Another cosine-based AF"""
    def __init__(self):
        super(GCU, self).__init__()

    def forward(self, z):
        return z * torch.cos(z)

class ASU(PyZjrActivation):
    """The amplifying sine unit (ASU)"""
    def __init__(self):
        super(ASU, self).__init__()

    def forward(self, z):
        return z * torch.sin(z)


class HcLSH(PyZjrActivation):
    """The hyperbolic cosine linearized squashing function (HcLSH) """
    def __init__(self):
        super(HcLSH, self).__init__()

    def _hclsh(self, x):
        return torch.where(x >= 0, torch.log(x * torch.cosh(x/2) + torch.cosh(x)),
                           torch.log(torch.cosh(x)) + x)

    def forward(self, x):
        return self._hclsh(x)

class Exponential(PyZjrActivation):
    """The exponential was used as an AF"""
    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, x):
        return torch.exp(-x)

class NCU(PyZjrActivation):
    """ Non-monotonic cubic unit (NCU)"""
    def __init__(self):
        super(NCU, self).__init__()

    def forward(self, x):
        return x - x.pow(3)

class Triple(PyZjrActivation):
    """
    Chen et al. tested values of a ∈ {0.1, 0.5, 1, 2} and observed that a = 1 reaches the
    best results
    """
    def __init__(self, a=1):
        super(Triple, self).__init__()
        self.a = a

    def forward(self, x):
        return self.a * x.pow(3)

class SQU(PyZjrActivation):
    """Shifted quadratic unit (SQU)"""
    def __init__(self):
        super(SQU, self).__init__()

    def forward(self, x):
        return x + x.pow(2)

class SCMish(PyZjrActivation):
    """
    where ai is a fixed parameter [425]; Mercioni and Holban used ai = 1. It also has a variant where the parameter ai
    is trainable. Such a variant is called soft clipping learnable mish (SCL-mish). When using the SCL-mish, Mercioni and
    Holban initalized the parameter ai = 0.25
    """
    def __init__(self, ai=1.0, trainable=True):
        super(SCMish, self).__init__()
        if trainable:
            self.ai = nn.Parameter(torch.Tensor([0.25]))
        else:
            self.ai = torch.Tensor([ai])

    def forward(self, z):
        return torch.relu(z) * torch.tanh(F.softplus(self.ai * z))

class TBSReLUL(PyZjrActivation):
    """Tangent-bipolar-sigmoid ReLU learnable (TBSReLUL)
    where a is a trainable parameter. Mercioni, Tat, and Holban used ai = 0.5 as the initial value
    """
    def __init__(self, ai=0.5):
        super(TBSReLUL, self).__init__()
        self.ai = nn.Parameter(torch.Tensor([ai]))

    def forward(self, x):
        tanh_part = torch.tanh(self.ai * (1 - torch.exp(-x)) / (1 + torch.exp(-x)))
        return x * tanh_part

class Swim(PyZjrActivation):
    """The swim is an adaptive variant of the PFLU
    Abdool and Dear used fixed ai = 0.5 in their experiments
    """
    def __init__(self, ai=0.5, trainable=True):
        super(Swim, self).__init__()

        if trainable:
            self.ai = nn.Parameter(torch.Tensor([ai]))
        else:
            self.ai = torch.Tensor([ai])

    def _swim(self, x):
        sqrt_term = torch.sqrt(1 + x**2)
        return x * (1 / (2 * (1 + self.ai * x / sqrt_term)))

    def forward(self, x):
        return self._swim(x)

class SquarePlus(PyZjrActivation):
    """The SquarePlus is very similar to the softplus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(SquarePlus, self).__init__()
        self.epsilon = epsilon

    def _squareplus(self, x):
        return 0.5 * (x + torch.sqrt(x**2 + self.epsilon))

    def forward(self, x):
        return self._squareplus(x)

class StepPlus(PyZjrActivation):
    """As the SquarePlus approximates the ReLU"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(StepPlus, self).__init__()
        self.epsilon = epsilon

    def _stepplus(self, x):
        return 0.5 * (1 + x / torch.sqrt(x**2 + self.epsilon))

    def forward(self, x):
        return self._stepplus(x)

class BipolarPlus(PyZjrActivation):
    """The sign function is smoothed into the BipolarPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(BipolarPlus, self).__init__()
        self.epsilon = epsilon

    def _bipolarplus(self, x):
        return x / torch.sqrt(x**2 + self.epsilon)

    def forward(self, x):
        return self._bipolarplus(x)

class vReLUPlus(PyZjrActivation):
    """The vReLUPlus is a MSRF smoothed variant of the vReLU"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(vReLUPlus, self).__init__()
        self.epsilon = epsilon

    def _vreluplus(self, x):
        return torch.sqrt(x**2 + self.epsilon)

    def forward(self, x):
        return self._vreluplus(x)

class BReLUPlus(PyZjrActivation):
    """The BReLUPlus is a MSRF smoothed variant of the BReLU"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(BReLUPlus, self).__init__()
        self.epsilon = epsilon

    def _breluplus(self, x):
        return 0.5 * (1 + torch.sqrt(x**2 + self.epsilon) - torch.sqrt((x-1)**2 + self.epsilon))

    def forward(self, x):
        return self._breluplus(x)


class HardTanhPlus(PyZjrActivation):
    """Similarly, the smoothed variant of the HardTanh named HardTanhPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(HardTanhPlus, self).__init__()
        self.epsilon = epsilon

    def _hardtanhplus(self, x):
        return 0.5 * (torch.sqrt((x+1)**2 + self.epsilon) - torch.sqrt((x-1)**2 + self.epsilon))

    def forward(self, x):
        return self._hardtanhplus(x)

class SwishPlus(StepPlus):
    """The mollified variant of the swish named SwishPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(SwishPlus, self).__init__(epsilon=epsilon)

    def _swishplus(self, x):
        return x * self._stepplus(x)

    def forward(self, x):
        return self._swishplus(x)

class MishPlus(BipolarPlus):
    """The mollified variant of the swish named MishPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(MishPlus, self).__init__(epsilon=epsilon)

    def _mishplus(self, x):
        return x * self._bipolarplus(self._bipolarplus(x))

    def forward(self, x):
        return self._mishplus(x)


class LogishPlus(StepPlus):
    """The mollified variant of the logish named LogishPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(LogishPlus, self).__init__(epsilon=epsilon)

    def _logishplus(self, x):
        return x * torch.log(1 + self._stepplus(x))

    def forward(self, x):
        return self._logishplus(x)

class SoftsignPlus(PyZjrActivation):
    """The mollified variant of the softsign named SoftsignPlus"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(SoftsignPlus, self).__init__()
        self.epsilon = epsilon

    def _softsignplus(self, x):
        return x / torch.log(1 + torch.sqrt(x**2 + self.epsilon))

    def forward(self, x):
        return self._softsignplus(x)


class SignReLUPlus(PyZjrActivation):
    """Pan et al. provide a mollified version for an approximation of the SignReLU"""
    def __init__(self, epsilon=4 * (torch.log(torch.Tensor([2.0]))**2)):
        super(SignReLUPlus, self).__init__()
        self.epsilon = epsilon

    def _signreluplus(self, x):
        z = torch.sqrt(x**2 + self.epsilon)
        return .5 * (x + z) + (x - z) / 2 * torch.sqrt((1-x)**2 + self.epsilon)

    def forward(self, x):
        return self._signreluplus(x)

if __name__=="__main__":
    x = torch.tensor([[ 0.6328, -0.2803,  0.2340,  0.1001,  0.1168]])
    # x = torch.cat([x, x], dim=-1)
    # x = torch.tensor([[[[0.6328], [-0.2803], [0.2340], [0.1001], [0.1168]],
    #                    [[0.4321], [-0.1234], [0.5678], [0.9876], [0.5432]]]])   # FReLU示例
    activation = SignReLUPlus()

    x = activation(x)
    print("activation:", x)

    # Example usage:
    test = [Sigmoid, Tanh, ReLU, LeakyReLU, Swish, Mish, PReLU, Softmax, RReLU, ReLU6,  ELU, CELU, SELU,
            GELU ,Hardsigmoid, Hardtanh, Hardswish, Hardshrink, Threshold, Softshrink, Softplus, Softmin, LogSoftmax, Softsign
           ,SiLU, Swish, LogSigmoid, Tanhshrink]


    # for i in test:
    #     plot_activation_function(i)


    """
    x = torch.tensor([[ 0.6328, -0.2803,  0.2340,  0.1001,  0.1168]])
    原始数据均用的x, 内部手写实现与torch官方实现进行了比较
    
    Sigmoid activation: tensor([[0.6531, 0.4304, 0.5582, 0.5250, 0.5292]])
    Tanh activation: tensor([[ 0.5600, -0.2732,  0.2298,  0.0998,  0.1163]])
    ReLU activation: tensor([[0.6328, 0.0000, 0.2340, 0.1001, 0.1168]])
    hardsigmoid activation: tensor([[0.6055, 0.4533, 0.5390, 0.5167, 0.5195]])
    ReLU6 activation: tensor([[0.6328, 0.0000, 0.2340, 0.1001, 0.1168]])
    hardtanh activation: tensor([[ 0.6328, -0.2803,  0.2340,  0.1001,  0.1168]])
    Swish/SiLU activation: tensor([[ 0.4133, -0.1206,  0.1306,  0.0526,  0.0618]])
    Hardswish activation: tensor([[ 0.3831, -0.1271,  0.1261,  0.0517,  0.0607]])
    ELU activation: tensor([[ 0.6328, -0.2444,  0.2340,  0.1001,  0.1168]])
    CELU activation: tensor([[ 0.6328, -0.2444,  0.2340,  0.1001,  0.1168]])
    SELU ctivation: tensor([[ 0.6649, -0.4298,  0.2459,  0.1052,  0.1227]])
    
    GLU activation: tensor([[ 0.4133, -0.1206,  0.1306,  0.0526,  0.0618]])      x = torch.cat([x, x], dim=-1) 维度要为偶数
    GELU activation: tensor([[ 0.4661, -0.1092,  0.1386,  0.0540,  0.0638]])
    Hardshrink activation: tensor([[0.6328, 0.0000, 0.0000, 0.0000, 0.0000]])
    LeakyReLU activation: tensor([[ 0.6328, -0.0028,  0.2340,  0.1001,  0.1168]])
    RReLU activation: tensor([[ 0.6328, -0.0808,  0.2340,  0.1001,  0.1168]])   不固定
    
    Mish activation: tensor([[ 0.4969, -0.1430,  0.1576,  0.0632,  0.0744]])  
    PReLu tensor([[ 0.6328, -0.0701,  0.2340,  0.1001,  0.1168]],
       grad_fn=<PreluBackward0>)
    
    nn.Threshold(0, 0.235) Threshold activation: tensor([[0.6328, 0.2350, 0.2340, 0.1001, 0.1168]])
    
    Softsign activation: tensor([[ 0.3876, -0.2189,  0.1896,  0.0910,  0.1046]])
    Tanhshrink activation: tensor([[ 0.0728, -0.0071,  0.0042,  0.0003,  0.0005]])
    Softmin activation: tensor([[0.1196, 0.2981, 0.1782, 0.2037, 0.2004]])
    
    Softmax activation: tensor([[0.3071, 0.1232, 0.2061, 0.1803, 0.1833]])
    LogSigmoid activation: tensor([[-0.4260, -0.8431, -0.5830, -0.6443, -0.6365]])
    LogSoftmax activation: tensor([[-1.1806, -2.0937, -1.5794, -1.7133, -1.6966]])
    Softplus activation: tensor([[1.0588, 0.5628, 0.8170, 0.7444, 0.7533]])
    
    Softshrink activation: tensor([[0.1328, 0.0000, 0.0000, 0.0000, 0.0000]])
    
    FReLU activation: tensor([[[[ 1.7796],
                              [-0.2803],
                              [ 0.2340],
                              [ 0.1001],
                              [ 0.1168]],
                    
                             [[ 1.0228],
                              [ 1.0072],
                              [ 0.5678],
                              [ 0.9876],
                              [ 0.5432]]]], grad_fn=<MaximumBackward0>)

    #-------------------------------------------------------------------------------------------
    Shifted_Scaled_Sigmoid activation: tensor([[6.2224e-06, 6.1098e-06, 6.1730e-06, 6.1565e-06, 6.1585e-06]])
    VariantSigmoid activation:  tensor([[0.4784, 0.3650, 0.4292, 0.4125, 0.4146]])
    ScaledHyperbolicTangent activation: tensor([[ 0.6838, -0.3170,  0.2655,  0.1143,  0.1333]])
    Arctan activation: tensor([[ 0.5642, -0.2733,  0.2299,  0.0998,  0.1163]])
    ArctanGR activation: tensor([[ 0.3305, -0.1601,  0.1347,  0.0584,  0.0681]])
    SigmoidAlgebraic activation: tensor([[0.5992, 0.4443, 0.5482, 0.5229, 0.5264]])
    TripleStateSigmoid activation: tensor([[1.2497, 0.5347, 0.9072, 0.8007, 0.8136]])
    ImprovedLogisticSigmoid activation: tensor([[0.6613, 0.5292, 0.5586, 0.5250, 0.5292]])
    SigLin activation: tensor([[0.7480, 0.3883, 0.5933, 0.5400, 0.5467]])
    PTanh activation: tensor([[ 0.5600, -0.2277,  0.2298,  0.0998,  0.1163]])
    SoftRootSign activation: tensor([[ 0.5619, -0.2927,  0.2246,  0.0984,  0.1145]])
    SoftClipping activation: tensor([[0.5356, 0.3019, 0.4291, 0.3943, 0.3986]])
    SmoothStep activation: tensor([[1.0000, 0.1236, 0.8254, 0.6481, 0.6720]])
    Elliott activation: tensor([[0.6938, 0.3905, 0.5948, 0.5455, 0.5523]])
    SincSigmoid activation: tensor([[0.4321, 0.7220, 0.5607, 0.6044, 0.5990]])
    SigmoidGumbel activation: tensor([[0.3840, 0.1146, 0.2530, 0.2124, 0.2174]])
    NewSigmoid activation: tensor([[ 0.4886, -0.2635,  0.2240,  0.0993,  0.1155]])
    Root2Sigmoid activation: tensor([[ 0.0642, -0.0288,  0.0241,  0.0103,  0.0120]])
    LogLog activation: tensor([[0.5880, 0.2662, 0.4532, 0.4046, 0.4108]])
    CLogLog activation: tensor([[0.4120, 0.7338, 0.5468, 0.5954, 0.5892]])
    ModifiedCLogLog activation: tensor([[-0.3790,  0.2081, -0.1493, -0.0617, -0.0728]])
    SechSig activation: tensor([[0.6591, 0.2182, 0.4303, 0.3672, 0.3747]])
    TanhSig activation: tensor([[1.0184, 0.1448, 0.6016, 0.4728, 0.4886]])
    SymMSAF activation: tensor([[ 0.2833, -0.1636,  0.0917,  0.0250,  0.0334]])
    RootSig activation: tensor([[ 0.2898, -0.1375,  0.1154,  0.0499,  0.0582]])
    SGELU activation: tensor([[0.5096, 0.1053, 0.0737, 0.0136, 0.0185]])
    CaLU activation: tensor([[ 0.4300, -0.1158,  0.1341,  0.0532,  0.0627]])
    LaLU activation: tensor([[ 0.4648, -0.1059,  0.1414,  0.0548,  0.0648]])
    CoLU activation: tensor([[ 0.6669, -0.2387,  0.2469,  0.1032,  0.1209]])
    GeneralizedSwish activation: tensor([[ 0.3985, -0.2214,  0.1610,  0.0713,  0.0828]])
    ExponentialSwish activation: tensor([[0.3469, 0.5696, 0.4418, 0.4750, 0.4708]])
    DerivativeSigmoid activation: tensor([[0.2266, 0.2452, 0.2466, 0.2494, 0.2491]])
    Gish activation: tensor([[ 0.3886, -0.1192,  0.1265,  0.0513,  0.0602]])
    Logish activation: tensor([[ 0.3181, -0.1003,  0.1038,  0.0422,  0.0496]])
    LogLogish activation: tensor([[ 0.5365, -0.1486,  0.1679,  0.0670,  0.0788]])
    ExpExpish activation: tensor([[ 0.3721, -0.0746,  0.1061,  0.0405,  0.0480]])
    SelfArctan activation: tensor([[0.3570, 0.0766, 0.0538, 0.0100, 0.0136]])
    ParametricLogish activation: tensor([[ 0.4381, -0.0156,  0.1517,  0.0549,  0.0662]])
    Phish activation: tensor([[0.2753, 0.0305, 0.0322, 0.0054, 0.0074]])
    Suish activation: tensor([[ 0.6328, -0.2118,  0.2340,  0.1001,  0.1168]])
    TSReLU activation: tensor([[ 0.3631, -0.1137,  0.1186,  0.0482,  0.0566]])
    TBSReLU activation: tensor([[0.1880, 0.0388, 0.0271, 0.0050, 0.0068]])
    dSiLU activation: tensor([[0.7965, 0.3617, 0.6159, 0.5500, 0.5583]])
    DoubleSiLU activation: tensor([[ 0.3809, -0.1317,  0.1246,  0.0514,  0.0602]])
    MSiLU activation: tensor([[ 0.4749, -0.0356,  0.2177,  0.1436,  0.1525]])
    TSiLU activation: tensor([[ 0.2812, -0.1364,  0.1150,  0.0499,  0.0581]])
    ATSiLU activation: tensor([[ 0.3919, -0.1201,  0.1299,  0.0525,  0.0617]])
    RectifiedHyperbolicSecant activation: tensor([[ 0.5243, -0.2696,  0.2277,  0.0996,  0.1160]])
    LiSHT activation: tensor([[0.3544, 0.0766, 0.0538, 0.0100, 0.0136]])
    Smish activation: tensor([[ 0.2938, -0.0963,  0.0975,  0.0399,  0.0468]])
    TanhExp activation: tensor([[ 0.6042, -0.1790,  0.1994,  0.0803,  0.0945]])
    Serf activation: tensor([[ 0.5478, -0.1609,  0.1760,  0.0708,  0.0833]])
    SinSig activation: tensor([[ 0.5412, -0.1754,  0.1799,  0.0735,  0.0863]])
    SiELU activation: tensor([[ 0.4661, -0.1092,  0.1386,  0.0540,  0.0638]])
    ShiftedReLU activation: tensor([[ 0.6328, -0.2803,  0.2340,  0.1001,  0.1168]])
    SlReLU activation: tensor([[1.2656, 0.0000, 0.4680, 0.2002, 0.2336]])
    NReLU activation: tensor([[0.4690, 0.0000, 0.0141, 0.0000, 0.1639]])
    SineReLU activation: tensor([[ 0.6328, -1.2376,  0.2340,  0.1001,  0.1168]])
    Minsin activation: tensor([[ 0.5914, -0.2803,  0.2319,  0.0999,  0.1165]])
    SLU activation: tensor([[ 0.6328, -0.2607,  0.2340,  0.1001,  0.1168]])
    ReSP activation: tensor([[1.6423, 0.5628, 1.0441, 0.8433, 0.8683]])
    TRec activation: tensor([[0., 0., 0., 0., 0.]])
    mReLU activation: tensor([[0.3672, 0.7197, 0.7660, 0.8999, 0.8832]])
    SoftModulusQ activation: tensor([[0.6328, 0.2803, 0.2340, 0.1001, 0.1168]])
    SoftModulusT activation: tensor([[0.6328, 0.2803, 0.2340, 0.1001, 0.1168]])
    EPLAF activation: tensor([[0.2002, 0.0393, 0.0274, 0.0050, 0.0068]])
    DRLU activation: tensor([[0.5728, 0.0000, 0.1740, 0.0401, 0.0568]])
    DisReLU activation: tensor([[ 0.6328, -0.2803,  0.2340,  0.1001,  0.1168]])
    FlattedTSwish activation: tensor([[ 0.2133, -0.2000, -0.0694, -0.1474, -0.1382]])
    ReLUSwish activation: tensor([[0.4133, 0.0000, 0.1306, 0.0526, 0.0618]])
    OAF activation: tensor([[ 1.0461, -0.1206,  0.3646,  0.1527,  0.1786]])
    REU activation: tensor([[ 0.6328, -0.2118,  0.2340,  0.1001,  0.1168]])
    SigLU activation: tensor([[ 0.6328, -0.2732,  0.2340,  0.1001,  0.1168]])
    SaRa activation: tensor([[ 0.6328, -0.1743,  0.2340,  0.1001,  0.1168]])
    Maxsig activation: tensor([[0.6531, 0.4304, 0.5582, 0.5250, 0.5292]])
    ThLU activation: tensor([[ 0.6328, -0.1392,  0.2340,  0.1001,  0.1168]])
    DiffELU activation: tensor([[ 0.6328, -0.0927,  0.2340,  0.1001,  0.1168]])
    PolyLU activation: tensor([[ 0.6328, -0.2189,  0.2340,  0.1001,  0.1168]])
    PoLU activation: tensor([[ 0.6328, -0.3097,  0.2340,  0.1001,  0.1168]])
    PFLU activation: tensor([[ 0.4856, -0.1023,  0.1437,  0.0550,  0.0652]])
    FPFLU activation: tensor([[ 0.6328, -0.4152,  0.2340,  0.1001,  0.1168]])
    ELiSH activation: tensor([[ 0.4133, -0.1052,  0.1306,  0.0526,  0.0618]])
    SQNL activation: tensor([[ 0.5327, -0.2607,  0.2203,  0.0976,  0.1134]])
    SQLU activation: tensor([[ 0.6328, -0.2607,  0.2340,  0.1001,  0.1168]])
    Squish activation: tensor([[ 0.6453, -0.2410,  0.2357,  0.1004,  0.1172]])
    SqREU activation: tensor([[ 0.6328, -0.2410,  0.2340,  0.1001,  0.1168]])
    SqSoftplus activation: tensor([[ 0.6328, -0.2562,  0.5034,  0.2802,  0.3070]])
    LogSQNL activation: tensor([[0.7663, 0.3697, 0.6102, 0.5488, 0.5567]])
    ISRLU activation: tensor([[ 0.6328, -0.2699,  0.2340,  0.1001,  0.1168]])
    MEF activation: tensor([[1.0347, 0.2301, 0.7278, 0.5996, 0.6160]])
    SQRT activation: tensor([[ 0.7955, -0.5294,  0.4837,  0.3164,  0.3418]])
    BentIdentity activation: tensor([[ 0.7245, -0.2610,  0.2475,  0.1026,  0.1202]])
    Mishra activation: tensor([[ 0.2689, -0.0855,  0.1128,  0.0496,  0.0578]])
    SahaBora activation: tensor([[0.6792, 1.0000, 0.7068, 0.7727, 0.7606]])
    Logarithmic activation: tensor([[ 0.4903, -0.2471,  0.2103,  0.0954,  0.1105]])
    Symexp activation: tensor([[ 0.8829, -0.3235,  0.2636,  0.1053,  0.1239]])
    PUAF activation: tensor([[0.6328, 0.0000, 0.2340, 0.1001, 0.1168]])
    PSoftplus activation: tensor([[ 0.5485, -0.1955,  0.1857,  0.0770,  0.0902]])
    ArandaOrdaz activation: tensor([[0.5419, 0.3689, 0.4675, 0.4419, 0.4451]])
    PMAF activation: tensor([[ 0.0131, -0.0023,  0.0034,  0.0021,  0.0023]])
    PRBF activation: tensor([[0.6700, 0.9244, 0.9467, 0.9900, 0.9865]])
    MArcsinh activation: tensor([[ 0.0396, -0.0122,  0.0093,  0.0026,  0.0033]])
    HyperSinh activation: tensor([[ 0.2253, -0.0055,  0.0787,  0.0334,  0.0390]])
    Arctid activation: tensor([[-0.3145,  0.3550, -0.1812, -0.0901, -0.1033]])
    Sine activation: tensor([[0.3631, 0.2720, 0.3234, 0.3100, 0.3117]])
    Cosine activation: tensor([[0.1936, 0.0390, 0.0273, 0.0050, 0.0068]])
    Cosid activation: tensor([[0.1736, 1.2413, 0.7387, 0.8949, 0.8764]])
    Sinp activation: tensor([[-0.3578,  0.1438, -0.1191, -0.0502, -0.0587]])
    GCU activation: tensor([[ 0.5103, -0.2694,  0.2276,  0.0996,  0.1160]])
    ASU activation: tensor([[0.3742, 0.0775, 0.0543, 0.0100, 0.0136]])
    HcLSH activation: tensor([[ 0.6269, -0.2415,  0.2336,  0.1001,  0.1167]])
    Exponential activation: tensor([[0.5311, 1.3235, 0.7914, 0.9047, 0.8898]])
    NCU activation: tensor([[ 0.3794, -0.2583,  0.2212,  0.0991,  0.1152]])
    Triple activation: tensor([[ 0.2534, -0.0220,  0.0128,  0.0010,  0.0016]])
    SQU activation: tensor([[ 1.0332, -0.2017,  0.2888,  0.1101,  0.1304]])
    SCMish activation: tensor([[0.4113, 0.0000, 0.1448, 0.0609, 0.0712]], grad_fn=<MulBackward0>)
    TBSReLUL activation: tensor([[0.0961, 0.0195, 0.0136, 0.0025, 0.0034]], grad_fn=<MulBackward0>)
    Swim activation: tensor([[ 0.2497, -0.1620,  0.1050,  0.0477,  0.0552]], grad_fn=<MulBackward0>)
    SquarePlus activation: tensor([[1.0783, 0.5670, 0.8200, 0.7450, 0.7540]])
    StepPlus activation: tensor([[0.7076, 0.4009, 0.5832, 0.5360, 0.5420]])
    BipolarPlus activation: tensor([[ 0.4153, -0.1982,  0.1664,  0.0720,  0.0840]])
    vReLUPlus activation: tensor([[1.5239, 1.4143, 1.4059, 1.3899, 1.3912]])
    BReLUPlus activation: tensor([[0.5449, 0.2636, 0.4110, 0.3686, 0.3737]])
    HardTanhPlus activation: tensor([[ 0.3539, -0.1625,  0.1361,  0.0585,  0.0682]])
    SwishPlus activation: tensor([[ 0.4478, -0.1124,  0.1365,  0.0537,  0.0633]])
    MishPlus activation: tensor([[0.1816, 0.0397, 0.0279, 0.0052, 0.0071]])
    LogishPlus activation: tensor([[ 0.3386, -0.0945,  0.1075,  0.0430,  0.0506]])
    SoftsignPlus activation: tensor([[ 0.6835, -0.3180,  0.2665,  0.1149,  0.1340]])
    SignReLUPlus activation: tensor([[ 0.4394, -1.0319, -0.1081, -0.3209, -0.2934]])
    """
