"""
Paper address: https://arxiv.org/pdf/2307.08388v2.pdf
According to https://github.com/YaoleiQi/DSCNet Newly submitted S3_DSConv_pro.py file was rewritten
because it used einops, so I changed everything involved to Torch handwriting implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=9, extend_scope=1.0,
                 morph=0, if_offset=True, device='cuda' if torch.cuda.is_available() else 'cpu',):
        """
        :param in_channels: 输入通道的数量
        :param out_channels: 输出通道的数量
        :param kernel_size: 内核的大小
        :param extend_scope: 要扩大的范围。此方法的默认值为 1
        :param morph: 卷积核的形态主要沿 x轴（0）和 y轴（1）分为两类
        :param if_offset: 是否需要变形，如果为False，则为标准卷积核。
        :param device: 设备，'cpu' 或 'cuda'
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        if morph in (0, 1):
            self.morph = morph
        else:
            raise ValueError("[DSConv]:Only two types of morph: 0 (x-axis) or 1 (y-axis)")
        self.if_offset = if_offset
        self.device = device

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input):
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )
        output = None
        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output

def _offset_to_center(kernel_size, width, height, device):
    """
    将偏移量转换为中心坐标
    :param kernel_size: 卷积核尺寸
    :param width: 输入图像的宽度
    :param height: 输入图像的高度
    :param device: 设备类型
    :return: _x_center: 转换后的 x 中心坐标，形状为 [kernel_size, width, height]
             _y_center: 转换后的 y 中心坐标，形状为 [kernel_size, width, height]
    """
    y_center = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center = y_center.unsqueeze(0).unsqueeze(2)
    _y_center = y_center.repeat(kernel_size, 1, height)
    x_center = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_expanded = x_center.unsqueeze(0)
    _x_center = x_center_expanded.repeat(kernel_size, width, 1)
    return _x_center, _y_center

def get_coordinate_map_2D(offset, morph, extend_scope=1.0, device="cuda"):
    """
    :param offset: 通过形状为[B，2*K，W，H]的网络预测偏移量。这里的K是指内核大小。
    :param morph: 卷积核的形态主要沿 x轴（0）和 y轴（1）分为两类
    :param extend_scope: 要扩大的范围。此方法的默认值为1
    :param device: 设备，'cpu' 或 'cuda'
    :return: y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
             x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """
    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)
    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)
    x_center, y_center = _offset_to_center(kernel_size,width,height,device)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_spread_expanded = y_spread_.unsqueeze(-1).unsqueeze(-1)
        y_grid_ = y_spread_expanded.repeat(1, width, height)
        x_spread_expanded = x_spread_.unsqueeze(-1).unsqueeze(-1)
        x_grid_ = x_spread_expanded.repeat(1, width, height)

        y_new_ = y_center + y_grid_
        x_new_ = x_center + x_grid_

        y_new_ = y_new_.repeat(batch_size, 1, 1, 1)
        x_new_ = x_new_.repeat(batch_size, 1, 1, 1)

        y_offset_ = y_offset_.permute(1, 0, 2, 3)
        y_offset_new_ = y_offset_.detach().clone()

        # 中心位置保持不变，其余位置开始摆动
        # 这部分很简单。其主要思想是“偏移是一个迭代过程”

        y_offset_new_[center] = 0
        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                    y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                    y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = y_offset_new_.permute(1, 0, 2, 3)
        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = y_new_.permute(0, 2, 1, 3).reshape(batch_size, -1, height)
        x_coordinate_map = x_new_.permute(0, 2, 1, 3).reshape(batch_size, -1, height)
        return y_coordinate_map, x_coordinate_map
    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_spread_expanded = y_spread_.unsqueeze(-1).unsqueeze(-1)
        y_grid_ = y_spread_expanded.repeat(1, width, height)
        x_spread_expanded = x_spread_.unsqueeze(-1).unsqueeze(-1)
        x_grid_ = x_spread_expanded.repeat(1, width, height)

        y_new_ = x_center + y_grid_
        x_new_ = x_center + x_grid_

        y_new_ = y_new_.repeat(batch_size, 1, 1, 1)
        x_new_ = x_new_.repeat(batch_size, 1, 1, 1)

        x_offset_ = x_offset_.permute(1, 0, 2, 3)
        x_offset_new_ = x_offset_.detach().clone()

        x_offset_new_[center] = 0
        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                    x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                    x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = x_offset_new_.permute(1, 0, 2, 3)
        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))
        y_coordinate_map = y_new_.permute(0, 2, 3, 1).reshape(batch_size, width, -1)
        x_coordinate_map = x_new_.permute(0, 2, 3, 1).reshape(batch_size, width, -1)

        return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(input_feature, y_coordinate_map, x_coordinate_map, interpolate_mode="bilinear",):
    """
    根据坐标映射，对 DSCNet 的特征进行插值，插值后的特征，形状为 [B, C, K_H * H, K_W * W]
    :param input_feature: 需要插值的特征，形状为 [B, C, H, W]
    :param y_coordinate_map: 沿 y 轴的坐标映射，形状为 [B, K_H * H, K_W * W]
    :param x_coordinate_map: 沿 x 轴的坐标映射，形状为 [B, K_H * H, K_W * W]
    :param interpolate_mode: F.grid_sample 的 'mode' 参数，可以是 'bilinear' 或 'bicubic'。默认为 'bilinear'。
    """
    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = F.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(coordinate_map, origin, target=(-1, 1)):
    """
    将坐标映射尺度进行缩放
    :param coordinate_map: 坐标映射张量
    :param origin: 原始尺度范围，e.g.[coordinate_map.min(), coordinate_map.max()]
    :param target: 目标尺度范围，默认为 (-1, 1)
    """
    min, max = origin
    a, b = target
    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)
    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = scale_factor * (coordinate_map_scaled - min)  # 转成[0, max-min]的范围，缩放到 [0, (b-a)]
    coordinate_map_scaled = coordinate_map_scaled + a  # 平移到目标尺度范围 [a, b]

    return coordinate_map_scaled


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.tensor([[[[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9]]],

                      [[[1.0, 1.1, 1.2],
                        [1.3, 1.4, 1.5],
                        [1.6, 1.7, 1.8]]],

                      [[[2.0, 2.1, 2.2],
                        [2.3, 2.4, 2.5],
                        [2.6, 2.7, 2.8]]],

                      [[[3.0, 3.1, 3.2],
                        [3.3, 3.4, 3.5],
                        [3.6, 3.7, 3.8]]]])

    print("输入形状:", A.shape)  # torch.Size([4, 1, 3, 3])
    conv0 = DSConv(
        in_channels=1,
        out_channels=10,
        kernel_size=15,
        extend_scope=1,
        morph=0,
        if_offset=True)
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out = conv0(A)
    print("输出形状:", out.shape)  # torch.Size([4, 10, 3, 3])