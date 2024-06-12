import math

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn


# from models.attention import SEBasicBlock, Fusion_block
# from attention import SEBasicBlock, Fusion_block


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expansion=2, downsample=None):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        # if sigmoid:
        #     self.sigmoid = nn.Sigmoid()
        # else:
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECALayerBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(ECALayerBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=7, stride=1,
                                       padding=7 // 2, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(inplanes, inplanes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(inplanes)
        self.se = SELayer(inplanes, reduction)
        self.ECA = ECALayer(inplanes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ECA(out)

        out = residual + out
        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, inplanes, stride)
        self.group_conv3x3 = nn.Conv1d(inplanes, inplanes, kernel_size=3, stride=1,
                                       padding=1, groups=inplanes // 16, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(inplanes, reduction)

    def forward(self, x):
        residual = x
        out = self.group_conv3x3(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.se(out)
        out = residual + out
        return out


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class Scale_Fusion(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_Fusion, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2

class Scale_split(nn.Module):
    def __init__(self, x1_channels, x2_channels, ):
        super(Scale_split, self).__init__()

        self.up = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(size=127, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )

    def forward(self, x1, x2):
        up1 = self.up(x1)
        down1 = self.down(x2)
        feat1 = down1 + x1
        feat2 = up1 + x2
        x1 = self.layer1(feat1)
        x2 = self.layer2(feat2)
        return x1, x2


class Scale_Fusion2(nn.Module):
    def __init__(self, x1_channels, x2_channels, x3_channels):
        super(Scale_Fusion2, self).__init__()

        self.up1_2 = nn.Sequential(
            nn.Conv1d(x1_channels, x2_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.ReLU(),
        )

        self.up1_3 = nn.Sequential(
            nn.Conv1d(x1_channels, x3_channels, kernel_size=3, stride=2, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.up2_3 = nn.Sequential(
            nn.Conv1d(x2_channels, x3_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x3_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.down2_1 = nn.Sequential(
            nn.Conv1d(x2_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv1d(x3_channels, x1_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x1_channels),
            nn.Upsample(scale_factor=4.0, mode='nearest'),
        )

        self.down3_2 = nn.Sequential(
            nn.Conv1d(x3_channels, x2_channels, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(x2_channels),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
            SEBasicBlock(x1_channels),
            ECALayerBlock(x1_channels),
            BasicBlock(x1_channels, x1_channels, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
            SEBasicBlock(x2_channels),
            ECALayerBlock(x2_channels),
            BasicBlock(x2_channels, x2_channels, expansion=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(x3_channels, x3_channels, expansion=1),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            BasicBlock(x3_channels, x3_channels, expansion=1),
            SEBasicBlock(x3_channels),
            ECALayerBlock(x3_channels),
            BasicBlock(x3_channels, x3_channels, expansion=1),
        )

    def forward(self, x1, x2, x3):
        # up1_2 = self.up1_2(x1)
        # up1_3 = self.up1_3(x1)
        # up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        # x1 = down2_1 + down3_1 + x1
        # x2 = up1_2 + down3_2 + x2
        # x3 = up1_3 + up2_3 + x3
        # x1 = self.layer1(x1)
        # x2 = self.layer2(x2)
        # x3 = self.layer3(x3)

        up1_2 = self.up1_2(x1)
        up1_3 = self.up1_3(x1)
        up2_3 = self.up2_3(x2)
        # down2_1 = self.down2_1(x2)
        # down3_1 = self.down3_1(x3)
        # down3_2 = self.down3_2(x3)
        x1 = x1
        x2 = up1_2 + x2
        x3 = up1_3 + up2_3 + x3
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        return x1, x2, x3

# if __name__ == '__main__':
#     a = torch.randn(8, 12, 1000)
#     out = model(a)
