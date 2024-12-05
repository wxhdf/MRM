import math

import torch
from torch import nn
import torch.nn.functional as F
import torch
import torch.nn as nn
# models.
from models.attention import BasicBlock, SEBasicBlock, ECALayerBlock, LayerNorm, Scale_Fusion, Scale_Fusion2


class BaseScale(nn.Module):
    def __init__(self, input_channels, mid_channels, stride=1):
        super(BaseScale, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=3, stride=stride, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        filter_sizes = [3, 13, 23, 33]
        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[0],
                               stride=stride, bias=False, padding=(filter_sizes[0] // 2))
        self.conv2 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[1],
                               stride=stride, bias=False, padding=(filter_sizes[1] // 2))
        self.conv3 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[2],
                               stride=stride, bias=False, padding=(filter_sizes[2] // 2))
        self.conv4 = nn.Conv1d(input_channels, mid_channels, kernel_size=filter_sizes[3],
                               stride=stride, bias=False, padding=(filter_sizes[3] // 2))
        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.dro = nn.Dropout(0.2)
        self.max = nn.MaxPool1d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_concat = torch.mean(torch.stack([x1, x2, x3, x4], 2), 2)
        x_concat = self.dro(self.max(self.relu(self.bn(x_concat))))
        x5 = self.conv_block(x)
        x = x5 + x_concat
        return x


class BaseBlock(nn.Module):
    def __init__(self, input_channels=12, mid_channels=16, stride=1):
        super(BaseBlock, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, mid_channels, kernel_size=11, stride=1, bias=False, padding=11 // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=7, stride=1, bias=False, padding=7 // 2),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, mid_channels * 4, kernel_size=5, stride=1, bias=False, padding=5 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(mid_channels * 4, mid_channels * 4, kernel_size=3, stride=1, bias=False, padding=3 // 2),
            nn.BatchNorm1d(mid_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.BaseScale1 = BaseScale(mid_channels * 4, mid_channels * 4, stride=1)
        self.BaseScale2 = BaseScale(mid_channels * 4, mid_channels * 8, stride=2)
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            # ECALayerBlock(mid_channels * 4),
            # SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            # ECALayerBlock(mid_channels * 4),
            # SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x1 = self.BaseScale1(x)
        x2 = self.BaseScale2(x)
        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        return x1, x2


class Fusion_block(nn.Module):
    def __init__(self, input_channels, kernel_size=7, c=1):
        super(Fusion_block, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(input_channels * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(input_channels * 3, eps=1e-6, data_format="channels_first")
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 8, 1, bias=False),
            nn.Conv1d(input_channels // 8, input_channels, 1, bias=False),
            nn.BatchNorm1d(input_channels),
        )
        self.conv1 = nn.Conv1d(input_channels, input_channels, kernel_size=1, )
        self.gelu = nn.GELU()
        self.fc = nn.Linear(372 // c, 250 // c)

        self.residual = BasicBlock(input_channels * 3, input_channels)
        self.residual2 = BasicBlock(input_channels * 2, input_channels)
        self.drop_path = nn.Identity()
        downsample = nn.Sequential(
            nn.Conv1d(input_channels * 2, input_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(input_channels)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(input_channels * 2, input_channels, expansion=1, downsample=downsample),
            ECALayerBlock(input_channels),
            SEBasicBlock(input_channels),
            BasicBlock(input_channels, input_channels, expansion=1),
        )

    def forward(self, l):
        max_result = self.maxpool(l)
        avg_result = self.avgpool(l)
        max_out = self.conv_block3(max_result)
        avg_out = self.conv_block3(avg_result)
        l1 = self.sigmoid(max_out + avg_out) * l

        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l2 = self.conv_block2(result)
        l2 = self.sigmoid(l2) * l

        fuse = torch.cat([l1, l2], 1)
        fuse = self.norm1(fuse)
        fuse = self.layer1(fuse)
        fuse = self.drop_path(fuse)
        return fuse


class Dui1(nn.Module):

    def __init__(self, num_classes=5, input_channels=12, mid_channels=16, mu2=1, sigma2=0.1):
        super(Dui1, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Fusion_block1 = Fusion_block(64, c=1)
        self.Fusion_block2 = Fusion_block(128, c=1)
        self.Fusion_block3 = Fusion_block(512, c=1)
        self.conv_norm1 = nn.LayerNorm(64, eps=1e-6)
        self.conv_norm2 = nn.LayerNorm(128, eps=1e-6)
        self.conv_norm3 = nn.LayerNorm(128 * 4, eps=1e-6)
        self.BaseBlock = BaseBlock(input_channels=input_channels)
        self.layer2 = nn.Sequential(
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
            ECALayerBlock(mid_channels * 8),
            SEBasicBlock(mid_channels * 8),
            BasicBlock(mid_channels * 8, mid_channels * 8, expansion=1),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            BasicBlock(mid_channels * 4, mid_channels * 4, expansion=1),
            ECALayerBlock(mid_channels * 4),
            SEBasicBlock(mid_channels * 4),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.down1 = nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        self.down2 = nn.Conv1d(256, 128, kernel_size=1, stride=1, bias=False)
        self.scale_f = Scale_Fusion(128, 256)
        self.scale_f2 = Scale_Fusion2(128, 256, 512)
        self.up1_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=2, bias=False, padding=3 // 2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(64, num_classes)
        self.fc2 = nn.Linear(128, num_classes)
        self.fc3 = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2 = self.BaseBlock(x)
        # feat2 = self.scale_f(x1, feat1)
        x1 = self.Fusion_block1(x1)
        x2 = self.Fusion_block2(x2)
        feat1 = x1.view(x1.size(0), -1)
        feat2 = x2.view(x2.size(0), -1)
        x1 = self.avg_pool(x1).squeeze()
        x2 = self.avg_pool(x2).squeeze()
        out1 = self.fc1(self.conv_norm1(x1))
        out2 = self.fc2(self.conv_norm2(x2))
        return feat1, out1, feat2, out2


if __name__ == '__main__':
    model = Dui1()
    input1 = torch.randn(8, 12, 1000)
    out = model(input1)
