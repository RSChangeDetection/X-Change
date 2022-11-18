import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, group_num=16):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_num, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_num, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, straight_up=False, change_agg=False, factor=2,
                 group_channel=32):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv_1 = nn.Conv2d(in_channels, in_channels // factor, kernel_size=1, stride=1, bias=False)
            self.conv_2 = nn.Conv2d(in_channels // factor, out_channels, kernel_size=1, stride=1, bias=False)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels // factor, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // factor, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // factor, out_channels)
        if straight_up:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.change_agg = change_agg

        self.norm1 = nn.GroupNorm(in_channels // factor // group_channel, in_channels // factor)
        self.norm2 = nn.GroupNorm(in_channels // factor // group_channel, in_channels // factor)

        self.SA1 = SpatialAttention(3)
        self.SA2 = SpatialAttention(3)

    def forward(self, x1, x2=None):
        if self.bilinear:
            x1 = self.conv_1(x1)
        x1 = self.up(x1)
        if x2 != None:
            if self.bilinear:
                x2 = self.conv_2(x2)
            if self.change_agg == True:
                x1 = x1 + x1 * self.SA1(x1)
                x2 = x2 + x2 * self.SA2(x2)
            x = self.norm1(x2) + self.norm2(x1)
        else:
            return x1
        return self.conv(x)


class Change_Head(nn.Module):
    def __init__(self, inchannel, group_channel=32, type='cat'):
        super(Change_Head, self).__init__()
        self.norm_a = nn.GroupNorm(inchannel // group_channel, inchannel)
        self.norm_b = nn.GroupNorm(inchannel // group_channel, inchannel)
        if type == 'cat':
            self.reduce = nn.Conv2d(inchannel * 2, inchannel, kernel_size=1, stride=1)
        self.type = type

    def forward(self, before, after):
        if self.type == 'sub':
            differ = self.norm_a(after) - self.norm_b(before)
        elif self.type == 'cat':
            differ = self.reduce(torch.cat([self.norm_a(after), self.norm_b(before)], dim=1))
        return differ


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
