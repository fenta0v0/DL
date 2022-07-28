# test
import torch
from torch import nn


# basic_block
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.identidy = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.conv(x)
        out1 = self.identidy(x)
        out = out1 + out
        out = self.relu(out)
        return out


class NET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = double_conv(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv2 = double_conv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv3 = double_conv(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv4 = double_conv(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv5 = double_conv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = double_conv(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        d1 = self.down1(c1)
        c2 = self.conv2(d1)
        d2 = self.down2(c2)
        c3 = self.conv3(d2)
        d3 = self.down3(c3)
        c4 = self.conv4(d3)
        d4 = self.down4(c4)
        c5 = self.conv5(d4)

        up1 = self.up1(c5)
        concat4 = torch.cat([c4, up1], dim=1)
        c6 = self.conv6(concat4)
        up2 = self.up2(c6)
        concat3 = torch.cat([up2, c3], dim=1)
        c7 = self.conv7(concat3)
        up3 = self.up3(c7)
        concat2 = torch.cat([c2, up3], dim=1)
        c8 = self.conv8(concat2)
        up4 = self.up4(c8)
        concat1 = torch.cat([c1, up4], dim=1)
        c9 = self.conv9(concat1)
        c10 = self.conv10(c9)

        out = nn.Sigmoid()(c10)
        return out


if __name__ == '__main__':
    x = torch.randn(1, 1, 512, 512)
    net = NET(1, 1)
    print(net(x).shape)
