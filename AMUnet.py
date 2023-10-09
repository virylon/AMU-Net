import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.bc = nn.Sequential(
            nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.bc(x)
        return x


class Attention(nn.Module):
    """input_channels, out_channels"""

    def __init__(self, in_channels, out_channels):
        super(Attention,self).__init__()

        sub_channels = int(out_channels / 4)
        self.q = BasicConv2d(in_channels, sub_channels, 1)
        self.k = BasicConv2d(in_channels, sub_channels, 1)
        self.v = BasicConv2d(in_channels, out_channels, 3 , padding=1)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            BasicConv2d(sub_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        psi = self.psi(q*k)
        out=psi*v

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Attention(in_channels, out_channels),
            Attention(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Multiscale(nn.Module):
    def __init__(self, in_channels):
        super(Multiscale, self).__init__()
        sub_channels= int(in_channels/4)
        self.s0 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s2 = nn.Sequential(
            nn.Conv2d(in_channels, sub_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))
        self.s3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, sub_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(sub_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        s0=self.s0(x)
        s1=self.s1(x)
        s2=self.s2(x)
        s3=self.s3(x)
        cats=torch.cat((s0,s1,s2,s3),dim=1)

        return cats


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ms = Multiscale(out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        ms = self.ms(x2)
        x = torch.cat([ms, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MAUnet(nn.Module):
    """ Full assembly of the parts to form the complete network """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(MAUnet, self).__init__()
        self.name = 'MAUnet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
