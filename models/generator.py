import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from option import opt
from models.dehaze import DehazeModule


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (
            1 - mix_factor.expand_as(fea2)
        )
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self, img_channels=3, out=3):
        super(Generator, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(
                img_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.down2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)

        self.down3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)

        self.dehaze_block = DehazeModule()

        self.up1 = ConvBlock(
            256, 128, down=False, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up2 = ConvBlock(
            128, 64, down=False, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(
                64,
                img_channels,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )

        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):
        x_down1 = self.down1(input)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x_dehaze = self.dehaze_block(x_down3)

        x_out_mix = self.mix1(x_down3, x_dehaze)
        x_up1 = self.up1(x_out_mix)
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix)
        out = self.up3(x_up2)

        return out


def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
