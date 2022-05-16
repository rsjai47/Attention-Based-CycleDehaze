import torch
import torch.nn as nn
import functools


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Discriminator, self).__init__()

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):

        return self.model(input)


def test():
    x = torch.randn((5, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)


if __name__ == "__main__":
    test()
