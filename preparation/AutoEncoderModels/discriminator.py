import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms


class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.config = {
            "discriminator_blocks": 3,
            "discriminator_channels": 64,
        }
        layers = [ConvBNRelu(3, self.config['discriminator_channels'])]
        for _ in range(self.config['discriminator_blocks'] - 1):
            layers.append(ConvBNRelu(self.config['discriminator_channels'], self.config['discriminator_channels']))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(self.config['discriminator_channels'], 1)

    def forward(self, image):
        X = self.before_linear(image)
        # 压缩维度, 去掉最后两个维度
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        # X = torch.sigmoid(X)
        return X