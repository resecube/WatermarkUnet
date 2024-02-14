import torch
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


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.config = {
            "decoder_blocks": 7,
            "decoder_channels": 2560,
            "message_length": 30
        }
        self.channels = self.config['decoder_channels']

        layers = [ConvBNRelu(1280, self.channels)]
        for _ in range(self.config['decoder_blocks'] - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        layers.append(ConvBNRelu(self.channels, self.config['message_length']))

        # 将输入的特征图压缩成一个固定长度的特征向量
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        # 将网络的输出压缩为指定的水印消息长度。
        self.linear = nn.Linear(self.config['message_length'], self.config['message_length'])

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        x.squeeze_(3).squeeze_(2) # 维度压缩，去掉最后两个维度
        x = self.linear(x)
        x = torch.sigmoid(x)
        x = (x > 0.5).int()
        # print("消息：", x)
        return x




