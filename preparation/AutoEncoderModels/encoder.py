import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,channels_in=3,channels_out=3,channels_hidden=64,watermark_length=30,H=128,W=128):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels_hidden, channels_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels_hidden, channels_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels_hidden, channels_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True),
        )
        self.after_concat_layer = nn.Sequential(
            nn.Conv2d(channels_in+channels_hidden+watermark_length, channels_hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_hidden),
            nn.ReLU(inplace=True)
        )
        self.final_layer = nn.Conv2d(channels_hidden, channels_out, kernel_size=1)

        # initialize H, W
        self.H = H
        self.W = W

    def forward(self, image, message):
        # 首先，将消息的最后两个维度添加两个虚拟维度。
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        # 然后，将消息扩展为图像大小
        expanded_message = expanded_message.expand(-1, -1, self.H, self.W)
        conved_image = self.conv_layers(image)
        # 接着，连接 expanded_message, conved_image, image
        # print("conved_image.shape: ", conved_image.shape)
        # print("expanded_message.shape: ", expanded_message.shape)
        # print("image.shape: ", image.shape)
        concat = torch.cat([expanded_message, conved_image, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w



