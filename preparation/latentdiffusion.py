import copy

import torch
import torch.nn as nn
import ldm.modules.diffusionmodules.openaimodel as openaimodel
from ldm.modules.attention import SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
import torch.nn.functional as F
import random

bit_length = 48

# [1,0,1 , ..., 0,1,0]

def generate_random_bit_vector(bit_length=48):
    """生成随机的比特向量"""
    return torch.tensor([random.randint(0, 1) for _ in range(bit_length)]).int()


class WatermarkExtracor(nn.Module):
    def __init__(self):
        super(WatermarkExtracor, self).__init__()
        #[b,1280,8,8] 1280个通道，每个通道8*8的特征图，作为输入 b,w,h,c->b,c,w,h
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, bit_length)  # 适应最后一个卷积层的输出

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc(x))  # 使用sigmoid激活函数把像素值归一化到0到1之间
        return (x > 0.5).int()  # 二值化为0或1的比特串



class WatermarkUnet(openaimodel.UnetModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        transformer_depth = kwargs.get("transformer_depth", 1)
        context_dim = kwargs.get("context_dim", 768)
        # 用于嵌入水印的模块，只是重复了中间层SpatialTransformer   Hidden autoEncoder
        self.embedder = SpatialTransformer(self.mid_channels, num_heads, self.mid_dim_head, depth=transformer_depth,
                                           context_dim=context_dim)
        # ----------------------watermark embedding ended---------------------------#
        self.extractor = WatermarkExtracor()
        # self.mid_block.append(self.embedder)
        self.shortcut = nn.Sequential()
        # unet 下采样的结果
        self.unet_encoded = None
        # unet 中间层的结果，这里可以用来输入到原来的输出模块
        self.mid_out = None
        self.watermark = generate_random_bit_vector(bit_length)

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.dtype = self.time_embed[0].weight.dtype
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # 用于计算当前采样时间t的embedding
        t_emb = openaimodel.timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # 对输入模块进行循环，进行下采样并且融合时间特征与文本特征。
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        self.unet_encoded = h
        # 中间模块的特征提取
        h = self.middle_block(h, emb, context)
        # ----------------------watermark embedding ---------------------------#
        #                         水印嵌入
        self.mid_out = copy.deepcopy(h)
        # 直接将水印嵌入到中间层 获取嵌入水印的输出 hw
        hw = self.shortcut(h) + self.embedder(h, emb, self.watermark)
        hsw = copy.deepcopy(hs)
        for module in self.output_blocks:
            hw = torch.cat([hw, hsw.pop()], dim=1)
            hw = module(hw, emb, context)
        hw = hw.type(self.dtype)

        # ----------------------watermark embedding ---------------------------#
        # 上采样模块的特征提取
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(self.dtype)
        # 输出模块
        if self.predict_codebook_ids:
            return self.id_predictor(h), self.id_predictor(hw)
        else:
            return self.out(h), self.out(hw)

    def freeze(self,unfreezeEmbedder=True,unfreezeExtractor=True):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.embedder.parameters():
            param.requires_grad = unfreezeEmbedder
        for param in self.extractor.parameters():
            param.requires_grad = unfreezeExtractor

    def unfreezeExtractor(self):
        for param in self.extractor.parameters():
            param.requires_grad = True
    def unfreezeEmbedder(self):
        for param in self.embedder.parameters():
            param.requires_grad = True


    def get_bitAcc_loss(self):
        # 计算相同位数
        same_bits = torch.sum(self.watermark == self.extractor(self.unet_encoded)).item()
        return 1 - same_bits / len(self.watermark)


class WatermarkLatentDiffusion(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        # 将x_noisy传入self.model中
        x0,xw = self.model(x_noisy, t, **cond)
        return x0,xw
    #x0是原模型的预测噪声的潜在表示，xw是嵌入水印的输出
    def p_loss(self, x0, xw):
        # 计算x0和xw的loss
        loss0 = self.model.get_bitAcc_loss()
        lossw = F.mse_loss(x0, xw) # TODO: utilize mean matrix loss
        return loss0,lossw
    def train(self):
        pass


