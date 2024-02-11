import copy
import os
import torch
import torch.nn as nn
import ldm.modules.diffusionmodules.openaimodel as openaimodel
from ldm.modules.attention import SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
import torch.nn.functional as F
import random
from preparation.AutoEncoderModels.encoder import Encoder as CustomEmbedder
from preparation.AutoEncoderModels.decoder import Decoder as CustomExtractor
from preparation.AutoEncoderModels.discriminator import Discriminator as CustomDiscriminator
from tqdm import tqdm
bit_length = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



class WatermarkUnet(openaimodel.UNetModel):
    def __init__(self, embedder = CustomEmbedder(1280,1280,2560,bit_length,128,128),
                 extractor = CustomExtractor()
                 ,watermark = None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        num_heads = kwargs.get("num_heads", 8)
        transformer_depth = kwargs.get("transformer_depth", 1)
        context_dim = kwargs.get("context_dim", 768)
        self.embedder = embedder.to(device)
        # if exists embedder.pth
        if os.path.exists("model_data/embedder.pth"):
            print("load embedder.pth")
            self.embedder.load_state_dict(torch.load("embedder.pth"))

        # 用于嵌入水印的模块，只是重复了中间层SpatialTransformer   Hidden autoEncoder
        # self.embedder = SpatialTransformer(self.mid_channels, num_heads, self.mid_dim_head, depth=transformer_depth,
        #                                    context_dim=context_dim)
        # ----------------------watermark embedding ended---------------------------#
        self.extractor = extractor.to(device)
        # if exists extractor.pth
        if os.path.exists("model_data/extractor.pth"):
            print("load extractor.pth")
            self.extractor.load_state_dict(torch.load("extractor.pth"))
        self.shortcut = nn.Sequential()
        # unet 下采样的结果
        self.unet_encoded = None
        # unet 中间层的结果，这里可以用来输入到原来的输出模块
        self.mid_out = None
        if watermark is not None:
            self.watermark = watermark.to(device)
        else:
            self.watermark = generate_random_bit_vector(bit_length).to(device)
        self.bit_acc_loss = None
        self.loss_model = None
        self.loss = None

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
        hw = self.shortcut(self.mid_out) + self.embedder(self.mid_out, emb, self.watermark)
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
        self.model.freeze()
        self.optimizer = torch.optim.Adam([*self.model.diffusion_model.embedder.parameters(),*self.model.diffusion_model.extractor.parameters()], lr=1e-4)

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
        return x0, xw
    #x0是原模型的预测噪声的潜在表示，xw是嵌入水印的输出
    # def p_loss(self, x0, xw):
    #     # 计算x0和xw的loss
    #     loss0 = self.model.get_bitAcc_loss()
    #     lossw = F.mse_loss(x0, xw) # TODO: utilize mean matrix loss
    #     return loss0,lossw

    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps
        # ----------- make preparation for the first step ended ----------------------#
        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        #
        model_out, model_out_w = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out
            model_out_w, logits_w = model_out_w

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
            x_recon_w = self.predict_start_from_noise(x, t=t, noise=model_out_w)
        elif self.parameterization == "x0":
            x_recon = model_out
            x_recon_w = model_out_w
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
            x_recon_w.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
            x_recon_w, _, [_, _, indices_w] = self.first_stage_model.quantize(x_recon_w)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        model_mean_w, posterior_variance_w, posterior_log_variance_w = self.q_posterior(x_start=x_recon_w, x_t=x, t=t)
        self.loss_bit_acc = self.model.get_bitAcc_loss()
        self.loss_model = F.mse_loss(model_mean, model_mean_w)

        self.loss = self.loss_model + self.loss_bit_acc
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance


    def train(self):
        prompt = "a cute cat, with yellow leaf, trees"
        # 正面提示词
        a_prompt = "best quality, extremely detailed"
        # 负面提示词
        n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        # 正负扩大倍数
        scale = 9
        cond    = {"c_crossattn": [self.get_learned_conditioning([prompt + ', ' + a_prompt] * 1)]}
        # un_cond = {"c_crossattn": [self.get_learned_conditioning([n_prompt] * 1)]}
        for i in range(10000):
            self.progressive_denoising(cond,start_T=50,shape=[1,4,128,128],verbose=True,log_every_t=1)
            if i % 100 == 0:
                print(f"epoch:{i},loss:{self.loss.item()},bit_acc_loss:{self.loss_bit_acc.item()},model_loss:{self.loss_model.item()}")
                # save the model
                torch.save(self.model.embedder.state_dict(), "embedder.pth")
                torch.save(self.model.extractor.state_dict(), "extractor.pth")

