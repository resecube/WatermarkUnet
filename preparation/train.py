import torch
import torch.nn as nn
import ldm.modules.diffusionmodules.openaimodel as openaimodel
from ldm.modules.attention import SpatialTransformer
import preparation.latentdiffusion as ld
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ldm_hacked import *

sd_fp16     = True
vae_fp16    = True

# ----------------------- #
#   生成图片的参数
# ----------------------- #
# 生成的图像大小为input_shape，对于img2img会进行Centter Crop
input_shape = [512, 512]
# 一次生成几张图像
num_samples = 1
# 采样的步数
ddim_steps  = 20
# 采样的种子，为-1的话则随机。
seed        = 12345
# eta
eta         = 0
# denoise强度，for img2img
denoise_strength = 1.00

# ----------------------- #
#   提示词相关参数
# ----------------------- #
# 提示词
prompt      = "a cute cat, with yellow leaf, trees"
# 正面提示词
a_prompt    = "best quality, extremely detailed"
# 负面提示词
n_prompt    = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
# 正负扩大倍数
scale       = 9

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model





def main():
    # load the model and checkpoint
    seed_everything(12345)
    model = create_model('model_data/watermark.yaml').cpu()
    model.load_state_dict(torch.load('model_data/v1-5-pruned-emaonly.safetensors'), strict=False)
    model.cuda()
    #configure the optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # just optimize the embedder and extractor
    optimizer = torch.optim.Adam([model.embedder, model.extractor], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.98)
    #instantiating the DDIM sampler
    ddim_sampler = DDIMSampler(model)
    ddim_sampler.sample(20, 1, 0, 12345, 1.00, prompt, a_prompt, n_prompt, 9)




