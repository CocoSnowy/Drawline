from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_mlsd.pth'))
model = model.cpu()
ddim_sampler = DDIMSampler(model)


# 全过程
def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta, low_threshold, high_threshold):
    # torch 禁止梯度下降，直接用已经训练好的模型
    # 推理时，不需要更新模型，不需要计算梯度
    with torch.no_grad():

        # 获取input_imgage，调用canny算法，获取轮廓
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        # control 为轮廓图
        # 后面就是根据control做input，输出图像
        # 下面可以理解成一整个算法，因为我也不太懂

        # 种子，控制再次生成
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


# 需要的参数

# 输入图片 numpy.array
input_image = np.array(cv2.imread('test_imgs/castle.png'))

# prompt 关键词 string
prompt = ' '

# added_prompt string
a_prompt = 'best quality, extremely detailed'

# Negative_prompt string
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, ' \
           'worst quality, low quality '

# 生成图片数量 int
num_samples = 3

# 生成图片的质量（分辨率） int
# 官方 minimum=256, maximum=768, value=512, step=64
image_resolution = 512

# 迭代次数 int
# 越高生成效果越好，同时计算量越大
ddim_steps = 30

# 自动猜测绘画的风格或主题的功能 boolean
guess_mode = True

# 强度参数 float
# 控制强度越高，生成的图像越清晰，但是同时也会带来更多的噪声和不自然的细节。
# 控制强度越低，生成的图像可能会更加模糊或者缺少细节，但同时也会更加平滑和自然。
strength = 1.0

# 引导图像的影响程度 float
# 源码 minimum=0.1, maximum=30.0, value=9.0, step=0.1
scale = 9.0

# 生成种子，相同种子可以生成一样的结果
# minimum=-1, maximum=2147483647, step=1, randomize=True
seed = random.randint(-1, 2147483647)

# 图像的扭曲程度
# eta 值越大，扭曲程度就越大，图像的细节会变得更加扭曲和扭曲
# 通常，eta 的值在0.1到0.01之间进行调整。
eta = 0.0

# canny算法的低阈值
low_threshold = 100

# canny算法的高阈值
high_threshold = 200

# torch 禁止梯度下降，直接用已经训练好的模型
# 推理时，不需要更新模型，不需要计算梯度
with torch.no_grad():
    # 获取input_imgage，调用canny算法，获取轮廓
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

    detected_map = apply_canny(img, low_threshold, high_threshold)
    detected_map = HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().cpu() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()
    # control 为轮廓图
    # 后面就是根据control做input，输出图像
    # 下面可以理解成一整个算法，因为我也不太懂

    # 种子，控制再次生成
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control],
               "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
            [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
        np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    print(type(results[0]))
    #颜色通道修改
    for img in results:
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 2].copy()
        img[:, :, 2] = tmp

    for i in range(0,len(results)):
        cv2.imwrite('res'+str(i)+'.png', results[i])
