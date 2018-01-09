from __future__ import print_function
import argparse

import numpy as np
# import h5py
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from model import NetARCNN
from model import NetARCNN_deform

from PIL import Image
# from torchvision.transforms import ToTensor
from math import log10

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Deblocking Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--target_image', type=str, required=True, help='target image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
img = Image.open(opt.input_image)
target = Image.open(opt.target_image)

# model = NetARCNN()
model = NetARCNN_deform()

model.load_state_dict(torch.load(opt.model))
print("model built")

# img = np.asarray(img)
# img = img.astype(np.float32)/255

# target = target.astype(np.float32)

# target = torch.from_numpy(target)
# target = Variable(target.view(1, -1, target.size[1], target.size[0]))

# print(img.size)
# input = torch.from_numpy(img)
# input = Variable(input.view(1, -1, input.size[1], input.size[0])) # ? first 1 then 0?

# Actually, ToTensor() will operate the normalization automatically

input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
target0 = Variable(ToTensor()(target)).view(1, -1, target.size[1], target.size[0])

# print(input,target0)

# check dataprocess for detail

if opt.cuda:
    model = model.cuda()
    input = input.cuda()
    target0 = target0.cuda()

output = model(input)

criterion = nn.MSELoss()
mse1 = criterion(input, target0)
mse2 = criterion(output, target0)

psnr01 = 10 * log10(1/mse1.data[0])
psnr02 = 10 * log10(1/mse2.data[0])
print("Before: PSNR {} dB (provided by PyTorch)\nAfter: PSNR {} dB(provided by PyTorch)".format(psnr01, psnr02))


out = output.cpu()
out = out.data[0].numpy() * 255.0

# is clip necessary?
out = np.uint8(out.clip(0,255))
img = np.asarray(img)
# img = np.asarray(img).astype(np.float32)
target = np.asarray(target)

psnr1 = psnr(target, img)
ssim1 = ssim(target, img)

out = np.squeeze(out, axis=0)
# print(out)
# print(target)
psnr2 = psnr(target, out)
ssim2 = ssim(target, out)

print("Before: PSNR: {} dB, SSIM: {}\nAfter: PSNR: {} dB, SSIM: {}".format(psnr1, ssim1, psnr2, ssim2))

# out_img = Image.fromarray(np.uint8(out))	
out_img = Image.fromarray(out)
out_img.save(opt.output_filename)

# out_img = out_img.astype(np.int)
# out_img = Image.fromarray(out_img) 

out_img.save(opt.output_filename)
print("output image saved to", opt.output_filename)

