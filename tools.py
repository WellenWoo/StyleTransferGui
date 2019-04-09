# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:30:00 2019
@author: Wellenwoo
图像格式转换的工具;
"""
from __future__ import print_function

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

tensor2pil = transforms.ToPILImage()  # 转回 PIL 图像

class Tool(object):
    def __init__(self):
        self.use_cuda, self.dtype, self.imsize = self.config()
        
    def config(self):
        """根据是否有gpu,返回相关数据;"""
        use_cuda = torch.cuda.is_available() #是否有可用的gpu
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor #根据cpu/gpu选择数据类型
        imsize = 512 if use_cuda else 128  # 如果没有 GPU 则使用小尺寸
        return use_cuda,dtype,imsize

    def image_loader(self, fn):
        """读入图片并转为tensor;
        param:
            fn: str,图像路径;
        return:
            img: torch.Tensor,图像的张量, shape=([1, 3, 512, 512]);"""
        img = Image.open(fn)
        if img.size[0] == img.size[1]:
            pass
        else:
            size = min(img.size)
            img = img.resize((size,size))
        img = Variable(self.img2tensor(img))
        # 由于神经网络输入的需要, 添加 batch 的维度
        #使其shape变为(1,N_channel,height,weight)
        img = img.unsqueeze(0)
        return img
    
    def img2tensor(self, img):
        """将PIL.Image.Image转为tensor;
        param:
            img: PIL.Image.Image,图像;
        return: 
            torch.Tensor;"""
        trans = transforms.Compose([
                transforms.Resize(self.imsize),  # 缩放图像
                transforms.ToTensor()])  # 将其转化为 torch 张量   
        return trans(img)

    def batch_tensor2pil(self, batch_tensor):
        """将(batch_size,n_channel,height,weight)格式的tensor
        转为PIL.Image.Image
        param:
            tensor: torch.Tensor,
        return:
            img: PIL.Image.Image;"""
        arr = batch_tensor.clone().cpu()
        arr = arr.view(3,self.imsize, self.imsize)
        img = tensor2pil(arr)
        return img
    