# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:41:32 2018
@author: WellenWoo
"""
from __future__ import print_function
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import copy
import time
import re
from tools import Tool
tl = Tool()
 
class ContentLoss(nn.Module):
    """内容损失"""
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # 我们会从所使用的树中“分离”目标内容
        self.target = target.detach() * weight
        # 动态地计算梯度: 它是个状态值, 不是变量.
        # 否则评价指标的前向方法会抛出错误.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, inputs):
        self.loss = self.criterion(inputs * self.weight, self.target)
        self.output = inputs
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
    
class GramMatrix(nn.Module):
    """gram matrix"""
    def forward(self, inputs):
        a, b, c, d = inputs.size()  # a=batch size(=1)
        # b= 特征映射的数量
        # (c,d)= 一个特征映射的维度 (N=c*d)

        features = inputs.view(a * b, c * d)  # 将 F_XL 转换为 \hat F_XL

        G = torch.mm(features, features.t())  # 计算克产物 (gram product)

        # 我们用除以每个特征映射元素数量的方法
            # 标准化克矩阵 (gram matrix) 的值
        return G.div(a * b * c * d)
        
class StyleLoss(ContentLoss):
    """风格损失"""
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__(target = target,weight = weight)
        self.gram = GramMatrix()

    def forward(self, inputs):
        self.output = inputs.clone()
        self.G = self.gram(inputs)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output
    
class Transfer(object):
    def __init__(self,fn_content, fn_style, win, model_path = r'../models/squeezenet1_0-a815701f.pth'):
        """usage:
            net = Transfer('picasso.jpg','dancing.jpg', win)
            dt, img = net.fit()
            """
        self.use_cuda, dtype, imsize = tl.config()
        
        self.content_img = tl.image_loader(fn_content).type(dtype)
        self.style_img = tl.image_loader(fn_style).type(dtype)
        self.input_img = self.content_img.clone()
        
        self.win = win
        
        """目前只能用的预训练模型:vgg,alexNet,squeezenet,densenet;
        原因1:torchvision.model.resnet中的resnet没有features,
        需要用其他方法获取其features;
        2:get_style_model_and_losses函数是针对vgg模型的,
        要应用到其他模型,需要改写该函数;"""
        if 'vgg19' in model_path:
            self.seq = self.load_vgg19(model_path)
        elif 'resnet18' in model_path:
            self.seq = self.load_resnet18(model_path)
        elif "alexnet" in model_path:
            self.seq = self.load_alexnet(model_path)
        elif "squeezenet1_0" in model_path:
            self.seq = self.load_squeezenet(model_path,1.0)
        elif "squeezenet1_1" in model_path:
            self.seq = self.load_squeezenet(model_path,1.1)
        elif "inception" in model_path:
            self.seq = self.load_inception(model_path)
        elif "densenet121" in model_path:
            self.seq = self.load_densenet(model_path)
        if self.use_cuda:
            self.seq = self.seq.cuda()
        
    def load_vgg19(self,model_path):
        """加载vgg19预训练模型;"""
        cnn = models.VGG(models.vgg.make_layers(models.vgg.cfg['E']))
        cnn.load_state_dict(torch.load(model_path))
        return cnn.features
    
    def load_alexnet(self,model_path):
        """加载AlexNet预训练模型;"""
        model = models.AlexNet()
        model.load_state_dict(torch.load(model_path))
        return model.features
    
    def load_squeezenet(self, model_path,version):
        """加载SqueezeNet1.0预训练模型;"""
        model = models.SqueezeNet(version = version)
        model.load_state_dict(torch.load(model_path))
        return model.features
    
    def load_densenet(self, model_path):
        """加载densenet121预训练模型;"""
        model = models.DenseNet(num_init_features = 64, growth_rate = 32,
                                block_config = (6,12,24,16))
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        return model.features
    
    def load_inception(self, model_path):
        """加载Inception预训练模型;
        Inception model没有features;
        目前本函数不可用;"""
        model = models.Inception3()
        model.load_state_dict(torch.load(model_path))
        return model
    
    def load_resnet18(self, model_path):
        """此函数暂不可用,加载的Resnet没有features(sequential);"""
        cnn = models.ResNet(models.resnet.BasicBlock,[2,2,2,2])
        cnn.load_state_dict(torch.load(model_path))
        return cnn 
    
    def fit(self,num_steps = 300, content_weight = 1, style_weight = 1000):
        """返回的cnn:torch.nn.modules.container.Sequential;
        outout_img:PIL.Image.Image;
        style_weight需要远远大于content_weight;"""
        t0 = time.time()
        cnn, tensor = self.rebuild(self.seq, self.content_img,
                                   self.style_img, self.input_img,
                                   num_steps, content_weight,
                                   style_weight)
        output_img = tl.batch_tensor2pil(tensor)
        dt = time.time()-t0
        return dt, output_img

    def rebuild(self, cnn, content_img, style_img, input_img, num_steps,
                content_weight, style_weight ):
        """Run the style transfer."""
        model, style_losses, content_losses = self.get_losses(cnn, style_img, content_img, 
                                                              style_weight, content_weight)
        input_param, optimizer = self.get_optimizer(input_img)
        
        run = [0]
        while run[0] <= num_steps:
            
            def closure():
                # 校正更新后的输入图像值
                input_param.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_param)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.backward()
                for cl in content_losses:
                    content_score += cl.backward()

                run[0] += 1
                if run[0] % 50 == 0:

                    self.win.LogMsg('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score,content_score))
                    self.win.display_out(tl.batch_tensor2pil(input_param.data))

                return style_score + content_score

            optimizer.step(closure)

        # 最后一次的校正...
        input_param.data.clamp_(0, 1)

        return model, input_param.data
    
    def get_losses(self, cnn, style_img, content_img,
                  style_weight, content_weight,
                  content_layers=['conv_4'],
                  style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        cnn = copy.deepcopy(cnn)
    
        # 仅为了有一个可迭代的列表 内容/风格 损失
        content_losses = []
        style_losses = []

        model = nn.Sequential()  # 新建的 Sequential 网络模块

        gram = GramMatrix()  # 我们需要一个克模块 (gram module) 来计算风格目标
        
        # 可能的话将这些模块移到 GPU 上:
        if self.use_cuda:
            model = model.cuda()
            gram = gram.cuda()

        i = 1
        for layer in list(cnn):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)

                if name in content_layers:
                    # 加内容损失:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # 加风格损失:
                    target_feature = model(style_img).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                model.add_module(name, layer)

                if name in content_layers:
                    # 加内容损失:
                    target = model(content_img).clone()
                    content_loss = ContentLoss(target, content_weight)
                    model.add_module("content_loss_" + str(i), content_loss)
                    content_losses.append(content_loss)

                if name in style_layers:
                    # 加风格损失:
                    target_feature = model(style_img).clone()
                    target_feature_gram = gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, style_weight)
                    model.add_module("style_loss_" + str(i), style_loss)
                    style_losses.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                model.add_module(name, layer)  # ***

        return model, style_losses, content_losses 
    
    def get_optimizer(self, input_img):
        # 这行显示了输入是一个需要梯度计算的参数
        input_param = nn.Parameter(input_img.data)
        optimizer = optim.LBFGS([input_param])
        return input_param, optimizer       