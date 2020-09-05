from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lovasz import lovasz_hinge
import os
from .center import CenterLoss

# def binary_focal_loss(gamma=2, **_):
#     def func(input, target):
#         assert target.size() == input.size()

#         max_val = (-input).clamp(min=0)

#         loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#         invprobs = F.logsigmoid(-input * (target * 2 - 1))
#         loss = (invprobs * gamma).exp() * loss

#         return loss.mean()
#     return func

# def binary_lovasz_loss(gamma=2, **_):
#     def func(input, target):
#         assert target.size() == input.size()

#         # max_val = (-input).clamp(min=0)

#         # loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#         # invprobs = F.logsigmoid(-input * (target * 2 - 1))
#         # loss = (invprobs * gamma).exp() * loss
#         # fl = loss.mean()
#         # 增加 lovasz loss
#         lovasz_loss = lovasz_hinge(input, target)
#         return lovasz_loss
#     return func

# def lovasz_focal_loss(gamma=2, **_):
#     def func(input, target):
#         assert target.size() == input.size()

#         max_val = (-input).clamp(min=0)

#         loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
#         invprobs = F.logsigmoid(-input * (target * 2 - 1))
#         loss = (invprobs * gamma).exp() * loss
#         fl = loss.mean()
#         # 增加 lovasz loss
#         lovasz_loss = lovasz_hinge(input, target)
#         return lovasz_loss + fl
#     return func

def get_weight(path):
    number_list = os.listdir(path)
    weight_number = []

    for pathi in number_list:
        len_i = len(os.listdir(os.path.join(path, pathi)))

        weight_number.append(1.0/len_i)

    return weight_number


def get_center_loss(class_num, feature_num = 512, use_gpu = True):  # center_loss needs (classes, featrure dimension)
    return CenterLoss(class_num, feature_num, use_gpu= True)


def softmax_center(_):
    return torch.nn.CrossEntropyLoss()

def cross_weight(path):
    weight = get_weight(path)
    weight = torch.Tensor(weight[:86]).float().cuda()
    return torch.nn.CrossEntropyLoss(weight = weight)

def cross_entropy(_):
    return torch.nn.CrossEntropyLoss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(config.data.dir)
