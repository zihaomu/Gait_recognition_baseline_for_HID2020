from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim
from torch.optim import RMSprop

def adamw(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, amsgrad=False, **_):
  print("adam lr is :", lr)
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)

def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False, **_):
  print("adam lr is :", lr)
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
  return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

# def get_center_optimizer(parameters, lr = 0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
#   return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

def get_center_optimizer(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False, **_):
  print("center lr is :", lr)
  if isinstance(betas, str):
    betas = eval(betas)
  return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)

def get_optimizer(config, parameters):
  f = globals().get(config.optimizer.name)
  optimizer = f(parameters, config.optimizer.params.lr, weight_decay= config.optimizer.weight_decay)
  return optimizer

