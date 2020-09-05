from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import torch


def get_last_checkpoint(checkpoint_dir):
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  if checkpoints:
    return os.path.join(checkpoint_dir, list(sorted(checkpoints))[-1])  # out put the fist parameters
  return None


def get_initial_checkpoint(config):
  checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
  if os.path.exists(checkpoint_dir):
    return get_last_checkpoint(checkpoint_dir)
  else:
    os.makedirs(checkpoint_dir, exist_ok=True)
    return None

def get_checkpoint(config, epoch_num=None):
  if epoch_num is not None:
    name = "epoch_" + "{0:04}".format(epoch_num) + ".pth"
  else:
    name = "epoch_"+"{0:04}".format(config.test.epoch)+".pth"
  checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
  return os.path.join(checkpoint_dir, name)


def copy_last_n_checkpoints(config, n, name):
  checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')
  checkpoints = [checkpoint
                 for checkpoint in os.listdir(checkpoint_dir)
                 if checkpoint.startswith('epoch_') and checkpoint.endswith('.pth')]
  checkpoints = sorted(checkpoints)
  for i, checkpoint in enumerate(checkpoints[-n:]):
    shutil.copyfile(os.path.join(checkpoint_dir, checkpoint),
                    os.path.join(checkpoint_dir, name.format(i)))


def load_checkpoint_test(model, checkpoint):
  print('load checkpoint from', checkpoint)
  checkpoint = torch.load(checkpoint)
  checkpoint_dict = {}
  model.load_state_dict(checkpoint["state_dict"])  # , strict=False)


def load_checkpoint(model, optimizer, center_model, optimizer_center, checkpoint):
  print('load checkpoint from', checkpoint)
  checkpoint = torch.load(checkpoint)
  model.load_state_dict(checkpoint["state_dict"]) #, strict=False)

  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
  
  if optimizer_center is not None:
    print("load optimizer center dict success!")
    optimizer_center.load_state_dict(checkpoint['optimizer_center_dict'])
    center_model.load_state_dict(checkpoint['center_model'])

  step = checkpoint['step'] if 'step' in checkpoint else -1
  last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0

  return last_epoch+1, step+1


def save_checkpoint(config, model, optimizer, center_model, optimzer_center, epoch, step, weights_dict=None, name=None):
  checkpoint_dir = os.path.join(config.train.dir, 'checkpoint')

  if name:
    checkpoint_path = os.path.join(checkpoint_dir, '{}.pth'.format(name))
  else:
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{:04d}.pth'.format(epoch))

  optimzer_center_dict = None
  center_model_dict = None
  if optimzer_center is not None:
    center_model_dict = center_model.state_dict()
    optimzer_center_dict = optimzer_center.state_dict()
  
  if weights_dict is None:
    weights_dict = {
      'state_dict': model.state_dict(),
      'optimizer_dict': optimizer.state_dict(),
      'center_model': center_model_dict,
      'optimizer_center_dict': optimzer_center_dict,
      'epoch': epoch,
      'step': step,
    }
  torch.save(weights_dict, checkpoint_path)
