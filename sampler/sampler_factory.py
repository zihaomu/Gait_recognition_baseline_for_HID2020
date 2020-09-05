from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data.sampler import BatchSampler
from .weight import WeightSampler


def get_batch(dataset, config):
    # (sampler, batch_size, drop_last):
    return None

def get_weight(dataset, config):

    return WeightSampler(dataset, [config.train.batch_size.batch1, config.train.batch_size.batch2])

def get_sampler(dataset, config):
  f = globals().get('get_'+config.data.sampler)
  return f(dataset, config)