from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .silhouette import SilhouetteDataSet

def get_silhouette(seq_dir, condition, label, config):
    dataset = SilhouetteDataSet(seq_dir, condition, label, config)
    return dataset

def get_dataset(seq_dir, condition, label, config, transform=None, view=None):
    # print("out put is ", config)
    f = globals().get('get_'+config.data.name) # begin creat a class
    return f(seq_dir, condition, label, config)

