from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict


def _get_default_config():
  c = edict()
  c.if_train = True
  c.WORK_PATH = "./results"
  c.CUDA_VISIBLE_DEVICES = "5"
  c.writer = 'PATH_TO_TENSORBOARD_LOG'  # None
  c.writer_name = 'data/HID2020'

  # dataset
  c.data = edict()
  c.data.name = "silhouette"
  c.data.dir = 'PATH_TO_DATASET'          # dataset
  c.data.cache = True # this option can speed up the dataloading process
  c.data.cache_path = 'PATH_TO_CACHE_DATASET'
  c.data.pid_num = 500  # How many subjects do you use in training process?
  c.data.random_seed = 999
  c.data.pid_shuffle = False
  c.data.resolution = 64
  c.data.frame_num = 1
  c.data.num_workers = 4
  c.data.drop_last = False
  c.data.collate_fn = "clip"  # option: "select" and "clip"
  c.data.sampler = 'batch'  # options: batch and weight
  
  # model
  c.model = edict()
  c.model.name = 'SilhouetteNormal'  # model name
  c.model.params = edict()

  # train
  c.train = edict()
  c.train.finetuning = None
  c.train.weight = 1
  c.train.dir = './results'      # model save path
  c.train.restore_iter = 0          # training checkpoint       
  c.train.num_epochs = 8000
  c.train.num_grad_acc = None
  c.train.save_step = 20
  c.train.center_wight = 1
  c.train.batch_size = edict()
  c.train.validation = False  # if you want to set validation, please add the corresponding validation process in ROO_PATH/train.py, and genearate the val_dataset at the ROOT_PATH/utils/initialization.py


  c.test = edict()
  c.test.epoch = 9
  c.test.evaluator = 'l2'
  c.test.result_save = False
  c.test.sampler = 'seq'
  c.test.result_name = 'submission.csv'
  c.test.gallery_dir = "PATH_TO_TEST_GALLERY_DATASET"
  c.test.probe_dir = "PATH_TO_TEST_PROBE_DATASET"
  c.test.SampleSubmission_dir = "PATH_TO_SampleSubmission_FILE"  #


  # optimizer
  c.optimizer = edict()
  c.optimizer.weight_decay = 0
  c.optimizer.name = 'adam' # adam, adam_sgd-> softmax+center
  c.optimizer.params = edict()

  # scheduler
  c.scheduler = edict()
  c.scheduler.name = 'step' # step, multistep, exponential
  c.scheduler.base_lr = 1e-4
  c.scheduler.max_lr = 5e-4
  c.scheduler.params = edict()

  # losses
  c.loss = edict()
  c.loss.name = 'cross_entropy' # cross_entropy, softmax_center
  c.loss.params = edict()

  return c

def _merge_config(src, dst):
  if not isinstance(src, edict):
    return

  for k, v in src.items():
    if isinstance(v, edict):
      _merge_config(src[k], dst[k])
    else:
      dst[k] = v


def load(config_path):
  with open(config_path, 'r') as fid:
    yaml_config = edict(yaml.load(fid))

  config = _get_default_config()
  _merge_config(yaml_config, config)

  return config
