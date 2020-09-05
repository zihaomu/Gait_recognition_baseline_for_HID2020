from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
from tqdm import tqdm


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from models import get_model
from losses import get_loss, get_center_loss
from optimizers import get_optimizer, get_center_optimizer
from schedulers import get_scheduler
from sampler import get_sampler

import utils
from utils.checkpoint import get_checkpoint, load_checkpoint, save_checkpoint
import utils.metrics
from utils import get_collate_fn, Evaluator, get_initial_test
# change training parameters from py dictionary to

class Test(object):

    def __init__(self, config):

        self.config = config
        self.model = None
        self.optimizer = None
        self.optimizer_center = None
        self.scheduler = None
        self.writer = None
        self.sampler = None
        self.loss_function = None
        self.loss_center = None
        self.center_model = None
        # self.writer = self.config.writer
        self.writer = None
        self.data_loader = None
        self.dataset = None
        self.data_loader_test = None
        self.gallery = None
        self.collate_fn = None
        self.num_epochs = self.config.train.num_epochs
        self.num_workers = self.config.data.num_workers
        self.sample_type = 'all'
        self.last_epoch = 0
        self.step = -1
        self.iteration = 0
        self.writer = None


    def initialization(self):
        WORK_PATH = self.config.WORK_PATH
        os.chdir(WORK_PATH)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.CUDA_VISIBLE_DEVICES

        print("GPU is :", os.environ["CUDA_VISIBLE_DEVICES"])

        self.model = get_model(self.config)

        self.optimizer = get_optimizer(self.config, self.model.parameters())

        checkpoint = get_checkpoint(self.config)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.last_epoch, self.step = load_checkpoint(self.model, self.optimizer, self.center_model, self.optimizer_center, checkpoint)
        print("from checkpoint {} last epoch: {}".format(checkpoint, self.last_epoch))

        self.collate_fn = get_collate_fn(self.config, self.config.data.frame_num, self.sample_type)  #



    def extract_gallery_feature(self, data_gallery, len_gallery):

        features = list()

        for i in range(len_gallery):

            seq = data_gallery[i].values
            seq = torch.from_numpy(np.asarray(seq))
            seq = torch.unsqueeze(seq, 0)

            fc, out = self.model(seq)
            n, num_bin = fc.size()
            feat = fc.view(n, -1).data.cpu().numpy()

            # feature normalization
            for ii in range(n):
                feat[ii] = feat[ii] / np.linalg.norm(feat[ii])
            features.append(feat)

        return features

    def save_npy(self, data, path):
        np.save(os.path.join(self.config.train.dir, path), data)

    def load_npy(self, path):
        return np.load(os.path.join(self.config.train.dir, path))
        
    def run(self):
        # checkpoint
        self.model = self.model.eval()
        self.dataset, test_gallery = get_initial_test(self.config, test=True)  # return dataset instance

        print("data set len is :",len(self.dataset))
        data_gallery, vID_gallery, label_gallery = test_gallery[0], test_gallery[1], test_gallery[2]

        print("sample leve ----------->", self.config.test.sampler)

        # dataloader define
        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=1,
            sampler=SequentialSampler(self.dataset),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        len_gallery = len(label_gallery)

        feature_gallery = self.extract_gallery_feature(data_gallery, len_gallery)

        probe_feature = list()
        probe_vID = list()


        for seq, vID, label, _ in tqdm(self.data_loader):

            seq = torch.from_numpy(seq).float().cuda()
            # print(seq.size())
            fc, out = self.model(seq)
            n, num_bin = fc.size()
            feat = fc.view(n, -1).data.cpu().numpy()

            for ii in range(n):
                feat[ii] = feat[ii] / np.linalg.norm(feat[ii])

            probe_feature.append(feat)
            probe_vID += vID

        test_gallery = feature_gallery, vID_gallery, label_gallery
        feature_probe = np.concatenate(probe_feature, 0)
        test_probe = feature_probe, probe_vID

        self.save_npy(feature_gallery, "feature_gallery.npy")
        self.save_npy(vID_gallery, "vID_gallery.npy")
        self.save_npy(label_gallery, "label_gallery.npy")
        self.save_npy(feature_probe, "feature_probe.npy")
        self.save_npy(probe_vID, "probe_vID.npy")

        evaluation = Evaluator(test_gallery, test_probe, self.config)
        evaluation.run()

    def inference(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='config file')

    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="./configs/baseline_config.yml", type=str)

    parser.add_argument('--epoch', dest='epoch',
                        help='epoch',
                        default=None, type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config_file is None:
        raise Exception("Miss configuration file.")

    config = utils.config.load(args.config_file)
    config.train.dir = os.path.join(config.train.dir, os.path.basename(args.config_file)[:-4])
    print(config.train.dir)

    if args.epoch is not None:
        config.test.epoch = int(args.epoch)
        print("Epoch ", config.test.epoch)

    tester = Test(config)
    tester.initialization()

    tester.run()
    print("Test complete !!")


if __name__ == '__main__':
    main()
