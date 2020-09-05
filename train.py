from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
import argparse
import torch
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as Tranformer

from models import get_model
from losses import get_loss, get_center_loss
from optimizers import get_optimizer, get_center_optimizer
from schedulers import get_scheduler
from sampler import get_sampler

import utils
from utils.checkpoint import get_initial_checkpoint, load_checkpoint, save_checkpoint
import utils.metrics
from utils import get_initial, get_collate_fn


# def transformer(): # you can add augmentation here
#     transfor = Tranformer.Compose([
#         Tranformer.ToPILImage(),
#         # Tranformer.RandomVerticalFlip(p=1),
#         Tranformer.ToTensor()
#     ])
#     return transfor


class Train(object): 

    def __init__(self, config): 

        self.config = config
        self.model = None
        self.optimizer = None
        self.optimizer_center = None  # reserved for center loss
        self.scheduler = None
        self.scheduler_center = None
        self.writer = None
        self.label = None
        self.label_encoder = None
        self.sampler = None
        self.loss_function = None
        self.center_model = None       # reserved for center loss
        self.writer = None
        self.loss_data = []
        self.loss_center_data = []
        self.data_loader = None
        self.data_loader_val = None
        self.dataset = None
        self.more_label = None
        self.collate_fn = None
        self.num_epochs = self.config.train.num_epochs
        self.num_workers = self.config.data.num_workers
        self.sample_type = 'random'
        self.last_epoch = -1
        self.step = -1
        self.iteration = 0
        if self.writer is not None:
            self.writer = SummaryWriter(self.config.writer)
        ###
        # self.transformer = transformer()  # baseline don't use any augmentation strategy

    def plot_loss(self):
        len_loss = len(self.loss_data)
        x = np.arange(0, len_loss)
        y = self.loss_data

        if len(self.loss_center_data) != 0:
            plt.plot(x, self.loss_center_data, 'b-', label='center loss ')
        plt.plot(x, y, 'g-', label='cross entropy loss')

        plt.legend()

        plt.xlabel('iters')
        plt.ylabel('loss')
        plt.title('Loss curve in training')
        plt.savefig(os.path.join(self.config.train.dir, "_loss.png"))
        print("save image success!")
        print("image path ", os.path.join(self.config.train.dir, "_loss.png"))
        plt.cla()

    def set_new_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        if self.optimizer_center is not None:
            for param_group in self.optimizer_center.param_groups:
                param_group['lr'] = new_lr


    def initialization(self):

        SEED = self.config.data.random_seed
        if SEED != 999:
            print("random_seed is ",SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)
        else:
            print(" no random seeds!")

        self.dataset, val_dataset, self.label = get_initial(self.config, train = True)  # return dataset instance
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label)
        torch.cuda.empty_cache()

        self.model = get_model(self.config)
        self.optimizer = get_optimizer(self.config, self.model.parameters())
        checkpoint = get_initial_checkpoint(self.config)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

        if self.config.loss.name == "softmax_center":
            print("Add center loss !!")

            self.center_model = get_center_loss(class_num=self.config.data.pid_num, feature_num=512, use_gpu=True)
            self.optimizer_center = get_center_optimizer(self.center_model.parameters(), self.config.optimizer.params.lr)

        if checkpoint is not None:
            self.last_epoch, self.step = load_checkpoint(self.model, self.optimizer, self.center_model, self.optimizer_center, checkpoint)
        print("from checkpoint {} last epoch: {}".format(checkpoint, self.last_epoch))

        self.sampler = get_sampler(self.dataset, self.config) 
        self.loss_function = get_loss(self.config)

        self.collate_fn = get_collate_fn(self.config, self.config.data.frame_num, self.sample_type) #

        if self.sampler is not None:
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_sampler=self.sampler,
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers,)
        else:
            self.data_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.config.train.batch_size.batch1,
                collate_fn=self.collate_fn, 
                num_workers=self.num_workers,
                drop_last=self.config.data.drop_last,
                shuffle= self.config.data.pid_shuffle,
            )


    def train_sigle_iteration(self, seq, label):

        self.optimizer.zero_grad()
        if self.optimizer_center is not None:
            self.optimizer_center.zero_grad()

        seq = torch.Tensor(seq).float().cuda()

        fc, out = self.model(seq)

        label = self.label_encoder.transform(label)
        label = torch.Tensor(label).long().cuda()

        loss = self.loss_function(out, label)
        pred = torch.max(out, 1)[1]
        acc = (pred == label).sum()

        loss = loss
        loss_temp = loss.item()
        loss_center = 0
        loss_center_temp = 0
        if self.center_model is not None:
            # fc features -> normalization
            fc_norm = F.normalize(fc, p=2, dim=1)

            loss_center = self.center_model(fc_norm, label)*self.config.train.center_wight
            loss_center_temp = loss_center.item()

        loss = loss_center + loss
        loss.backward()  # caculate grad

        self.optimizer.step()  # update parameters
        if self.center_model is not None:
            self.optimizer_center.step()
        return acc.item(), loss_temp, loss_center_temp


    def train_weigh(self):
        acc_sample = 0
        count_all = 0
        all_loss = 0
        all_center_loss = 0
        total_num = len(self.dataset)
        batch_size = self.config.train.batch_size.batch1 * self.config.train.batch_size.batch2

        step_num = math.ceil(total_num/batch_size)

        epoch = self.last_epoch
        iteration = epoch*step_num
        # print("step number is ", step_num)
        for seq, vID, label, _ in self.data_loader:
            iteration += 1
            count_all += len(label)
            acc_i, loss, loss_center = self.train_sigle_iteration(seq, label)
            all_loss += loss
            all_center_loss += loss_center
            acc_sample += acc_i

            if iteration % step_num == step_num - 1:
                self.scheduler.step()
                if self.scheduler_center is not None:
                    self.scheduler_center.step()
                epoch += 1

                if (epoch % self.config.train.save_step) == (self.config.train.save_step - 1):
                    print("save loss log image")
                    self.plot_loss()
                    save_checkpoint(self.config, self.model, self.optimizer, self.center_model, self.optimizer_center,
                                    epoch, self.step)

                if self.writer is not None:
                    self.writer.add_scalar("train_loss", all_loss, epoch)

                acc_epoch = acc_sample * 1.0 / count_all
                if self.center_model is not None:
                    print(
                        "training in epoch :{}, the acc is {}% ,\n the cross loss is {}, the center loss is {}".format(epoch, acc_epoch * 100,
                                                                                          all_loss, all_center_loss))
                    self.loss_center_data.append(all_center_loss)
                else:
                    print(
                        "training in epoch :{}, the acc is {}% ,\n the loss is {}".format(epoch, acc_epoch * 100, all_loss))

                self.loss_data.append(all_loss)

                print("learning rate: ", self.optimizer.param_groups[0]['lr'])
                acc_sample = 0
                count_all = 0
                all_loss = 0
                all_center_loss = 0

            if epoch > self.config.train.num_epochs:
                break


    def train_single_epoch(self, epoch):
        acc_sample = 0
        count_all = 0
        all_loss = 0

        for seq, vID, label, _ in self.data_loader:
            count_all += len(label)
            acc_i, loss = self.train_sigle_iteration(seq, label)
            all_loss += loss
            acc_sample += acc_i
        if self.writer is not None:
            self.writer.add_scalar("train_loss", all_loss, epoch)
        acc_epoch = acc_sample * 1.0 / count_all

        print("training in epoch :{}, the acc is {}% ,\n the loss is {}".format(epoch, acc_epoch * 100, all_loss))
        print("learning rate: ", self.optimizer.param_groups[0]['lr'])


    def run(self):
        # checkpoint
        self.scheduler = get_scheduler(self.config, self.optimizer, self.last_epoch)
        self.model.train()
        postfix_dic = {
            'lr': 0.0,
            'acc' : 0.0,
            'loss' : 0.0,
        }

        if self.config.data.sampler == "weight":
            self.train_weigh()
        else:
            for epoch in range(self.last_epoch, self.num_epochs):

                self.train_single_epoch(epoch)

                if epoch % 200 == 199:
                    save_checkpoint(self.config, self.model, self.optimizer, self.optimizer_center, epoch, self.step)

                self.scheduler.step()
                if epoch > self.config.train.num_epochs:
                    break


def parse_args():
    parser = argparse.ArgumentParser(description='config file')

    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default="./configs/baseline_config.yml", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.config_file is None:
        raise Exception("no configuration file.")

    config = utils.config.load(args.config_file)
    config.train.dir = os.path.join(config.train.dir, os.path.basename(args.config_file)[:-4])

    trainer = Train(config)
    trainer.initialization()
    trainer.run()
    print("Training complete!")

if __name__ == '__main__':
    main()
