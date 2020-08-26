import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr
import pandas as pd

class SilhouetteDataSet(tordata.Dataset):
    def __init__(self, seq_dir, vID, label, config):
        self.seq_dir = seq_dir
        print("the dataset name is ", config.data.name)
        self.vID = vID
        self.label = label
        self.config = config
        self.cache = self.config.data.cache
        self.resolution = int(self.config.data.resolution)
        self.cut_padding = int(float(self.resolution)/self.resolution*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        # print("label len", len(label))
        # print("label set len", len(self.label_set))
        self.vID_set = set(self.vID)
        _ = np.zeros((len(self.label_set),
                      len(self.vID_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'video_name': sorted(list(self.vID_set))},
            dims=['label', 'video_name'])

        for i in range(self.data_size):
            _label = self.label[i]
            _vID = self.vID[i]
            self.index_dict.loc[_label, _vID] = i

    def load_all_data(self):
        for i in range(self.data_size): # load data process
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def loader(self, path):
        return self.__loader__(path[0])

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def __getitem__(self, index):
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.vID[index], self.label[
            index],

    def img2xarray(self, filepath):
        # print(flie_path)
        imgs = sorted(list(os.listdir(filepath)))
        # print(imgs)
        frame_list = [np.reshape(
            cv2.resize(cv2.imread(osp.join(filepath, _img_path)), (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(filepath, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)
