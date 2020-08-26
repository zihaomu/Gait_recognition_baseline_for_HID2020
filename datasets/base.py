import torch.utils.data as tordata
import numpy as np
import os.path as osp
import os
import pickle
import cv2
import xarray as xr

from torchvision import transforms as tfs


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, date, label, cache, resolution):
        self.seq_dir = seq_dir
        self.date = date
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)
        self.date_set = set(self.date)
        _ = np.zeros((len(self.label_set),
                      len(self.date_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(   # 知道第几个图片的坐标是什么
            _,
            coords={'label': sorted(list(self.label_set)),
                    'date': sorted(list(self.date_set))},
            dims=['label', 'date'])

        for i in range(self.data_size):
            _label = self.label[i]
            _date = self.date[i]
            self.index_dict.loc[_label, _date] = i

    def load_all_data(self):
        for i in range(self.data_size):  # load data process
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def __getitem__(self, index):
        # pose sequence sampling
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

        return data, frame_set, self.date[index], self.label[
            index],

    def img2xarray(self, flie_path):
        # print(flie_path)
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.resize(cv2.imread(osp.join(flie_path, _img_path)),(64,64),interpolation=cv2.INTER_CUBIC),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.label)


def test(flie_path="path"):
    resolution = 64
    imgs = sorted(list(os.listdir(flie_path)))
    frame_list = [np.reshape(
        cv2.resize(cv2.imread(osp.join(flie_path, _img_path)), (64, 64), interpolation=cv2.INTER_CUBIC),
        [resolution, resolution, -1])[:, :, 0]
                  for _img_path in imgs
                  if osp.isfile(osp.join(flie_path, _img_path))]
    num_list = list(range(len(frame_list)))

    data_dict = xr.DataArray(
        frame_list,
        coords={'frame': num_list},
        dims=['frame', 'img_y', 'img_x'],
    )
    return data_dict