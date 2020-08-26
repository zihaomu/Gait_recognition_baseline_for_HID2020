# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
from os import path as osp

import numpy as np
import pickle
from datasets.dataset_factory import get_dataset
import random

import pickle
import cv2
import xarray as xr
import scipy.io as sio


def int_list(temp_list):
    temp = []
    for i in range(len(temp_list)):
        temp.append(int(temp_list[i]))
    return temp

def load_data_from_path_pose(path):
    mat = sio.loadmat(path)['matrix']
    return mat


def load_data_from_path(path, resolution = 64):
    cut_padding = int(float(resolution) / 64 * 10)
    return img2xarray(path[0], resolution)[:, :, cut_padding:-cut_padding].astype('float32') / 255.0


def img2xarray(filepath, resolution):
    # print(flie_path)
    # load data from a given every file path
    imgs = sorted(list(os.listdir(filepath)))
    frame_list = [np.reshape(
        cv2.resize(cv2.imread(osp.join(filepath, _img_path)), (64, 64), interpolation=cv2.INTER_CUBIC),
        [resolution, resolution, -1])[:, :, 0]
                  for _img_path in imgs
                  if osp.isfile(osp.join(filepath, _img_path))]
    num_list = list(range(len(frame_list)))
    data_dict = xr.DataArray(
        frame_list,
        coords={'frame': num_list},
        dims=['frame', 'img_y', 'img_x'],
    )
    return data_dict


def split_val_dataset(seq, date, label):

    temp_label = None
    index_list = 2*[]
    index_len = []
    k = 0
    j = 0
    for i in range(len(label)):
        if temp_label == None:
            temp_label = label[0]
            index_list += [0]
            j += 1

        elif temp_label == label[i]:
            index_list[k] += i
            j += 1 
        else:
            temp_label = label[i]
            k += 1
            index_list += [i]
            index_len.append(j)
            j = 0
    
    seq_train = []
    date_train = []
    label_train = []

    seq_val = []
    date_val = []
    label_val = []

    for j in range(len(index_list)):
        if index_len[i] > 3:
            
            num = index_len[i] - int(index_len[i]/10) + 1
        
            seq_train += [seq[i] for i in index_list[j][:num]]
            date_train += [date[i] for i in index_list[j][:num]]
            label_train += [label[i] for i in index_len[j][:num]]

            seq_val += [seq[i] for i in index_list[j][num:]]
            date_val += [date[i] for i in index_list[j][num:]]
            label_val += [label[i] for i in index_len[j][num:]]
        else:
            seq_train += [seq[i] for i in index_list[j]]
            date_train += [date[i] for i in index_list[j]]
            label_train += [label[i] for i in index_len[j]]
    
    return seq_train, date_train, label_train, seq_val, date_val, label_val


def load_data(config, dataset_path, resolution, dataset, pid_num, cache, pid_shuffle=False):
    seq_dir = list()
    vID = list()
    label = list()
    print("Dataset path is ", dataset_path)
    # load all data from file path

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)        # 005
        for seqs in sorted(list(os.listdir(label_path))):
            seqs_path = osp.join(label_path, seqs)

            # check if there is image?

            if len(os.listdir(seqs_path)) == 0:
                print("no image in ", seqs_path)
                continue
            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            vID.append(seqs)

    label_set = list(set(label))
    train_list, val_list = label_set[:pid_num], label_set[pid_num:]

    if pid_shuffle:
        np.random.shuffle(train_list)

    train_source = get_dataset(
        [seq_dir[i] for i, l in enumerate(label) if l in train_list],
        [vID[i] for i, l in enumerate(label) if l in train_list],
        [label[i] for i, l in enumerate(label) if l in train_list],
        config)

    # val_source = get_dataset(
    #     [seq_dir[i] for i, l in enumerate(label) if l in val_list],
    #     [vID[i] for i, l in enumerate(label) if l in val_list],
    #     [label[i] for i, l in enumerate(label) if l in val_list],
    #     config)

    return train_source, train_list


def load_probe_data(config, dataset_path, pid_shuffle=False):
    seq_dir = list()
    vID = list()
    label = list()
    print("path", dataset_path)
    # load all data from file path

    for seqs in sorted(list(os.listdir(dataset_path))):
        seqs_path = osp.join(dataset_path, seqs)

        # check if there is image?

        if len(os.listdir(seqs_path)) == 0:
            print("no image in ", seqs_path)
            continue
        seq_dir.append([seqs_path])
        int_label = int(0)
        label.append(int_label)
        vID.append(seqs)

    source = get_dataset(seq_dir, vID, label, config)

    return source, label


def load_gallery_data(config, dataset_path, pid_shuffle=False):
    seq_dir = list()
    vID = list()
    label = list()
    print("path", dataset_path)
    # load all data from file path

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)        # 005
        for seqs in sorted(list(os.listdir(label_path))):
            seqs_path = osp.join(label_path, seqs)

            # check if there is image?

            if len(os.listdir(seqs_path)) == 0:
                print("no image in ", seqs_path)
                continue
            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            vID.append(seqs)

    gallery_data = [seq_dir, vID, label]

    return gallery_data


def save_data_to_pickle(data, data_path):
    print("begin save data.")
    with open(data_path, 'wb') as file:
        pickle.dump(data, file)

    print("saving complete!!")


def load_all_data_frome_pickle(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    print("load data complete!!")
    return data


def get_initial(config, train=False):
    print("Initializing data source...")
    train_source, train_list = load_data(config, config.data.dir, config.data.resolution, config.data.name, config.data.pid_num, cache=train)

    val_source = None

    if config.train.validation:
        val_path = "PATH_TO_VALIDATION_SET"
        val_source, val_list = load_data(config, val_path, config.data.resolution, config.data.name, config.data.pid_num, config.data.appendix, cache=train)

    if train:
        data_path = os.path.join(config.data.cache_path, config.data.name+"_"+"train.npy")

        if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
            print("Loading training data from pickle")
            train_source = load_all_data_frome_pickle(data_path)

        else:
            print("Loading training data...")
            train_source.load_all_data()
            save_data_to_pickle(train_source, data_path)

    print("Data initialization complete.")
    return train_source, val_source, train_list

def load_data_test(config, dataset_path, dataset, pid_num, pid_shuffle=False):
    dataset_path = "/home/mzh/dataset/gait_kaggle/test/dataset"
    seq_dir = list()
    vID = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for seqs in sorted(list(os.listdir(label_path))):
            seqs_path = osp.join(label_path, seqs)

            if len(os.listdir(seqs_path)) == 0:
                print("no image in ", seqs_path)
                continue

            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            vID.append(seqs)

    pid_list = sorted(list(set(label)))

    pid_list_test = int_list(pid_list)

    get_gallery = globals().get("get_gallery_"+config.test.gallery_model)
    test_source, test_gallery = get_gallery(pid_list_test, seq_dir, vID, label, config)
    return test_source, test_gallery


def get_gallery_data(test_dataset, test_gallery):
    seq_dir_gallery, vID_gallery, label_gallery = test_gallery[0], test_gallery[1], test_gallery[2]
    data_gallery = list()

    for i in range(len(label_gallery)):
        data_gallery.append(test_dataset.loader(seq_dir_gallery[i]))
    test_gallery = [data_gallery, vID_gallery, label_gallery]

    return test_gallery


def get_initial_test(config, train= False, test= True ):
    print("Initialzing test dataset...")
    # gallery
    gallery_data = load_gallery_data(config, config.test.gallery_dir)

    # probe
    test_probe_source, test_probe_list = load_probe_data(config, config.test.probe_dir)

    data_path = os.path.join(config.data.cache_path, config.data.name + "_" + "test.npy")

    if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
        print("Loading testing data from pickle")
        test_probe_source = load_all_data_frome_pickle(data_path)

    else:
        print("Loading testing data...")
        test_probe_source.load_all_data()
        save_data_to_pickle(test_probe_source, data_path)

    test_gallery = get_gallery_data(test_probe_source, gallery_data)
    #
    print("len probe set = ", len(test_probe_source), ", len gallery set = ", len(test_gallery[2]))
    return test_probe_source,  test_gallery


def find_list_from_path(dataset_path, appendix):
    seq_dir = list()
    date = list()
    label = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for seqs in sorted(list(os.listdir(label_path))):
            if appendix is not None:
                seqs_path = osp.join(label_path, seqs, appendix)
            else:
                seqs_path = osp.join(label_path, seqs)
            seq_dir.append([seqs_path])
            int_label = int(_label)
            label.append(int_label)
            date.append(seqs)

    return seq_dir, date, label


def get_gallery_data_loader(config, train= False, test= True):
    test_source, test_gallery = load_data_test(config, config.data.dir, config.data.name, config.data.pid_num, config.data.appendix)
    print("Loading finte tuning data...")
    test_gallery_dataset = get_dataset(*test_gallery, config)  # use gallery dataset instant to present gallery
    test_gallery_dataset.load_all_data()

    return test_gallery_dataset


def get_initial_test_save(config, train= False, test= True ):
    print("Initialzing test dataset...")
    test_source, test_gallery = load_data_test(config, config.data.dir, config.data.name, config.data.pid_num, config.data.appendix)
    print("Loading testing data...")

    data_path = os.path.join(config.data.cache_path, config.data.name + "_" + "test.npy")

    if os.path.exists(data_path):  # if the pickle file exists, dirctly load data
        print("Loading training data from pickle")
        test_source = load_all_data_frome_pickle(data_path)

    else:
        print("Loading test data from raw...")
        test_source.load_all_data()
        save_data_to_pickle(test_source, data_path)

    test_gallery = get_gallery_data(test_source, test_gallery)

    return test_source, test_gallery



