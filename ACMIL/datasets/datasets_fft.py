import random

import h5py
import numpy as np
import torch
from timm.utils import accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
from PIL import Image
import sys
import json

def split_dataset_bracs(file_path, conf, quick):
    csv_path = './dataset_csv/bracs_clam.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')
    class_transfer_dict_3class = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2}
    class_transfer_dict_2class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}

    file_path = '/data/fftmil/BRACS/features_CLAM/h5_files/features_CLAM.h5'
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for i, slide_id in enumerate(slide_names):
        if quick:
            if len(test_split) >= 10:
                break
        if 'BRACS' not in slide_id:
            continue

        slide = h5_data[slide_id]
        label = slide.attrs['label']
        feat = slide.attrs['feats']
        fft = conf.data_dir2 + feat.split('/')[-1].split('.')[0] + '.npz'
        coords = 0

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_luad(file_path, conf, quick):
    csv_path = './dataset_csv/luad_clam.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')

    file_path = '/data/fftmil/LUAD/features_CLAM/h5_files/features_CLAM.h5'
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for i, slide_id in enumerate(slide_names):
        slide = h5_data[slide_id]
        if conf.fft:
            if slide_id == 'C3L-03717-21':
                continue

        label = slide.attrs['label']

        feat = slide.attrs['feats']
        fft = conf.data_dir2 + feat.split('/')[-1].split('.')[0] + '.npz'
        coords = 0

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_imp(file_path, conf, quick):
    csv_path = './dataset_csv/imp_clam.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')

    file_path = '/data/fftmil/IMP/features_CLAM/h5_files/features_CLAM.h5'
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for i, slide_id in enumerate(slide_names):
        slide = h5_data[slide_id]

        label = slide.attrs['label']

        feat = slide.attrs['feats']
        fft = conf.data_dir2 + feat.split('/')[-1].split('.')[0] + '.npz'
        coords = 0

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_luad1(file_path, conf, quick):
    csv_path = './dataset_csv/luad_clam.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')

    file_path = '/data/fftmil/LUAD/features_CLAM/h5_files/features_CLAM.h5'
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for i, slide_id in enumerate(slide_names):
        slide = h5_data[slide_id]
        if conf.fft:
            if slide_id == 'C3L-02165-24':
                continue

        label = slide.attrs['label']

        feat = slide.attrs['feats']
        fft = conf.data_dir2 + feat.split('/')[-1].split('.')[0] + '.npz'
        coords = 0

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_cam16(file_path, conf, quick):
    csv_path = './dataset_csv/cam16_clam.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')

    file_path = '/data/fftmil/CAMELYON16/features_CLAM/h5_files/features_CLAM.h5'
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for i, slide_id in enumerate(slide_names):
        slide = h5_data[slide_id]

        label = slide.attrs['label']

        feat = slide.attrs['feats']
        fft = conf.data_dir2 + feat.split('/')[-1].split('.')[0] + '.npz'
        coords = 0

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label, 'fft': fft}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


class HDF5_feat_dataset2(object):
    def __init__(self, data_dict, data_names):
        self.data_dict = data_dict
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """

        return self.data_dict[self.data_names[index]]

class HDF5_feat_dataset3(object):
    def __init__(self, file_path, data_names):
        self.data_names = data_names
        self.file_path = file_path

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """

        return torch.load(os.path.join(self.file_path, self.data_names[index]+ '.pth'))




def generate_fewshot_dataset(train_split, train_names, num_shots):
    if num_shots < len(train_names) and num_shots > 0:
        labels = [it['label'] for it in train_split.values()]
        train_split_ = {}
        train_names_ = []
        for l in set(labels):
            indices = [index for index, element in enumerate(labels) if element == l]
            selected_indices = random.sample(indices, num_shots)
            names = [train_names[index] for index in selected_indices]
            train_names_ += names
            split = {name: train_split[name] for name in names}
            train_split_.update(split)
        return train_split_, train_names_
    else:
        return train_split, train_names


def build_HDF5_feat_dataset(file_path, conf, quick=False):
    if conf.dataset == 'bracs':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_bracs(file_path, conf, quick)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'luad':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_luad(file_path, conf, quick)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'imp':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_imp(file_path, conf, quick)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'luad1':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_luad1(file_path, conf, quick)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'cam16':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_cam16(file_path, conf, quick)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)







if __name__ == '__main__':
    split_dataset_lct('/mnt/Xsky/zyl/dataset/lct/roi_feats_backbone_Resnet50_pretrain_natural_supervised')
