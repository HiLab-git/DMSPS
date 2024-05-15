import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import copy
from scipy import ndimage
import json
import random
from scipy.ndimage.interpolation import zoom

from dataloader.RW4Scribble import pseudo_label_generator_abdomen, pseudo_label_generator_acdc
from dataloader.Graphcuts4Scribble import graph_cut_for_ACDC_slice

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', data_txt = "train.txt", labeled_num=None, transform=None, sup_type="label", num_classes = 8):
        self._base_dir = base_dir
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.num_classes = num_classes
        self.sample_list = []

        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]

        if labeled_num is not None: #for semi
            self.image_list = self.image_list[:labeled_num]
        print("total {} samples for {}".format(len(self.image_list), self.split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        if self.split == "train":
            image = h5f['image'][:]
            gt = h5f['label'][:]
            if self.sup_type=="random_walker":
                if self.data_name == "WORD":
                    label = pseudo_label_generator_abdomen(image, h5f["scribble"][:]) 
                elif self.data_name == "ACDC":
                    label = pseudo_label_generator_acdc(image, h5f["scribble"][:], beta=100, mode='bf') 
            elif self.sup_type == "graphcut":
                label = graph_cut_for_ACDC_slice(image, h5f["scribble"][:])
            else:
                label = h5f[self.sup_type][:]
                if 255 in np.unique(label):
                    new_label = copy.deepcopy(label)
                    new_label[label==255] = self.num_classes
                    label = new_label
            sample = {'image': image, 'label': label.astype(np.uint8), "gt":gt.astype(np.uint8)}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            gt = h5f['label'][:]
            sample = {'image': image, 'label': label.astype(np.uint8), "gt":gt.astype(np.uint8)}
            
        sample["idx"] = idx
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)