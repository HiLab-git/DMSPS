import itertools
import os
import random
import re
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
from torch.utils.data import DataLoader
import copy


class BaseDataSets_s2l(Dataset):
    def __init__(self, args, split='train', data_txt = "train.txt", labeled_num=None, transform=None, sup_type="label"):
        self._base_dir = args.data_root_path
        self.num_classes = args.num_classes
        self.patch_size = args.patch_size
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.sample_list = []
        # load data from txt
        data_path = self._base_dir + '/' + data_txt
        with open(data_path, 'r') as f:
            self.sample_list = f.readlines()
        self.sample_list = [item.replace('\n', '').split(",")[0] for item in self.sample_list]
        
        # if len(self.patch_size) == 2:
        #     self.weight = [np.zeros((self.patch_size[0],self.patch_size[1],self.num_classes), \
        #                             dtype = np.float32)] * len(self.sample_list) 
            
        # elif len(self.patch_size) == 3:
        #     self.weight = [np.zeros((self.patch_size[0],self.patch_size[1],self.patch_size[2], self.num_classes), \
        #                             dtype = np.float32)] * len(self.sample_list) 
        
        self.images = defaultdict(dict)
        for idx, case in enumerate(self.sample_list):
            image_name = self.sample_list[idx]
            h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
            img = h5f['image']
            gt = h5f['label']
            label = h5f[self.sup_type]

            self.images[idx]['id'] = case
            self.images[idx]['image'] = np.array(img)
            self.images[idx]['gt'] = np.array(gt)
            self.images[idx][self.sup_type] = np.array(label)
            # print("gt.shape:{}".format(gt.shape))
            if gt.ndim == 2:
                h, w = gt.shape
                self.images[idx]['weight'] = np.zeros((h, w, self.num_classes), dtype=np.float32)
            else:
                h, w, d = gt.shape
                self.images[idx]['weight'] = np.zeros((h, w, d, self.num_classes), dtype=np.float32)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # image_name = self.image_list[idx]
        # h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        # image = h5f['image'][:]
        # gt = h5f['label'][:]
        # label = h5f[self.sup_type][:]
        # if 255 in np.unique(label):
        #     new_label = copy.deepcopy(label)
        #     new_label[label==255] = self.num_classes
        #     label = new_label
        # weight = self.weight[idx]

        # sample = {'image': image, 'label': label.astype(np.uint8), "gt":gt.astype(np.uint8),'weight': weight}
        # if self.split == "train":
        #     sample = self.transform(sample)
        # sample['idx'] = idx
        ###############
        case = self.images[idx]['id']
        image = self.images[idx]['image']
        gt = self.images[idx]['gt']
        label = self.images[idx][self.sup_type]
        if 255 in np.unique(label):
            new_label = copy.deepcopy(label)
            new_label[label==255] = self.num_classes
            label = new_label
        weight = self.images[idx]['weight']
        sample = {'image': image, 'gt': gt.astype(np.uint8), "label": label.astype(np.uint8), 'weight': weight}
        if self.split == "train":
            sample = self.transform(sample)
        sample['id'] = case

        return sample

def random_rot_flip(image, label, gt, weight):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    gt = np.rot90(gt, k)
    weight = np.rot90(weight, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    gt = np.flip(gt, axis=axis).copy()
    weight = np.flip(weight, axis=axis).copy()
    return image, label, gt, weight

def random_rotate(image, label, gt, weight, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
    gt = ndimage.rotate(gt, angle, order=0, reshape=False, mode="constant", cval=0)
    weight = ndimage.rotate(weight, angle, order=0, reshape=False)
    return image, label, gt, weight

class RandomGenerator_s2l(object):
    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes= num_classes

    def __call__(self, sample):
        image, label, gt, weight = sample["image"], sample["label"], sample['gt'], sample['weight']
        if random.random() > 0.5:
            image, label, gt,  weight = random_rot_flip(image, label, gt, weight)
        elif random.random() > 0.5:
            if self.num_classes in np.unique(label):
                image, label, gt, weight = random_rotate(image, label, gt, weight, cval=self.num_classes)
            else:
                image, label, gt, weight = random_rotate(image, label, gt, weight, cval=0)
        
        if len(self.output_size) == 2:
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            weight = zoom(weight, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        if len(self.output_size) == 3:
            x, y, z = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2]/z ), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2]/z), order=0)
            gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2]/z), order=0)
            weight = zoom(weight, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2]/z,  1), order=0)
            
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        weight = torch.from_numpy(weight.astype(np.float32))
        sample = {'image': image, 'label': label, "gt":gt, 'weight': weight}
    
        return sample

class RandomCrop_s2l(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, num_class =8, with_sdf=False):
        self.output_size = output_size
        self.num_class = num_class
        self.with_sdf = with_sdf 

    def __call__(self, sample):
        image, label, gt, weight = sample["image"], sample["label"], sample['gt'], sample['weight']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)

            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.num_class in np.unique(label):
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=self.num_class)
            else:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            gt = np.pad(gt, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            weight = np.pad(weight, [(pw, pw), (ph, ph), (pd, pd), (0,0)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        gt = gt[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        weight = weight[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'gt':gt, 'weight': weight, 'sdf': sdf}
        else:
            return {'image': image, 'label': label, 'gt':gt, 'weight': weight}
        
class ToTensor_s2l(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 
                    'label': torch.from_numpy(sample['label']).long(),
                    'gt': torch.from_numpy(sample['gt']).long(),
                    'weight': torch.from_numpy(sample['weight']),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 
                    'label': torch.from_numpy(sample['label']).long(), 
                    'gt': torch.from_numpy(sample['gt']).long(),
                    'weight': torch.from_numpy(sample['weight'])}
