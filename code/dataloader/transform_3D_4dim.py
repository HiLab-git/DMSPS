import numpy as np
from scipy import ndimage
import json
import torch

class CenterCrop_3D(object): 
    def __init__(self, output_size, num_class = 8):#num_class = organ_num + background
        self.output_size = output_size
        self.num_class = num_class

    def __call__(self, sample):
        image, label, gt  = sample['image'], sample['label'], sample['gt']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            if image.ndim ==3: 
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            elif image.ndim == 4:
                image = np.pad(image, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
                
            if self.num_class in np.unique(label):
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=self.num_class)
            else:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            gt = np.pad(gt, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = label.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if image.ndim ==3:
            image = image[w1:w1 + self.output_size[0], h1:h1 +
                        self.output_size[1], d1:d1 + self.output_size[2]]
        elif image.ndim ==4:
            image = image[:, w1:w1 + self.output_size[0], h1:h1 +
                        self.output_size[1], d1:d1 + self.output_size[2]]
        gt = gt[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label, 'gt':gt}
    
class RandomCrop(object): 
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
        image, label, gt = sample['image'], sample['label'], sample['gt']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            if image.ndim == 3:
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            elif image.ndim == 4:
                image = np.pad(image, [(0,0), (pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
                
            if self.num_class in np.unique(label):
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=self.num_class)
            else:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                            mode='constant', constant_values=0)
            gt = np.pad(gt, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)
                
        (w, h, d) = label.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if image.ndim==3:
            image = image[w1:w1 + self.output_size[0], h1:h1 +
                        self.output_size[1], d1:d1 + self.output_size[2]]
        elif image.ndim ==4:
            image = image[:, w1:w1 + self.output_size[0], h1:h1 +
                        self.output_size[1], d1:d1 + self.output_size[2]]
        gt = gt[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'gt':gt, 'sdf': sdf}
        else:
            return {'image': image, 'label': label, 'gt':gt}

class RandomNoise_3D(object): #3D？
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        if image.ndim==3:
            noise = np.clip(self.sigma * np.random.randn(
                image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
            noise = noise + self.mu
            image = image + noise

        elif image.ndim == 4:
            noise = np.clip(self.sigma * np.random.randn(
                image.shape[1], image.shape[2], image.shape[3]), -2*self.sigma, 2*self.sigma)
            noise = np.tile(noise[None,:,:,:], (4,1,1,1)) #将噪声做了平铺扩展,扩展为与图像同shape的(4,w,h,d)
            noise = noise + self.mu
            image = image + noise
        return {'image': image, 'label': label, 'gt':gt}
    
class CreateOnehotLabel_3D(object): 
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'gt':gt, 'onehot_label': onehot_label}
    
class ToTensor(object): #3D？
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        if image.ndim==3:
            image = image.reshape(
                1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        elif image.ndim == 4:
            image = image.astype(np.float32)

        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'gt': torch.from_numpy(sample['gt']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 
                    'gt': torch.from_numpy(sample['gt']).long()}