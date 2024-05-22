import numpy as np
from scipy import ndimage
import torch
import json
from scipy.ndimage.interpolation import zoom
import random

class RandomCrop_2D(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, num_class = 8, with_sdf=False):
        self.output_size = output_size
        self.num_class = num_class
        self.with_sdf = with_sdf # 
    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)

            image = np.pad(image, [(pw, pw), (ph, ph),],
                           mode='constant', constant_values=0)
            if self.num_class in np.unique(label):
                label = np.pad(label, [(pw, pw), (ph, ph)],
                            mode='constant', constant_values=self.num_class)
            else:
                label = np.pad(label, [(pw, pw), (ph, ph)],
                            mode='constant', constant_values=0)
            gt = np.pad(gt, [(pw, pw), (ph, ph)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph)],
                             mode='constant', constant_values=0)

        (w, h) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1]]
        gt = gt[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1]]

        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1]]
            return {'image': image, 'label': label, 'gt':gt, 'sdf': sdf}
        else:
            return {'image': image, 'label': label, 'gt':gt}

class RandomRotFlip_2D(object):#random90 and flip
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        gt = np.rot90(gt, k)

        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        gt = np.flip(gt, axis=axis).copy()
        return {'image': image, 'label': label, 'gt':gt}

class RandomRotate_2D(object):
    def __init__(self, cval=8):
        self.cval = cval
    def __call__(self, sample):
        image, label, gt = sample['image'], sample['label'], sample['gt']
        angle = np.random.randint(-20,20)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=self.cval)
        gt = ndimage.rotate(gt, angle, order=0, reshape=False, mode="constant", cval=0)
        return {'image': image, 'label': label, 'gt':gt}

def random_rot_flip(image, label,gt):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    gt = np.rot90(gt, k)

    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    gt = np.flip(gt, axis=axis).copy()
    return image, label, gt


def random_rotate(image, label, gt, cval): #2d
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False, mode="constant", cval=cval)
    gt = ndimage.rotate(gt, angle, order=0, reshape=False, mode="constant", cval=0)
    return image, label, gt

class RandomGenerator(object): #2d
    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes= num_classes

    def __call__(self, sample):
        image, label, gt = sample["image"], sample["label"], sample['gt']
        if random.random() > 0.5:
            image, label, gt = random_rot_flip(image, label, gt)
        elif random.random() > 0.5:
            if self.num_classes in np.unique(label):
                image, label, gt = random_rotate(image, label,gt, cval=self.num_classes)
            else:
                image, label, gt = random_rotate(image, label,gt, cval=0)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        sample = {"image": image, "label": label, "gt":gt}
        return sample
    
class RandomGenerator4Abdomen(object):
    def __init__(self, output_size, num_classes):
        self.output_size = output_size
        self.num_classes= num_classes

    def __call__(self, sample):
        image, label, gt = sample["image"], sample["label"], sample['gt']
        # shape: image:(256, 184), label:(256, 184), gt:(256, 184)
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]

        # if random.random() > 0.5:
        #     image, label, gt = RandomRotFlip(sample)
        # elif random.random() > 0.5:
        #     image, label, gt =RandomRotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        gt = zoom(gt, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        gt = torch.from_numpy(gt.astype(np.uint8))
        sample = {"image": image, "label": label, "gt":gt}
        return sample