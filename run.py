# -*- coding: utf-8 -*-
'''`
Code for the following paper:
  https://www.sciencedirect.com/science/article/abs/pii/S1361841524001993?dgcid
    @article{han2024dmsps,
    title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
    author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
    journal={Medical Image Analysis},
    pages={103274},
    year={2024},
    publisher={Elsevier}
}

'''

from __future__ import print_function, division
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from pymic.util.parse_config import *
from pymic.transform.trans_dict import TransformDict
from pymic.net.net_dict_seg import SegNetDict
from pymic.net.net2d.unet2d import Encoder, Decoder
from pymic.net_run.weak_sup import  WSLDMSPS


def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout2d(x, p)
    return x

class DBNet(nn.Module): 
    def __init__(self, params):
        super(DBNet, self).__init__()
        self.encoder = Encoder(params)
        self.main_decoder = Decoder(params)
        self.aux_decoder1 = Decoder(params) 

    def forward(self, x): 
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        feature = self.encoder(x)
        main_out = self.main_decoder(feature)
        aux_feature = [Dropout(i) for i in feature]
        aux_out1 = self.aux_decoder1(aux_feature)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(main_out.shape)[1:]
            main_out = torch.reshape(main_out, new_shape)
            main_out = torch.transpose(main_out, 1, 2)
            aux_out1 = torch.reshape(aux_out1, new_shape)
            aux_out1 = torch.transpose(aux_out1, 1, 2)
        
        return main_out, aux_out1

class CustomTransform(object): 
    def __init__(self, param):
        self.output_size = param["CustomTransform_output_size".lower()]

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, (-2, -1))
            label = np.rot90(label, k, (-2, -1))
            axis  = np.random.randint(-2, -1)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
        elif random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, axes=(-2, -1), order=0, reshape=False)
            label = ndimage.rotate(label, angle, axes=(-2, -1), order=0, reshape=False, mode="constant", cval=4)
            
        img_shape = image.shape
        zoom_f = [1.0, self.output_size[0]/img_shape[-2], self.output_size[1]/img_shape[-1]]
        image = ndimage.zoom(image, zoom_f, order=0)
        label = ndimage.zoom(label, zoom_f, order=0)
        sample = {"image": image, "label": label, "gt":label}
        return sample

def main():
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 3. e.g.')
        print('   python run.py train config.cfg')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type = str, help="stage of train or test")
    parser.add_argument("cfg", type = str, help="configuration file")
    parser.add_argument("--test_csv", type = str,  required=False, help="the csv file for testing images",
        default='./data/ACDC2017/ACDC_for2D/test.csv')
    parser.add_argument("--dmsps_test_mode", type = str, required=False, default='0',
        help="mode for testing. 0: obtain segmentation label map from the main decoder;" + 
        "1: obtain the uncertainty map and expanded seeds based on the two decoders.")    
    parser.add_argument("--output_dir", type = str, required=False, help="the path of output dir",
        default=None)    
    args = parser.parse_args()
    if(not os.path.isfile(args.cfg)):
        raise ValueError("The config file does not exist: " + args.cfg)
    config   = parse_config(args)
    config   = synchronize_config(config)

    log_dir  = config['training']['ckpt_dir']
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
    if sys.version.startswith("3.9"):
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s', force=True) # for python 3.9
    else:
        logging.basicConfig(filename=log_dir+"/log_{0:}.txt".format(args.stage), level=logging.INFO,
                            format='%(message)s') # for python 3.6
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)

    agent  = WSLDMSPS(config, args.stage)
    trans_dict = {"CustomTransform":CustomTransform}
    trans_dict.update(TransformDict)
    net_dict  = {"DBNet": DBNet}
    net_dict.update(SegNetDict)
    agent.set_transform_dict(trans_dict)
    agent.set_net_dict(net_dict)
    agent.run()

if __name__ == "__main__":
    main()
