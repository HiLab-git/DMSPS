import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetConv3_dropout, UnetConv3_dropout1, \
                           UnetUp3, UnetUp3_CT, \
                           UnetUp3_CT_HM, UnetUp3_CT_HM1

def Dropout(x, p=0.5):
    x = torch.nn.functional.dropout3d(x, p)
    return x


class Encoder_dp(nn.Module): 
    def __init__(self, params):
        super(Encoder_dp, self).__init__()
        self.params = params
        self.in_channels = self.params['in_chns'] 
        self.feature_scale = self.params['feature_scale']
        self.n_class = self.params['class_num']
        self.is_deconv = self.params['is_deconv']
        self.is_batchnorm = self.params['is_batchnorm']
        self.dropout_p = self.params['dropout']

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = UnetConv3_dropout(self.in_channels, filters[0], self.dropout_p[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3_dropout(filters[0], filters[1], self.dropout_p[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3_dropout(filters[1], filters[2], self.dropout_p[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3_dropout(filters[2], filters[3], self.dropout_p[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3_dropout(filters[3], filters[4], self.dropout_p[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        for m in self.modules(): 
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        
        maxpool4 = self.maxpool4(conv4)
        center = self.center(maxpool4)

        return [conv1, conv2, conv3, conv4, center]

class Decoder_dp(nn.Module): 
    def __init__(self, params):
        super(Decoder_dp, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.feature_scale = self.params['feature_scale']
        self.is_batchnorm = self.params['is_batchnorm']
        self.n_class = self.params['class_num']

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.up_concat4 = UnetUp3_CT_HM(filters[4], filters[3], 0.0, self.is_batchnorm)
        self.up_concat3 = UnetUp3_CT_HM(filters[3], filters[2], 0.0, self.is_batchnorm)
        self.up_concat2 = UnetUp3_CT_HM(filters[2], filters[1], 0.0, self.is_batchnorm)
        self.up_concat1 = UnetUp3_CT_HM(filters[1], filters[0], 0.0, self.is_batchnorm) 

        self.final = nn.Conv3d(filters[0], self.n_class, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, feature):
        conv1 = feature[0]
        conv2 = feature[1]
        conv3 = feature[2]
        conv4 = feature[3]
        center = feature[4]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        output = self.final(up1)
        return output

class UNet_CCT_dp_3D(nn.Module): 
    def __init__(self, in_channels=1, n_classes=8, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_CCT_dp_3D, self).__init__()

        params = {'in_chns':in_channels,
                  'class_num': n_classes,
                  'feature_scale': feature_scale, 
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'is_deconv': is_deconv,
                  'is_batchnorm': is_batchnorm,
                  'acti_func': 'relu'} 
        self.encoder = Encoder_dp(params)
        self.main_decoder = Decoder_dp(params)
        self.aux_decoder1 = Decoder_dp(params) 

    def forward(self, x): 
        feature = self.encoder(x)
        main_seg = self.main_decoder(feature)
        aux1_feature = [Dropout(i) for i in feature]
        aux_seg1 = self.aux_decoder1(aux1_feature)
        return main_seg, aux_seg1



