# fast distace calculated by 'distance_metrics_fast.py' from luo
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

import argparse
import shutil
import re
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import time 
import math
import pandas as pd
import logging

# from networks.efficientunet import UNet
from networks.net_factory_3d import net_factory_3d
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from test.uttils import calculate_metric_percase, logInference, get_the_first_k_largest_components, get_rgb_from_uncertainty


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model and file parameter
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/Abdomen_word/WORD-V0.1.0-Admin_cropWL_for3D',
                    help='training data root path') 
parser.add_argument('--data_type', type=str,
                    default='Abdomen', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='word_3d', help='Data name')
parser.add_argument('--testData', type=str,
                    default='test.txt', help='Data text: train.txt, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TsResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct_dropout_3D', help='select model: unet_3D, unet_cct_dropout_3D')
parser.add_argument('--exp', type=str,
                    default='W_weakly_SPS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=1,
                    help='test times num, 1: branch 1, 2:branch 2') 
# for test
parser.add_argument('--stride_xy', type=int, default=64,
                    help='stride_xy') 
parser.add_argument('--stride_z', type=int, default=64,
                    help='stride_z') 
parser.add_argument('--patch_size', type=list,  default=[80, 96, 96],
                    help='patch size of network input') 

def pred_single_case_3d(net, net_type, image, stride_xy, stride_z, patch_size, num_classes=1):
    if image.ndim ==3:
        w, h, d = image.shape
    elif image.ndim == 4:
        w, h, d = image.shape[-3:]

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        if image.ndim ==3:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                (dl_pad, dr_pad)], mode='constant', constant_values=0)
        elif image.ndim == 4:
            image = np.pad(image, [(0,0), (wl_pad, wr_pad), (hl_pad, hr_pad),
                                (dl_pad, dr_pad)], mode='constant', constant_values=0)
    if image.ndim ==3:
        ww, hh, dd = image.shape
    elif image.ndim == 4:
        ww, hh, dd = image.shape[-3:]

    # deal the input image into patches to dit the net's input size
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1 
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0]) 
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])

                if image.ndim == 3: 
                    test_patch = image[xs:xs+patch_size[0],
                                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(
                        test_patch, axis=0), axis=0).astype(np.float32)   
                if image.ndim == 4: 
                    test_patch = image[:, xs:xs+patch_size[0],
                                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)

                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    if net_type == "unet_cct_dropout_3D" or net_type == "attention_unet_2dual_3d"\
                        or net_type == "unetr_2dual_3d":
                        y_main, _ = net(test_patch)
                    else:
                        y_main = net(test_patch)
                y = torch.softmax(y_main, dim=1)
                y = y.cpu().data.numpy()

                # stack the each patch's prediction into a input-size zero matrix
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map 

def test_single_volume_3d(case_path, net, test_save_path, FLAGS):
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))

    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    # get the prediction 
    prediction = pred_single_case_3d(
                net, FLAGS.model, image, FLAGS.stride_xy, FLAGS.stride_z, FLAGS.patch_size, num_classes=FLAGS.num_classes)
    
    # get the space information of original image
    org_img_path = FLAGS.data_root_path.replace("_for3D","")
    if FLAGS.savedir == "TrResult":
        org_img_path = org_img_path + "/imagesTr/{}.nii.gz".format(case_name)
    elif FLAGS.savedir == "ValResult":
        org_img_path = org_img_path + "/imagesVal/{}.nii.gz".format(case_name)
    else:
        org_img_path = org_img_path + "/imagesTs/{}.nii.gz".format(case_name)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    metric_evalIndicatorsOfOrgans = [0] * (FLAGS.num_classes-1)
    
    for i in range (1,FLAGS.num_classes):
        metric_evalIndicatorsOfOrgans[i-1] = calculate_metric_percase(
        prediction == i, label == i, (spacing[2], spacing[0], spacing[1]))

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + "/" + case_name + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + "/" + case_name + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + "/" + case_name + "_gt.nii.gz")
    return metric_evalIndicatorsOfOrgans 

def Inference(FLAGS, test_save_path):
    with open(FLAGS.data_root_path + '/{}'.format(FLAGS.testData), 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.data_root_path + "/{}".format(item.replace('\n', '')) for item in image_list]
    logging.info("test volume num:{}".format(len(image_list)))
    
    ####definite and load net model
    snapshot_path = "../../model/{}_{}/{}_{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.fold)
    net = net_factory_3d(net_type = FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_mode_path).items()})
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 

    metric_onTest_all_list = [] 
    for image_path in tqdm(image_list):
        metric_singleEval= test_single_volume_3d(image_path, net, test_save_path, FLAGS) 
        metric_singleEval =  np.asarray(metric_singleEval) 
        metric_onTest_all_list.append(metric_singleEval)
        
    logInference(metric_onTest_all_list)

    return metric_onTest_all_list

if __name__ == '__main__': 
    
    FLAGS = parser.parse_args()
    
    test_save_path = "../../result/{}_{}/{}_{}_{}_{}".format(
         FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.savedir, FLAGS.tt_num)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    os.makedirs(test_save_path + "/log")

    logging.basicConfig(filename=test_save_path+"/log/test"+str(FLAGS.tt_num)+"_info.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', 
                       )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(str(FLAGS))
    start_time = time.time()
    Metric = Inference(FLAGS, test_save_path)
    ss =time.time()-start_time
    logging.info("time cost: {}s, i.e.{}h".format(ss,ss/3600))

