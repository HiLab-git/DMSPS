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
import time #记录时间
import math
import pandas as pd
import logging

# from networks.efficientunet import UNet
from networks.net_factory_3d import net_factory_3d
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from test.uttils import calculate_metric_percase, logInference, get_the_first_k_largest_components, get_rgb_from_uncertainty
from test_3d_forall_fast_txtver import pred_single_case_3d

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model and file parameter
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingPre',
                    help='training data root path') 
parser.add_argument('--data_type', type=str,
                    default='BraTS', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='brats2020_3d', help='Data name')
parser.add_argument('--testData', type=str,
                    default='test.txt', help='Data text: trainvol.txt for retrain, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TsResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct_dropout_3D', help='select mode: unet_3D, ')
parser.add_argument('--exp', type=str,
                    default='T3_weakly_SPS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name') 
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=1,
                    help='test times_num') 

# for test
parser.add_argument('--stride_xy', type=int, default=64,
                    help='stride_xy')  
parser.add_argument('--stride_z', type=int, default=64,
                    help='stride_z') 
parser.add_argument('--patch_size', type=list,  default=[80, 96, 96],
                    help='patch size of network input') 

def test_single_volume_3d_BrsTS(case_path, net, test_save_path, FLAGS):
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))

    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    # get the prediction 
    prediction = pred_single_case_3d(
                net, FLAGS.model, image, FLAGS.stride_xy, FLAGS.stride_z, FLAGS.patch_size, num_classes=FLAGS.num_classes)

    # get the space information of original image
    org_img_path =  FLAGS.data_root_path + "/t1/{}_t1.nii.gz".format(case_name) 

    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    metric_evalIndicatorsOfOrgans = [0] * (FLAGS.num_classes-1)
    
    for i in range (1,FLAGS.num_classes):
        metric_evalIndicatorsOfOrgans[i-1] = calculate_metric_percase(
        prediction == i, label == i, (spacing[2], spacing[0], spacing[1]))

    imgt1_itk = sitk.GetImageFromArray(image[0].astype(np.float32))
    imgt1_itk.CopyInformation(org_img_itk)
    imgt1ce_itk = sitk.GetImageFromArray(image[1].astype(np.float32))
    imgt1ce_itk.CopyInformation(org_img_itk)
    imgt2_itk = sitk.GetImageFromArray(image[2].astype(np.float32))
    imgt2_itk.CopyInformation(org_img_itk)
    imgflair_itk = sitk.GetImageFromArray(image[3].astype(np.float32))
    imgflair_itk.CopyInformation(org_img_itk)

    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    sitk.WriteImage(prd_itk, test_save_path + "/" + case_name + "_pred.nii.gz")
    sitk.WriteImage(imgt1_itk, test_save_path + "/" + case_name + "_img_t1.nii.gz")
    sitk.WriteImage(imgt1ce_itk, test_save_path + "/" + case_name + "_img_t1ce.nii.gz")
    sitk.WriteImage(imgt2_itk, test_save_path + "/" + case_name + "_img_t2.nii.gz")
    sitk.WriteImage(imgflair_itk, test_save_path + "/" + case_name + "_img_flair.nii.gz")
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
    net = net_factory_3d(net_type = FLAGS.model, in_chns=4,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_mode_path).items()})
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 

    metric_onTest_all_list = [] 
    
    for image_path in tqdm(image_list):
        metric_singleEval= test_single_volume_3d_BrsTS(image_path, net, test_save_path, FLAGS) 
        metric_singleEval =  np.asarray(metric_singleEval) 
        metric_onTest_all_list.append(metric_singleEval)
        
    logInference(metric_onTest_all_list)

    return  metric_onTest_all_list

if __name__ == '__main__': 
    FLAGS = parser.parse_args()
    
    test_save_path = "../../result/{}_{}/{}_{}_{}_{}_{}".format(
         FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.fold, FLAGS.savedir, FLAGS.tt_num)

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
    metric = Inference(FLAGS, test_save_path)
    ss =time.time()-start_time
    logging.info("time cost: {}s, i.e.{}h".format(ss,ss/3600))

