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
from scipy import ndimage
from PIL import Image
import random

from networks.net_factory_3d import net_factory_3d
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from networks.net_factory_3d import net_factory_3d
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from test.uttils import calculate_metric_percase, logInference, get_the_first_k_largest_components, get_rgb_from_uncertainty
from test_3d_forall_fast_txtver_forTrainSetUncertaintyOnly_Mean import pred_single_case_3d_forTrainUncertainty
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
                    default='train.txt', help='Data text: train.txt for retrain, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct_dropout_3D', help='select mode: unet_cct_dp_3D, \
                        attention_unet_2dual_3d, unetr_2dual_3d ')
parser.add_argument('--exp', type=str,
                    default='T_weakly_SPS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name') 
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times_num')

parser.add_argument('--threshold', type=float, default=0.4,
                    help='uncertainty threshold') 
parser.add_argument('--uncertainty_show_path', type=str, default="../../figure/MedIA_BraTs2020/uncertaintyOnly_thre04/",
                    help='') 
parser.add_argument('--seed', type=int,  default=2022, help='random seed')

# for test
parser.add_argument('--stride_xy', type=int, default=48,
                    help='stride_xy') 
parser.add_argument('--stride_z', type=int, default=40,
                    help='stride_z') 
parser.add_argument('--patch_size', type=list,  default=[80, 96, 96],
                    help='patch size of network input') 

FLAGS = parser.parse_args()

def weight_with_uncertainty_class_np(preds, C):#pred:numpy, [n, c, d, h, w]
    uncertainty = -1.0 * np.sum(preds * np.log(preds + 1e-16), axis=1, keepdims=True)/ np.log(C)
    return uncertainty

def calculate_metric_percase(pred, gt, spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = asd_fast(pred, gt, voxelspacing=spacing)
        hd95 = hd95_fast(pred, gt, voxelspacing=spacing)
        # asd = metric.binary.asd(pred, gt, voxelspacing=spacing)
        # hd95 = metric.binary.hd95(pred, gt, voxelspacing=spacing)
        return np.array([dice, asd, hd95])
    else:
        return np.array([0,0,0])

def test_single_volume_3d_braTS(case_path, net, test_save_path, FLAGS):
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))

    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    # get the prediction 
    prediction_list = pred_single_case_3d_forTrainUncertainty(
                net, image, FLAGS.stride_xy, FLAGS.stride_z, FLAGS.patch_size, num_classes=FLAGS.num_classes, flags = FLAGS.flags)

    # get the information of original image
    org_img_path =  FLAGS.data_root_path + "/t1/{}_t1.nii.gz".format(case_name) 
    # Use t1 to obtain information. In fact, t1, t1ce, t2, flair, etc. are all the same 

    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    metric = [0] * (FLAGS.num_classes-1)
    metric_auxi = [0] * (FLAGS.num_classes-1)
    metric_mean = [0] * (FLAGS.num_classes-1)
    metric_cum = [0] * (FLAGS.num_classes-1)
    
    for i in range (1,FLAGS.num_classes):
        metric[i-1] = calculate_metric_percase(
        prediction_list[0] == i, label == i, (spacing[2], spacing[0], spacing[1]))
        
        metric_auxi[i-1] = calculate_metric_percase(
        prediction_list[1] == i, label == i, (spacing[2], spacing[0], spacing[1]))

        metric_mean[i-1] = calculate_metric_percase(
        prediction_list[2] == i, label == i, (spacing[2], spacing[0], spacing[1]))
        
        metric_cum[i-1] = calculate_metric_percase(
        prediction_list[3] == i, label == i, (spacing[2], spacing[0], spacing[1]))

    uncertainty_c = prediction_list[5]
    unMask = prediction_list[6]

    uncertainty_c_show = uncertainty_c[60,:,:]
    unMask_show = unMask[60,:,:]

    out_img = get_rgb_from_uncertainty(uncertainty_c_show)
    out_img.save(FLAGS.uncertainty_show_path + case_name + '_uncertainty.png')
    mask_img = Image.fromarray(unMask_show)
    mask_img.save(FLAGS.uncertainty_show_path + case_name + '_mask.png')

    ############### save in .itk form
    pred_itk = sitk.GetImageFromArray(prediction_list[0].astype(np.float32))
    pred_itk.CopyInformation(org_img_itk)
    pred_auxi_itk = sitk.GetImageFromArray(prediction_list[1].astype(np.float32))
    pred_auxi_itk.CopyInformation(org_img_itk)
    pred_mean_itk = sitk.GetImageFromArray(prediction_list[2].astype(np.float32))
    pred_mean_itk.CopyInformation(org_img_itk)

    pred_cum_itk = sitk.GetImageFromArray(prediction_list[3].astype(np.float32))
    pred_cum_itk.CopyInformation(org_img_itk)
    pred_cum_itk_unannotated = sitk.GetImageFromArray(prediction_list[4].astype(np.float32))
    pred_cum_itk_unannotated.CopyInformation(org_img_itk)

    # uncertainty_map_itk = sitk.GetImageFromArray(prediction_list[5].astype(np.float32))
    # uncertainty_map_itk.CopyInformation(org_img_itk)
    # mask_map_itk = sitk.GetImageFromArray(prediction_list[6].astype(np.float32))
    # mask_map_itk.CopyInformation(org_img_itk)

    image = image.transpose(1,2,3,0) #image:(134, 169, 129, 4)
    img_itk = sitk.GetImageFromArray(image.astype(np.float32)) #img_itk_size:(129, 169, 134)
    img_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)

    sitk.WriteImage(pred_itk, test_save_path + "/" + case_name + "_pred1.nii.gz")
    sitk.WriteImage(pred_auxi_itk, test_save_path + "/" + case_name + "_pred2.nii.gz")
    sitk.WriteImage(pred_mean_itk, test_save_path + "/" + case_name + "_pred_mean.nii.gz")
    sitk.WriteImage(pred_cum_itk, test_save_path + "/" + case_name + "_pred_cuM.nii.gz")
    sitk.WriteImage(pred_cum_itk_unannotated, test_save_path + "/" + case_name + "_pred_cuM_unannotated.nii.gz")

    # sitk.WriteImage(uncertainty_map_itk, test_save_path + "/" + case_name + "_uncerty.nii.gz")
    # sitk.WriteImage(mask_map_itk, test_save_path + "/" + case_name + "_unMask.nii.gz")

    sitk.WriteImage(img_itk, test_save_path + "/" + case_name + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + "/" + case_name + "_gt.nii.gz")

    return [metric, metric_auxi, metric_mean, metric_cum] 

def Inference(FLAGS, test_save_path):
    with open(FLAGS.data_root_path + '/{}'.format(FLAGS.testData), 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.data_root_path + "/{}".format(item.replace('\n', '')) for item in image_list]
    logging.info("test volume num:{}".format(len(image_list)))
    
    #definite net model
    snapshot_path = "../../model/{}_{}/{}_{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.fold)
    net = net_factory_3d(net_type = FLAGS.model, in_chns=4,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_mode_path).items()})
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 

    metric_onTest_all_list1 = [] 
    metric_onTest_all_list2 = []
    metric_onTest_all_list3 = []
    metric_onTest_all_list4 = []

    if not os.path.exists(FLAGS.uncertainty_show_path):
        os.makedirs(FLAGS.uncertainty_show_path)
    
    for image_path in tqdm(image_list):
        metric_singleEval= test_single_volume_3d_braTS(image_path, net, test_save_path, FLAGS)
        metric_singleEval =  [np.asarray(i) for i in metric_singleEval] 
        metric_onTest_all_list1.append(metric_singleEval[0])
        metric_onTest_all_list2.append(metric_singleEval[1])
        metric_onTest_all_list3.append(metric_singleEval[2])
        metric_onTest_all_list4.append(metric_singleEval[3])
    logging.info("prediction main: ")
    logInference(metric_onTest_all_list1)
    logging.info("prediction auxi: ")
    logInference(metric_onTest_all_list2)   
    logging.info("prediction mean: ".format(FLAGS.threshold))  
    logInference(metric_onTest_all_list3)  
    logging.info("prediction with uncertainty-filtering of threshold {} + postProcessing: ".format(FLAGS.threshold))   
    logInference(metric_onTest_all_list4)   
        
if __name__ == '__main__': 
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)

    
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
    Inference(FLAGS, test_save_path)
    ss =time.time()-start_time
    logging.info("time cost: {}s, i.e.{}h".format(ss,ss/3600))

