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
from scipy import ndimage
from PIL import Image
import random

# from networks.efficientunet import UNet
from networks.net_factory_3d import net_factory_3d
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from test.uttils import calculate_metric_percase, logInference, get_the_first_k_largest_components, get_rgb_from_uncertainty

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model and file parameter
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/Abdomen_word/WORD-V0.1.0-Admin_cropWL_for3D',
                    help='training data root path; subfolders: train_dir, test dir, valid_dir') 
parser.add_argument('--data_type', type=str,
                    default='Abdomen', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='word_3d', help='Data name')
parser.add_argument('--testData', type=str,
                    default='train.txt', help='Data text: train.txt for retrain, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct_dp_3D', help='select mode: unet_cct_dp_3D, \
                        attention_unet_2dual_3d, unetr_2dual_3d')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name') 
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times num:3,4 for uncertainty')

parser.add_argument('--threshold', type=float, default=0.3,
                    help='uncertainty threshold') 
parser.add_argument('--uncertainty_show_path', type=str, default="../../figure/MedIA_WORD/uncertainty_thre03/",
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
    uncertainty = -1.0 * np.sum(preds * np.log(preds + 1e-6), axis=1, keepdims=True)/ np.log(C)
    return uncertainty
  
def pred_single_case_3d_forTrainUncertainty(net, image, stride_xy, stride_z, patch_size, num_classes=1):
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

    score_map_auxi = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    uncertainty_c = np.zeros((ww, hh, dd)).astype(np.float32)
    mask = np.zeros((ww, hh, dd)).astype(np.float32)

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
                    out_main, out_auxi = net(test_patch) #net_type must be 2 decoders
                    out_main_soft, out_auxi_soft = torch.softmax(out_main, dim=1), torch.softmax(out_auxi, dim=1)

                out_main_soft = out_main_soft.cpu().data.numpy()
                out_auxi_soft = out_auxi_soft.cpu().data.numpy()

                out_mean_soft = (out_auxi_soft + out_main_soft)/2.0
                uncertainty_c_p = weight_with_uncertainty_class_np(out_mean_soft , C = FLAGS.num_classes)
                uncertainty_c_p = np.squeeze(np.squeeze(uncertainty_c_p, axis=0), axis=0)

                mask_p = (uncertainty_c_p < FLAGS.threshold)

                uncertainty_c[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]\
                = uncertainty_c[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + uncertainty_c_p
                mask[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]\
                = mask[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + mask_p

                # stack the each patch's prediction into a input-size zero matrix
                out_main_soft = out_main_soft[0, :, :, :, :] 
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + out_main_soft
    
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
                
                out_auxi_soft = out_auxi_soft[0, :, :, :, :]
                score_map_auxi[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map_auxi[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + out_auxi_soft
 
    score_map = score_map/np.expand_dims(cnt, axis=0) 
    prediction_main = np.argmax(score_map, axis=0)

    score_map_auxi = score_map_auxi/np.expand_dims(cnt, axis=0)     
    prediction_auxi  = np.argmax(score_map_auxi, axis=0)

    score_map_mean = (score_map + score_map_auxi) / 2.0
    prediction_mean  = np.argmax(score_map_mean, axis=0)

    uncertainty_c = uncertainty_c / cnt
    mask = mask / cnt
    mask = (mask > 0.5) 

    if add_pad:
        prediction_main = prediction_main[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        prediction_auxi = prediction_auxi[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map_auxi = score_map_auxi[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        uncertainty_c = uncertainty_c[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        mask = mask[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        prediction_mean = prediction_mean[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    

    prediction_cross = prediction_mean
    # multiple mask 
    prediction_cu = prediction_cross * mask
    # get the maximum area 
    prediction_cum = np.zeros_like(prediction_main)
    for lab_ind in range(1, FLAGS.num_classes):
        organ_max = get_the_first_k_largest_components(prediction_cu == lab_ind, 1)
        prediction_cum = np.where(organ_max==1, lab_ind, prediction_cum)

    # for visualization: multiple mask and get the maximum area + unannotated version 
    ## background: 0-->(FLAGS.num_classe or unannotated_index)， unannotated pixel: (FLAGS.num_classe or unannotated_index)-->0
    prediction_cross_unannotated_temp = prediction_cross
    prediction_cross_unannotated_temp[prediction_cross_unannotated_temp == 0] = FLAGS.num_classes
    prediction_cu_unannotated = prediction_cross_unannotated_temp * mask
    prediction_cum_unannotated = np.zeros_like(prediction_main)
    for lab_ind in range(1, FLAGS.num_classes + 1):
        organ_max = get_the_first_k_largest_components(prediction_cu_unannotated == lab_ind, 1)
        prediction_cum_unannotated = np.where(organ_max==1, lab_ind, prediction_cum_unannotated)
    
    # background: (8 or unannotated_index)-->0, unannotated pixel:0-->255
    prediction_cum_unannotated[prediction_cum_unannotated == 0]  = 255
    prediction_cum_unannotated[prediction_cum_unannotated == FLAGS.num_classes]  = 0

    return [prediction_main, prediction_auxi, prediction_mean, \
            prediction_cum, prediction_cum_unannotated, \
            uncertainty_c, mask]

def test_single_volume_3d_WORD(case_path, net, test_save_path, FLAGS):
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))

    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    ################## get the prediction 
    prediction_list = pred_single_case_3d_forTrainUncertainty(
                net, image, FLAGS.stride_xy, FLAGS.stride_z, FLAGS.patch_size, num_classes=FLAGS.num_classes)

    ################## Calculate metrics for each type
    # get the information of original image
    org_img_path = FLAGS.data_root_path.replace("_for3D","")
    if FLAGS.savedir == "TrResult":
        org_img_path = org_img_path + "/imagesTr/{}.nii.gz".format(case_name)
    elif FLAGS.savedir == "ValResult":
        org_img_path = org_img_path + "/imagesVal/{}.nii.gz".format(case_name)
    else:
        org_img_path = org_img_path + "/imagesTs/{}.nii.gz".format(case_name)
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

    ################# for visualization
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

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
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
    snapshot_path = "../../model/{}_{}/{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model)
    net = net_factory_3d(net_type = FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(save_mode_path).items()})
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 


    if not os.path.exists(FLAGS.uncertainty_show_path):
        os.makedirs(FLAGS.uncertainty_show_path)

    metric_onTest_all_list1 = [] 
    metric_onTest_all_list2 = []
    metric_onTest_all_list3 = []
    metric_onTest_all_list4 = []
    for image_path in tqdm(image_list):
        metric_singleEval= test_single_volume_3d_WORD(image_path, net, test_save_path, FLAGS)
        metric_singleEval =  [np.asarray(i) for i in metric_singleEval] 
        metric_onTest_all_list1.append(metric_singleEval[0])
        metric_onTest_all_list2.append(metric_singleEval[1])
        metric_onTest_all_list3.append(metric_singleEval[2])
        metric_onTest_all_list4.append(metric_singleEval[3])
    logging.info("prediction main: ")
    logInference(metric_onTest_all_list1)
    logging.info("prediction auxi: ")
    logInference(metric_onTest_all_list2)   
    logging.info("prediction mean: ")
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

