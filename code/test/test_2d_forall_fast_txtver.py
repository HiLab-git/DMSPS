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
import logging
import pandas as pd

from networks.net_factory import net_factory
from utils.distance_metrics_fast import hd95_fast, asd_fast, assd_fast
from test.uttils import calculate_metric_percase, logInference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/ACDC2017/ACDC_for2D', help='Data root path') 
parser.add_argument('--data_type', type=str,
                    default='Heart', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='ACDC', help='Data name')
parser.add_argument('--testData', type=str,
                    default='test.txt', help='Data text: trainvol.txt for retrain, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TsResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name:unet,unet_cct')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_2d', help='Experiment name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network') 
parser.add_argument('--patch_size', type=list,  default=[256,256],
                    help='patch size of network input. For swinunet, the patch size should be [224,224]')
parser.add_argument('--tt_num', type=int, default=1,
                    help='test times num, 1: branch 1, 2:branch 2')

#test the model and save the prediction segmentation 
def test_single_volume_2d(case_path, net, test_save_path, FLAGS):   
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))
    
    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0) 
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_cct" or FLAGS.model == "unet_2dual":
                if FLAGS.tt_num == 2:
                    _, out_main = net(input) #d2
                else:
                    out_main, _ = net(input)#d1
            elif FLAGS.model == "unet": 
                out_main = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            # out = torch.argmax((out_aux1_soft+out_aux2_soft)/2.0, dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction[ind] = pred 
        
    org_img_path = FLAGS.data_root_path.replace("_for2D","")
    if FLAGS.savedir == "TrResult":
        org_img_path = org_img_path + "/train/images_N/{}.nii.gz".format(case_name)
    elif FLAGS.savedir == "ValResult":
        org_img_path = org_img_path + "/val/images_N/{}.nii.gz".format(case_name)
    else:
        org_img_path = org_img_path + "/TestSet/images_N/{}.nii.gz".format(case_name)

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

    #definite and load net model
    snapshot_path = "../../model/{}_{}/{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model)
    net = net_factory(net_type = FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path))
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 
   
    metric_onTest_all_list = [] 
    
    for image_path in tqdm(image_list):
        metric_singleEval= test_single_volume_2d(image_path, net, test_save_path, FLAGS) 
        metric_singleEval =  np.asarray(metric_singleEval) 
        metric_onTest_all_list.append(metric_singleEval)
    
    logInference(metric_onTest_all_list)

if __name__ == '__main__':
    start_time = time.time()
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
    Inference(FLAGS, test_save_path)
    ss =time.time()-start_time
    logging.info("time cost: {}s, i.e.{}min".format(ss,ss/60))

