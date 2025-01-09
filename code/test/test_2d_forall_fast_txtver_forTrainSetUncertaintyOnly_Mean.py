import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR) 

import argparse
import shutil
import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import time 
import logging
from scipy import ndimage
from PIL import Image
import random
import matplotlib.pyplot as plt

from networks.net_factory import net_factory
from test.uttils import calculate_metric_percase, logInference, get_rgb_from_uncertainty, get_the_first_k_largest_components

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/ACDC2017/ACDC_for2D', help='Data root path') 
parser.add_argument('--data_type', type=str,
                    default='Heart', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='ACDC', help='Data name')
parser.add_argument('--testData', type=str,
                    default='trainvol.txt', help='Data text: trainvol.txt for retrain, test.txt, valid.txt')
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')

parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name:unet,unet_cct, etc. the name of pretrained model from stage1')
parser.add_argument('--exp', type=str,
                    default='A3_weakly_PLS_soft_2d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times num:3,4 for uncertainty') 
parser.add_argument('--threshold', type=float, default=0.1,
                    help='uncertainty threshold') 
parser.add_argument('--uncertainty_show_path', type=str, default="../../figure/MedIA_ACDC/uncertainty_thre01/",
                    help='') 
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    
def weight_with_uncertainty_class(preds, C):
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)/ torch.tensor(np.log(C)).cuda()
    return uncertainty

def get_rgb_from_label_ACDC(label):
    h, w = label.shape
    label = label.astype(np.uint8)
    label = Image.fromarray(label)
    output_map = Image.new('RGB', tuple((w,h)), (0, 0, 0))
    for i in range(w):
        for j in range(h):
            p0 = label.getpixel((i, j))
            if p0 == 0: 
                p1 = [0,0,0]
            elif p0 == 1:
                p1 = [255, 0, 0]
            elif p0 ==2: 
                p1 = [0, 255, 0]
            elif p0 ==3:
                p1 = [0, 0, 255]
            elif p0 ==4:
                p1 = [255, 255, 0]
            output_map.putpixel((i,j), tuple(p1))
    return output_map

def convertMap(itk_map): # Convert the background and unannotated, the 5 here is actually class_num + 1
    itk_map[itk_map == 0] = 5
    itk_map[itk_map == 4] = 0 
    itk_map[itk_map == 5] = 4 
    return itk_map 

#test the model and save the prediction segmentation 
def test_single_volume_2d_forTrainUncertainty(case_path, net, test_save_path, FLAGS, uncertainty_path_save):  
    count_num = 0 
    case_name = case_path.split("/")[-1].replace(".h5","")
    logging.info("the testing case: {}".format(case_name))
    
    h5f = h5py.File(case_path, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    scribble = h5f['scribble'][:]

    prediction = np.zeros_like(label)
    prediction_auxi = np.zeros_like(label)
    prediction_mean = np.zeros_like(label)
    prediction_cum = np.zeros_like(label)

    for ind in range(image.shape[0]):

        gt_slice = label[ind, :,:]
        gt_show = get_rgb_from_label_ACDC(gt_slice)
        gt_show.save(uncertainty_path_save + case_name + '_sclice{}_gt.png'.format(ind))

        scri_slice = scribble[ind, :,:]
        scri_show = get_rgb_from_label_ACDC(scri_slice)
        scri_show.save(uncertainty_path_save + case_name + '_sclice{}_scri.png'.format(ind))

        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out_main, out_auxi = net(input)
            out_main_soft, out_auxi_soft = torch.softmax(out_main, dim=1), torch.softmax(out_auxi, dim=1)
            out_mean_soft = (out_main_soft+out_auxi_soft)/2.0

            out_arg = torch.argmax(out_main_soft, dim=1).squeeze(0)
            out_arg = out_arg.cpu().detach().numpy()
            pred_main = zoom(out_arg, (x / 256, y / 256), order=0)
            pred_main_show = get_rgb_from_label_ACDC(pred_main)
            pred_main_show.save(uncertainty_path_save + case_name + '_sclice{}_pred_main.png'.format(ind))
            prediction[ind] = pred_main 

            out_arg_auxi = torch.argmax(out_auxi_soft, dim=1).squeeze(0)
            out_arg_auxi = out_arg_auxi.cpu().detach().numpy()
            pred_auxi = zoom(out_arg_auxi, (x / 256, y / 256), order=0)
            pred_auxi_show = get_rgb_from_label_ACDC(pred_auxi)
            pred_auxi_show.save(uncertainty_path_save + case_name + '_sclice{}_pred_auxi.png'.format(ind))
            prediction_auxi[ind] = pred_auxi 

            out_arg_mean = torch.argmax(out_mean_soft, dim=1).squeeze(0)
            out_arg_mean = out_arg_mean.cpu().detach().numpy()
            pred_mean = zoom(out_arg_mean, (x / 256, y / 256), order=0)
            pred_mean_show = get_rgb_from_label_ACDC(pred_mean)
            pred_mean_show.save(uncertainty_path_save + case_name + '_sclice{}_pred_mean.png'.format(ind))
            prediction_mean[ind] = pred_mean 

            uncertainty_c = weight_with_uncertainty_class(out_mean_soft, FLAGS.num_classes)

            uncertainty_weight = uncertainty_c.cpu().numpy().squeeze(0).squeeze(0)
            uncertainty_weight = zoom(uncertainty_weight, (x / 256, y / 256), order=0) # uncertainty_weight_shape:(256, 216)

            mask = (uncertainty_weight < FLAGS.threshold)

            out_img = get_rgb_from_uncertainty(uncertainty_weight)
            out_img.save(uncertainty_path_save + case_name + '_sclice{}_uncertainty.png'.format(ind))

            mask_img = Image.fromarray(mask)
            mask_img.save(uncertainty_path_save + case_name + '_sclice{}_mask.png'.format(ind))

            pred = pred_mean

            # multiple mask 
            pred_uncertainty = pred * mask
            ## for visualization: 
            pred_unannotated_temp = pred 
            pred_unannotated_temp = convertMap(pred_unannotated_temp) # 0 is unannotated, 4 is background
            pred_uncertainty_unanotated = pred_unannotated_temp * mask
            pred_uncertainty_unanotated = convertMap(pred_uncertainty_unanotated) # 4 is unannotated, 0 is background
            pred_uncertainty_unannotated_show = get_rgb_from_label_ACDC(pred_uncertainty_unanotated)
            pred_uncertainty_unannotated_show.save(uncertainty_path_save + case_name + '_sclice{}_cu_unannotated.png'.format(ind))

            # get the maximum area
            lab_out = np.zeros_like(pred_main) 
            for lab_ind in range(1, FLAGS.num_classes):
                organ_max = get_the_first_k_largest_components(pred_uncertainty == lab_ind, 1)
                lab_out = np.where(organ_max==1, lab_ind, lab_out)
            prediction_cum[ind] = lab_out
            ## for visualization: 
            lab_out_unannotated = np.zeros_like(pred_main)
            for lab_ind in range(1, FLAGS.num_classes + 1):
                organ_max = get_the_first_k_largest_components(pred_uncertainty_unanotated == lab_ind, 1)# input binary map
                lab_out_unannotated = np.where(organ_max==1, lab_ind, lab_out_unannotated)
            lab_out_unannotated_show = convertMap(lab_out_unannotated)
            lab_out_unannotated_show = get_rgb_from_label_ACDC(lab_out_unannotated_show)
            lab_out_unannotated_show.save(uncertainty_path_save + case_name + '_sclice{}_cuM_unannotated.png'.format(ind))

            count_num +=1 

    #save and test
    org_img_path = FLAGS.data_root_path.replace("_for2D","")
    if FLAGS.savedir == "TrResult":
        org_img_path = org_img_path + "/train/images_N/{}.nii.gz".format(case_name)
    elif FLAGS.savedir == "ValResult":
        org_img_path = org_img_path + "/val/images_N/{}.nii.gz".format(case_name)
    else:
        org_img_path = org_img_path + "/TestSet/images_N/{}.nii.gz".format(case_name)
    org_img_itk = sitk.ReadImage(org_img_path)
    spacing = org_img_itk.GetSpacing()

    metric = [0] * (FLAGS.num_classes-1)
    metric_auxi = [0] * (FLAGS.num_classes-1)
    metric_cum = [0] * (FLAGS.num_classes-1)

        #calculate outputs metric from main branch and auxi branch
    for i in range (1,FLAGS.num_classes):
        metric[i-1] = calculate_metric_percase(
        prediction == i, label == i, (spacing[2], spacing[0], spacing[1]))
        
        metric_auxi[i-1] = calculate_metric_percase(
        prediction_auxi == i, label == i, (spacing[2], spacing[0], spacing[1]))
    
        metric_cum[i-1] = calculate_metric_percase(
        prediction_cum == i, label == i, (spacing[2], spacing[0], spacing[1]))

    # Save predicted data
    pred_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    pred_itk.CopyInformation(org_img_itk)
    pred_auxi_itk = sitk.GetImageFromArray(prediction_auxi.astype(np.float32))
    pred_auxi_itk.CopyInformation(org_img_itk)
    # pred_mean_itk = sitk.GetImageFromArray(prediction_mean.astype(np.float32))
    # pred_mean_itk.CopyInformation(org_img_itk)

    pred_cum_itk = sitk.GetImageFromArray(prediction_cum.astype(np.float32))
    pred_cum_itk.CopyInformation(org_img_itk)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.CopyInformation(org_img_itk)
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.CopyInformation(org_img_itk)
    scri_itk = sitk.GetImageFromArray(scribble.astype(np.float32))
    scri_itk.CopyInformation(org_img_itk)

    sitk.WriteImage(pred_itk, test_save_path + "/" + case_name + "_pred1.nii.gz")
    sitk.WriteImage(pred_auxi_itk, test_save_path + "/" + case_name + "_pred2.nii.gz")
    # sitk.WriteImage(pred_mean_itk, test_save_path + "/" + case_name + "_predMean.nii.gz")

    sitk.WriteImage(pred_cum_itk, test_save_path + "/" + case_name + "_pred_cuM.nii.gz")
    
    sitk.WriteImage(img_itk, test_save_path + "/" + case_name + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + "/" + case_name + "_gt.nii.gz")
    sitk.WriteImage(scri_itk, test_save_path + "/" + case_name + "_scri.nii.gz")

    return [metric, metric_auxi, metric_cum] #长度为num_classes - 1
    
def Inference(FLAGS, test_save_path):
    with open(FLAGS.data_root_path + '/{}'.format(FLAGS.testData), 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.data_root_path + "/{}".format(item.replace('\n', '')) for item in image_list]
    logging.info("test volume num:{}".format(len(image_list)))

    #definite net model
    snapshot_path = "../../model/{}_{}/{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model)
    net = net_factory(net_type = FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    #load model
    net.load_state_dict(torch.load(save_mode_path))
    logging.info("init weight from {}".format(save_mode_path))
    net.eval() 

    uncertainty_path_save = FLAGS.uncertainty_show_path
    if not os.path.exists(uncertainty_path_save ):
        os.makedirs(uncertainty_path_save )

    metric_onTest_all_list1 = [] 
    metric_onTest_all_list2 = []
    metric_onTest_all_list3 = []
    for image_path in tqdm(image_list):
        metric_singleEval = test_single_volume_2d_forTrainUncertainty(image_path, net, test_save_path, FLAGS, uncertainty_path_save) 
        metric_singleEval =  [np.asarray(i) for i in metric_singleEval]
        metric_onTest_all_list1.append(metric_singleEval[0])
        metric_onTest_all_list2.append(metric_singleEval[1])
        metric_onTest_all_list3.append(metric_singleEval[2])
    logging.info("prediction main: ")
    logInference(metric_onTest_all_list1)
    logging.info("prediction auxi: ")
    logInference(metric_onTest_all_list2) 
    logging.info("prediction with uncertainty-filtering of threshold {} + postProcessing: ".format(FLAGS.threshold))   
    logInference(metric_onTest_all_list3)    

if __name__ == '__main__':
    start_time = time.time()

    FLAGS = parser.parse_args()

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed(FLAGS.seed)
    
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

