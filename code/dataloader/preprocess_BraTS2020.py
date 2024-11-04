import sys
import os
import SimpleITK as sitk
import numpy as np
from numpy import *
import logging
import glob
from collections import defaultdict 
import pandas as pd
import h5py
import copy
import random
import math
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_root = "/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData"
predata_root = "/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingPre"
img_dirs = ["t1", "t1ce", "t2", "flair"]

def convert_BraTSLabel_into2():
    # img_names_root  = sorted(glob.glob(data_root +"BraTS20*"))
    # print(img_names_root)
    seg_data_path = sorted(glob.glob(data_root + "/*_seg.nii.gz" ))#369 cases
    # print(len(seg_data_path))

    for item in seg_data_path:
        seg_obj = sitk.ReadImage(item)
        seg_np = sitk.GetArrayFromImage(seg_obj)
        new_label = copy.deepcopy(seg_np)
        new_label[new_label==4] = 1
        print(np.unique(new_label))

        new_label_save_name = item.replace("_seg", "_2label")
        lab_out_obj = sitk.GetImageFromArray(new_label)
        lab_out_obj.SetSpacing(seg_obj.GetSpacing())
        lab_out_obj.SetOrigin(seg_obj.GetOrigin())
        lab_out_obj.SetDirection(seg_obj.GetDirection())
        sitk.WriteImage(lab_out_obj, new_label_save_name)

def brain_bbox(data, gt):
    mask = (data != 0)
    brain_voxels = np.where(mask != 0)
    # print("brain_voxels[0]:{}, num:{}".format(brain_voxels[0], len(brain_voxels[0])))
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    data_bboxed = data[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    return data_bboxed, gt_bboxed

def get_boundingBox_id(data):
    mask = (data != 0)
    brain_voxels = np.where(mask != 0)
    minZidx = int(np.min(brain_voxels[0]))
    maxZidx = int(np.max(brain_voxels[0]))
    minXidx = int(np.min(brain_voxels[1]))
    maxXidx = int(np.max(brain_voxels[1]))
    minYidx = int(np.min(brain_voxels[2]))
    maxYidx = int(np.max(brain_voxels[2]))
    return minZidx, maxZidx, minXidx, maxXidx, minYidx, maxYidx

def brain_bbox_4modalData(data_t1, data_t1ce, data_t2, data_flair, gt):
    minZidx_t1, maxZidx_t1, minXidx_t1, \
        maxXidx_t1, minYidx_t1, maxYidx_t1 = get_boundingBox_id(data_t1)
    minZidx_t1ce, maxZidx_t1ce, minXidx_t1ce, \
        maxXidx_t1ce, minYidx_t1ce, maxYidx_t1ce = get_boundingBox_id(data_t1ce)
    minZidx_t2, maxZidx_t2, minXidx_t2, \
        maxXidx_t2, minYidx_t2, maxYidx_t2 = get_boundingBox_id(data_t2)
    minZidx_flair, maxZidx_flair, minXidx_flair, \
        maxXidx_flair, minYidx_flair, maxYidx_flair = get_boundingBox_id(data_flair)
    minZidx = int(np.min([minZidx_t1, minZidx_t1ce, minZidx_t2, minZidx_flair]))
    maxZidx = int(np.max([maxZidx_t1, maxZidx_t1ce, maxZidx_t2, maxZidx_flair]))
    minXidx = int(np.min([minXidx_t1, minXidx_t1ce, minXidx_t2, minXidx_flair]))
    maxXidx = int(np.max([maxXidx_t1, maxXidx_t1ce, maxXidx_t2, maxXidx_flair]))
    minYidx = int(np.min([minYidx_t1, minYidx_t1ce, minYidx_t2, minYidx_flair]))
    maxYidx = int(np.max([maxYidx_t1, maxYidx_t1ce, maxYidx_t2, maxYidx_flair]))
    data_t1_bboxed = data_t1[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    data_t1ce_bboxed = data_t1ce[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    data_t2_bboxed = data_t2[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    data_flair_bboxed = data_flair[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    gt_bboxed = gt[minZidx:maxZidx, minXidx:maxXidx, minYidx:maxYidx]
    return data_t1_bboxed, data_t1ce_bboxed, data_t2_bboxed, data_flair_bboxed, gt_bboxed
 ###################################   


def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    # out_random = np.random.normal(0, 1, size=volume.shape)
    # out[volume == 0] = out_random[volume == 0]
    out = out.astype(np.float32)
    return out

class MedicalImageDeal(object):
    def __init__(self, img, percent=1):
        self.img = img
        self.percent = percent

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return (self.img - self.img.min()) / (self.img.max() - self.img.min())
    

def show_preBraTS_image_info():
    input_root = predata_root
    output_info_dir = "./log/preBraTS"
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)

    logging.basicConfig(filename = output_info_dir + "/preBraTS_info.txt", level = logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    info_img_dir = input_root + "/" + img_dirs[0] # take t1 as example
    print("info_img_dir:{}".format(info_img_dir))
    ori_img_path = sorted(glob.glob(info_img_dir + "/*.nii.gz")) 
    image_num = 0
    z_min = 10000
    z_max = 0
    h_min = 10000
    h_max = 0
    w_min = 10000
    w_max = 0

    for img_name in ori_img_path:
        case_name = img_name.split("/")[-1]
        logging.info("case name :{}".format(case_name))
        img_obj = sitk.ReadImage(img_name)    
        space = img_obj.GetSpacing()
        size = img_obj.GetSize()

        imageTr = sitk.GetArrayFromImage(img_obj)
        logging.info("image_np_shape: {}".format(imageTr.shape))
        if size[2]>z_max:
            z_max = size[2]
        if size[2]<z_min:
            z_min = size[2]

        if size[0]>h_max:
                h_max = size[0]
        if size[0]<h_min:
            h_min = size[0]

        if size[1]>w_max:
            w_max = size[1]
        if size[1]<w_min:
            w_min = size[1]
        logging.info("itk_space:{}, itk_size:{}".format(space, size))
        logging.info("##################################################################")
        image_num += 1
    logging.info("image num: {}".format(image_num))
    logging.info("z_min:{}, z_max:{}".format(z_min, z_max))
    logging.info("h_min:{}, h_max:{}".format(h_min, h_max))
    logging.info("w_min:{}, w_max:{}".format(w_min, w_max))

def show_oldBraTS_image_info():
    input_root = data_root
    output_info_dir = "./log/preBraTS"
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)

    logging.basicConfig(filename = output_info_dir + "/BraTS_info_beforePre.txt", level = logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    imgName_dir_list = sort(os.listdir(input_root))

    image_num = 0
    z_min = 10000
    z_max = 0
    h_min = 10000
    h_max = 0
    w_min = 10000
    w_max = 0

    for case_name in imgName_dir_list:
        logging.info("case name :{}".format(case_name))
        img_path = input_root + "/" + case_name +"/" +case_name + "_t1.nii.gz"
        img_obj = sitk.ReadImage(img_path)    
        space = img_obj.GetSpacing()
        size = img_obj.GetSize()

        imageTr = sitk.GetArrayFromImage(img_obj)
        logging.info("image_np_shape: {}".format(imageTr.shape))
        if size[2]>z_max:
            z_max = size[2]
        if size[2]<z_min:
            z_min = size[2]

        if size[0]>h_max:
                h_max = size[0]
        if size[0]<h_min:
            h_min = size[0]

        if size[1]>w_max:
            w_max = size[1]
        if size[1]<w_min:
            w_min = size[1]
        logging.info("itk_space:{}, itk_size:{}".format(space, size))
        logging.info("##################################################################")
        image_num += 1
    logging.info("image num: {}".format(image_num))
    logging.info("z_min:{}, z_max:{}".format(z_min, z_max))
    logging.info("h_min:{}, h_max:{}".format(h_min, h_max))
    logging.info("w_min:{}, w_max:{}".format(w_min, w_max))

def deal_all_preBraTS_to_volumes():
    input_root = predata_root
    save_root_path = input_root+"/Volumes"
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    
    volume_num = 0
    t1_image_paths = sorted(glob.glob(input_root + "/" + img_dirs[0] +"/*.nii.gz")) 

    for t1_case_path in t1_image_paths:
        print(t1_case_path)
        # t1_case_path = "/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingPre/t1/BraTS20_Training_032_t1.nii.gz"
        
        item_name = t1_case_path.split("/")[-1]
        idx = item_name.rfind("_") 
        item_name = item_name[:idx]
        print("item_name:{}".format(item_name))

        t1_image_itk = sitk.ReadImage(t1_case_path)
        t1_image = sitk.GetArrayFromImage(t1_image_itk)
        print("t1_image:{}".format(t1_image.shape))

        t1ce_path = t1_case_path.replace("t1", "t1ce")
        t1ce_itk = sitk.ReadImage(t1ce_path)
        t1ce_image = sitk.GetArrayFromImage(t1ce_itk)
        print("t1ce_image:{}".format(t1ce_image.shape))

        t2_path = t1_case_path.replace("t1", "t2")
        t2_itk = sitk.ReadImage(t2_path)
        t2_image = sitk.GetArrayFromImage(t2_itk)
        print("t2_image:{}".format(t2_image.shape))

        flair_path = t1_case_path.replace("t1", "flair")
        flair_itk = sitk.ReadImage(flair_path)
        flair_image = sitk.GetArrayFromImage(flair_itk)
        print("flair_image:{}".format(flair_image.shape))

        image = np.stack((t1_image, t1ce_image, t2_image, flair_image), axis=0)
        print("image:{}".format(image.shape))

        label_path = t1_case_path.replace("t1", "2label")
        label_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_itk)

        scribble_path = t1_case_path.replace("t1", "2scribble")
        scribble_itk = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble_itk)

        f = h5py.File(save_root_path + "/{}.h5".format(item_name),'w')
        f.create_dataset("image", data = image, compression = "gzip")
        f.create_dataset("label", data = label, compression = "gzip")
        f.create_dataset("scribble", data=scribble, compression = "gzip")
        f.close
        volume_num +=1
        print("#########################")
    print("Convert all BraTS2020 data to volumes, total {} volumes".format(volume_num)) #total 369 volumes

def write_images_nametxt_volumes(): #70，15，15
    output_info_dir = predata_root
    info_data_dir = predata_root + "/Volumes"
    data_list = sorted(os.listdir(info_data_dir))
    num_data = len(data_list)

    train_percent = 0.7
    val_percent = 0.15
    test_percent = 0.15

    num_train = math.ceil(num_data * train_percent)
    num_val = math.ceil(num_data * val_percent)
    num_test = num_data - num_train - num_val

    random.shuffle(data_list)

    train_list = data_list[:num_train]
    val_list = data_list[num_train:num_train+num_val] 
    test_list = data_list[-num_test:]

    with open(output_info_dir + '/train.txt', 'w') as f:
        for name in train_list:
            f.write(os.path.join("Volumes", name)+'\n')

    with open(output_info_dir + '/valid.txt', 'w') as f:
        for name in val_list:  
            f.write(os.path.join("Volumes", name)+'\n')
    
    with open(output_info_dir + '/test.txt', 'w') as f:
        for name in test_list:
            f.write(os.path.join("Volumes", name)+'\n')


def deal_trainSet_to_slices(): # 259 train
    input_root = predata_root
    save_root_path = input_root+"/trainSlices4"
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)

    with open(input_root + "/train.txt", "r") as f:
        trainData_name_list = f.readlines()
    trainData_name_list = [item.replace("\n", "").split(",")[0].split("/")[-1].split(".")[0] for item in trainData_name_list]
    # # print(trainData_name_list) 
    # print(len(trainData_name_list))

    trainData_path = sorted(predata_root + "/t1/{}_t1.nii.gz".format(item) for item in trainData_name_list)
    # print(trainData_path)
    # print(len(trainData_path))

    training_slice_num = 0

    for t1_case_path in trainData_path:
        # t1_case_path = "/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingPre/t1/BraTS20_Training_032_t1.nii.gz"
        item_name = t1_case_path.split("/")[-1]
        idx = item_name.rfind("_") # Reverse to find the last underscore
        item_name = item_name[:idx]
        print("item_name:{}".format(item_name))

        t1_image_itk = sitk.ReadImage(t1_case_path)
        t1_image = sitk.GetArrayFromImage(t1_image_itk)
        # print("t1_image:{}".format(t1_image.shape))

        t1ce_path = t1_case_path.replace("t1", "t1ce")
        t1ce_itk = sitk.ReadImage(t1ce_path)
        t1ce_image = sitk.GetArrayFromImage(t1ce_itk)
        # print("t1ce_image:{}".format(t1ce_image.shape))

        t2_path = t1_case_path.replace("t1", "t2")
        t2_itk = sitk.ReadImage(t2_path)
        t2_image = sitk.GetArrayFromImage(t2_itk)
        # print("t2_image:{}".format(t2_image.shape))

        flair_path = t1_case_path.replace("t1", "flair")
        flair_itk = sitk.ReadImage(flair_path)
        flair_image = sitk.GetArrayFromImage(flair_itk)
        # print("flair_image:{}".format(flair_image.shape))

        image = np.stack((t1_image, t1ce_image, t2_image, flair_image), axis=0)
        print("shape of stacked image:{}".format(image.shape))

        label_path = t1_case_path.replace("t1", "2label")
        label_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_itk)
        print("shape of stacked label:{}".format(label.shape))


        scribble_path = t1_case_path.replace("t1", "2scribble")
        scribble_itk = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble_itk)
        print("shape of stacked scribble:{}".format(scribble.shape))


        for imageSlice_ind in range(image.shape[1]):
            f = h5py.File(save_root_path + "/{}_s{}.h5".format(item_name, imageSlice_ind),'w')
            f.create_dataset("image", data = image[:,imageSlice_ind], compression = "gzip")
            f.create_dataset("label", data = label[imageSlice_ind], compression = "gzip")
            f.create_dataset("scribble", data=scribble[imageSlice_ind], compression = "gzip")
            f.close
            training_slice_num += 1
    print("#########################")
    print("Convert all BraTS2020 train data to slices, total {} slices".format(training_slice_num)) #141736 slices #4 channel: 35434 slices

def write_images_nametxt_trainSlices4(): 
    output_info_dir = predata_root
    info_data_dir = predata_root + "/trainSlices4"
    trainSlices4_data_list = sorted(os.listdir(info_data_dir))
    num_data = len(trainSlices4_data_list)
    print("Total {} 4 channal training slices".format(num_data))

    with open(output_info_dir + '/trainSlices4.txt', 'w') as f:
        for name in trainSlices4_data_list:
            f.write(os.path.join("trainSlices4", name)+'\n')


if __name__ =="__main__":
    flag = 7

    if flag == 0: 
        convert_BraTSLabel_into2()
    elif flag == 1: #crop、normalize image and crospponding 
        img_data_path = sorted(glob.glob(data_root + "/*/*_t1.nii.gz" ))
        new_pre_data_root = predata_root

        if not os.path.exists(new_pre_data_root):
            os.makedirs(new_pre_data_root)
            os.makedirs(new_pre_data_root + "/t1")
            os.makedirs(new_pre_data_root + "/t1ce")
            os.makedirs(new_pre_data_root + "/flair")
            os.makedirs(new_pre_data_root + "/t2")
            os.makedirs(new_pre_data_root + "/2label")

        for item in img_data_path :
            img_t1 = sitk.GetArrayFromImage(sitk.ReadImage(item))
            img_t1ce = sitk.GetArrayFromImage(sitk.ReadImage(item.replace("_t1", "_t1ce")))
            img_t2 = sitk.GetArrayFromImage(sitk.ReadImage(item.replace("_t1", "_t2")))
            img_flair = sitk.GetArrayFromImage(sitk.ReadImage(item.replace("_t1", "_flair")))
            lab = sitk.GetArrayFromImage(sitk.ReadImage(item.replace("_t1", "_2label")))
            #去除0背景
            img_t1, img_t1ce, img_t2, img_flair, lab = brain_bbox_4modalData(img_t1, img_t1ce, img_t2, img_flair, lab)
            
            img_t1 = MedicalImageDeal(img_t1, percent=0.999).valid_img
            img_t1 = itensity_normalize_one_volume(img_t1)
            img_t1ce = MedicalImageDeal(img_t1ce, percent=0.999).valid_img
            img_t1ce = itensity_normalize_one_volume(img_t1ce)
            img_t2 = MedicalImageDeal(img_t2, percent=0.999).valid_img
            img_t2 = itensity_normalize_one_volume(img_t2)
            img_flair = MedicalImageDeal(img_flair, percent=0.999).valid_img
            img_flair = itensity_normalize_one_volume(img_flair)

            img_t1_name = item.split("/")[-1]
            img_t1ce_name = img_t1_name.replace("_t1", "_t1ce")
            img_t2_name = img_t1_name.replace("_t1", "_t2")
            img_flair_name = img_t1_name.replace("_t1", "_flair")
            lab_name = img_t1_name.replace("_t1", "_2label")
            print(img_t1_name)
            sitk.WriteImage(sitk.GetImageFromArray(img_t1), new_pre_data_root + "/t1/{}".format(img_t1_name))
            sitk.WriteImage(sitk.GetImageFromArray(img_t1ce), new_pre_data_root + "/t1ce/{}".format(img_t1ce_name))
            sitk.WriteImage(sitk.GetImageFromArray(img_t2), new_pre_data_root + "/t2/{}".format(img_t2_name))
            sitk.WriteImage(sitk.GetImageFromArray(img_flair), new_pre_data_root + "/flair/{}".format(img_flair_name))
            sitk.WriteImage(sitk.GetImageFromArray(lab), new_pre_data_root + "/2label/{}".format(lab_name))
    ################There is a step in the middle that requires scribble_generater.py to generate the corresponding scribble
    elif flag == 2:
        show_preBraTS_image_info()
    elif flag == 3:
        deal_all_preBraTS_to_volumes()
    elif flag == 4:
        write_images_nametxt_volumes()
    elif flag == 5:
        deal_trainSet_to_slices()
    elif flag==6:
        write_images_nametxt_trainSlices4()
    elif flag == 7:
        show_oldBraTS_image_info()


