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
from preprocess_WORD import get_3d_bounding_box
from skimage import exposure

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
data_root = "/mnt/data/HM/Datasets/ACDC2017/ACDC"
# data_root = "/mnt/data/HM/Datasets/ACDC2017/ACDC_resample"


def show_images_info(img_dir ="train"):
    input_root = data_root
    output_info_dir = "./log/ACDC"
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)
    
    logging.basicConfig(filename = output_info_dir + "/" + img_dir +"trainData_info2.txt", level = logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    info_img_dir = input_root + "/" + img_dir + "/images"

    # img_names = sorted(os.listdir(info_img_dir)) 
    ori_img_path = sorted(glob.glob(info_img_dir + "/*.nii.gz")) 
    image_num = 0
    z_min = 1000
    z_max = 0
    h_min = 1000
    h_max = 0
    w_min = 1000
    w_max = 0
    for img_name in ori_img_path:
        case_name = img_name.split("/")[-1]
        img_obj = sitk.ReadImage(img_name)    
        space = img_obj.GetSpacing()
        size = img_obj.GetSize()

        imageTr = sitk.GetArrayFromImage(img_obj)
        logging.info("image_np_shape: image:{}, image_shape: {}".format(case_name, imageTr.shape))

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


        logging.info("itk_shape: {}: space:{}, size:{}\n".format(case_name, space, size))
        image_num += 1
    logging.info("image num: {}".format(image_num))
    logging.info("z_min:{}, z_max:{}".format(z_min, z_max))
    logging.info("h_min:{}, h_max:{}".format(h_min, h_max))
    logging.info("w_min:{}, w_max:{}".format(w_min, w_max))

def MedicalImages_deal(image_root, validImg_flag = False, percent=0.99, resample=True):
        input_root = image_root
        output_root = input_root + "_N2" 
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        img_input_list = sorted(glob.glob(input_root + "/*.nii.gz"))     

        for img_root in img_input_list:
            case_name = img_root.split("/")[-1]
            print(case_name)
            img_obj = sitk.ReadImage(img_root) 
            image_np = sitk.GetArrayFromImage(img_obj)

            if validImg_flag :
                cdf = exposure.cumulative_distribution(image_np)
                watershed = cdf[1][cdf[0] >= percent][0]
                image_np = np.clip(image_np, image_np.min(), watershed)
            
            #MinMax-Normalization
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

            #save
            img_out = sitk.GetImageFromArray(image_np)
            img_out.SetSpacing(img_obj.GetSpacing())
            img_out.SetOrigin(img_obj.GetOrigin())
            img_out.SetDirection(img_obj.GetDirection())
            img_out_name = output_root + "/" + case_name
            sitk.WriteImage(img_out, img_out_name)        

def deal_dataSet_to_volumes(subdir = "train"):#for volumes
    input_root = data_root 
    save_root_path = input_root + "_for2D"
    save_sub_dir = save_root_path + "/" + subdir + "rlessd16_volumes"
    # save_sub_dir = save_root_path + "/" + subdir + "o_volumes"
    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)
    volume_num = 0

    image_path = sorted(
        glob.glob(input_root + "/" + subdir + "/labels/*.nii.gz"))
        
    for case in image_path:
        item_name = case.split("/")[-1].split(".")[0].replace("_gt","") 
        print(item_name)

        label_itk = sitk.ReadImage(case)
        label = sitk.GetArrayFromImage(label_itk)  
        
        image_path = case.replace("labels","images_N").replace("_gt","")
        # image_path = case.replace("labels","images").replace("_gt","")
        image_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_itk)

    
        scribble_path = case.replace("labels", "scribbles_rlessd16").replace("_gt","_scribble")
        scribble_itk = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble_itk)
        
        
        if image.shape != label.shape:
            print("Error, the shape of validating image is not same as label")
 
        f = h5py.File(save_sub_dir + '/{}.h5'.format(item_name), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        if subdir =="train":
            f.create_dataset('scribble', data=scribble, compression ="gzip")
        f.close()
        volume_num += 1
    print("Converted all " + subdir +  " data to  volumes")
    print("Total {}  {} volumes".format(volume_num, subdir))

def write_images_nametxt(subdir = "train"):
    input_root = data_root + "_for2D"
    output_info_dir = input_root

    # img_dir = subdir + "_volumes"
    img_dir = subdir + "rlessdf16" + "_slices" #trainfor2D orless8_slices
    info_img_dir = input_root + "/" + img_dir 
    img_names = sorted(os.listdir(info_img_dir)) # 

    if subdir == "train":
        name = img_dir.split("_")[0]
    elif subdir == "val":
        name = "valid__"
    elif subdir == "TestSet":
        name = "test__"
    f = open(output_info_dir + "/" + name +".txt", "w")
    for img_name in img_names:
        f.write(os.path.join(img_dir,img_name) + "\n")
    f.close()

def deal_dataSet_to_slices(subdir = "train"):
    #for 2D. o means original image without normalization; r means less scribble with max components
    # resample 1,1,1: 12106  train slices for training 
    # original: Total 1356  train slices
    input_root = data_root 
    save_root_path = input_root + "_for2D"
    save_sub_dir = save_root_path + "/" + subdir + "rlessdf16_slices"
    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)
    slice_num = 0
    print(save_sub_dir)

    image_path = sorted(
        glob.glob(input_root + "/" + subdir + "/labels/*.nii.gz"))
        
    for case in image_path:
        item_name = case.split("/")[-1].split(".")[0].replace("_gt","") 
        print(item_name)

        label_itk = sitk.ReadImage(case)
        label = sitk.GetArrayFromImage(label_itk)  
        
        image_path = case.replace("labels","images_N").replace("_gt","")
        image_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_itk)

    
        scribble_path = case.replace("labels", "scribbles_rlessdf16").replace("_gt","_scribble")
        scribble_itk = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble_itk)
  
        
        if image.shape != label.shape:
            print("Error, the shape of image is not same as label")

        for slice_ind in range(image.shape[0]):
            f = h5py.File(save_sub_dir + '/{}_slice_{}.h5'.format(item_name, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=label[slice_ind], compression="gzip")
            if subdir =="train":
                f.create_dataset('scribble', data=scribble[slice_ind], compression ="gzip")
            f.close()
            slice_num += 1
     
    print("Converted data " + subdir +  " data to  slices")
    print("Total {}  {} slices".format(slice_num, subdir))

def image_resample(subdir = "train"): 
    input_root = data_root 
    output_root = data_root+"_resample"
    if not os.path.exists(output_root):
            os.makedirs(output_root) 
    subdirs = [subdir + "/images", subdir + "/labels", subdir + "/scribbles"]

    for sub_dir in subdirs :
        input_dir = input_root + '/' + sub_dir
        output_dir = output_root + '/' + sub_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) 

        img_names = sorted(os.listdir(input_dir))
        for img_name in img_names:
            print(img_name)
            img_obj = sitk.ReadImage(input_dir + '/' + img_name)
            spacing = img_obj.GetSpacing()
            origin  = img_obj.GetOrigin()
            direction = img_obj.GetDirection()
            input_size = img_obj.GetSize()

            output_spacing = [1.0,1.0,1.0] 
            output_size = [int((input_size[i] * spacing[i]) / output_spacing[i]) for i in range(3)]
            transform = sitk.Transform(3, sitk.sitkIdentity) #what's mean?

            interp = sitk.sitkLinear if "images" in sub_dir else sitk.sitkNearestNeighbor 
            resampled_img = sitk.Resample(img_obj, output_size, transform,
                interp, origin, output_spacing, direction)
            sitk.WriteImage(resampled_img, output_dir + '/' + img_name)

if __name__ == "__main__":
    func = 5
    imgdir_list=["train", "val", "TestSet"]#140,30,30
    imgdir = imgdir_list[0]
    
    if func == 1:
        show_images_info(imgdir)
    elif func == 2:
        image_root = data_root + "/" + imgdir + "/images"
        MedicalImages_deal(image_root)
    elif func==3:
        deal_dataSet_to_volumes(imgdir)
    elif func == 4:
        deal_dataSet_to_slices(imgdir)   
    elif func == 5:
        write_images_nametxt(imgdir)
    elif func ==6:
        image_resample(imgdir)
    elif func == 7:
        deal_dataSet_to_slices(imgdir)   
        
