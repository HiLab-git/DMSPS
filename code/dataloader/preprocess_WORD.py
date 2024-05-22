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
import shutil
from scipy import ndimage
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
data_root = "/mnt/data/HM/Datasets/Abdomen_word/WORD-V0.1.0-Admin_cropWL"
# crop as well as  adjust the window level and width
# W symbols the images have adjusted the window level and width
# L symbols the images have converted the label


def show_images_info(flag ="train"):
    input_root = data_root 
    output_info_dir = "./log/WORD"
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)
    
    logging.basicConfig(filename = output_info_dir + "/" + flag +"_info_WL.txt", level = logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if flag == "train":
        img_dir = "imagesTr"
    elif flag == "test":
        img_dir = "imagesTs"
    elif flag == "valid":
        img_dir = "imagesVal"

    info_img_dir = input_root + "/" + img_dir 

    img_names = sorted(os.listdir(info_img_dir)) 
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



        logging.info("itk_shape: {}: space:{}, size:{}".format(case_name, space, size))
        image_num += 1
    logging.info("image num: {}".format(image_num))
    logging.info("z_min:{}, z_max:{}".format(z_min, z_max))
    logging.info("h_min:{}, h_max:{}".format(h_min, h_max))
    logging.info("w_min:{}, w_max:{}".format(w_min, w_max))

def Load_LabelConvert_abdomenWORD(label_path=None):
    lab_obj = sitk.ReadImage(label_path)
    lab_np = sitk.GetArrayFromImage(lab_obj)#numpy.ndarray
    new_label = copy.deepcopy(lab_np)
    new_label[new_label == 7] = 0
    new_label[new_label == 8] = 7
    for i in range(9,17):
        #set some organs' labels to 0[background]
        new_label[new_label == i] = 0
    return new_label

def get_3d_bounding_box(seg_np = None):
    D, H, W = seg_np.shape
    d_idx, h_idx, w_idx = np.where(seg_np > 0) 
    d_min, d_max = d_idx.min(), d_idx.max()
    h_min, h_max = h_idx.min(), h_idx.max()
    w_min, w_max = w_idx.min(), w_idx.max()
    d_min, d_max = max(0, d_min - 20), min(D, d_max + 20)
    h_min, h_max = max(0, h_min - 20), min(H, h_max + 20)
    w_min, w_max = max(0, w_min - 20), min(W, w_max + 20)
    idx_min, idx_max = [d_min, h_min, w_min], [d_max, h_max, w_max]
    return idx_min, idx_max

def cropW_single_image(img_name, input_img_dir, idx_min, idx_max, output_img_dir, convert=False):
    img_name_full = input_img_dir + "/" + img_name
    img_obj = sitk.ReadImage(img_name_full)
    img = sitk.GetArrayFromImage(img_obj) 
    if convert == True:
        img = Load_LabelConvert_abdomenWORD(label_path=img_name_full)
        print("convert successfully")

    img_crop = img[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]

    #adjust the window level and width only on "images"
    if "image" in input_img_dir:
        thred_lower, thred_upper = -150, 250
        img_crop[img_crop < thred_lower] = thred_lower
        img_crop[img_crop > thred_upper] = thred_upper
        img_crop = (img_crop - thred_lower) / (thred_upper - thred_lower)

    img_crop_obj = sitk.GetImageFromArray(img_crop)
    img_crop_obj.SetSpacing(img_obj.GetSpacing())
    img_crop_obj.SetOrigin(img_obj.GetOrigin())
    img_crop_obj.SetDirection(img_obj.GetDirection())
    img_out_name = output_img_dir + "/" + img_name 
    sitk.WriteImage(img_crop_obj, img_out_name)

def cropW_3d_images_WORD(flag ="train"):
    input_root = data_root
    output_root = input_root + "_cropW" 

    if flag == "train":
        img_dir = "imagesTr"
        lab_dir = "labelsTr"
        scri_dir = "scribblesTr"
    elif flag == "test":
        img_dir = "imagesTs"
        lab_dir = "labelsTs"
    elif flag == "valid":
        img_dir = "imagesVal"
        lab_dir = "labelsVal"

    input_img_dir = input_root + "/" + img_dir
    input_lab_dir = input_root + "/" + lab_dir
    seg_dir = input_lab_dir
    output_img_dir = output_root + "/" + img_dir
    output_lab_dir = output_root +"/" + lab_dir
    if flag == "train":
        input_scri_dir = input_root + "/" + scri_dir 
        output_scri_dir = output_root + "/" + scri_dir

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        os.makedirs(output_lab_dir)
        if flag == "train":
            os.makedirs(output_scri_dir)
    
    img_names = sorted(os.listdir(input_img_dir)) 

    crop_location_d = defaultdict(defaultdict)   
    for img_name in img_names:
        print(img_name)
        seg_name = seg_dir + "/" + img_name
        
        seg_obj = sitk.ReadImage(seg_name)
        seg_np = sitk.GetArrayFromImage(seg_obj)#numpy.ndarray

        idx_min, idx_max = get_3d_bounding_box(seg_np)
        crop_location_d["idx_min"][img_name] = idx_min
        crop_location_d["idx_max"][img_name] = idx_max

        #crop the image
        cropW_single_image(img_name, input_img_dir, idx_min, idx_max, output_img_dir)

        #crop the label
        lab_name = img_name
        cropW_single_image(lab_name, input_lab_dir, idx_min, idx_max, output_lab_dir)

        #crop the scribble
        if flag =="train":
            scri_name = img_name
            cropW_single_image(scri_name, input_scri_dir, idx_min, idx_max, output_scri_dir)

    df = pd.DataFrame.from_dict(crop_location_d)
    df.to_csv("./log/" + flag + "_crop_location.csv")

def cropWL_3d_images_dealLabel_WORD(flag="train"):
    input_root = data_root
    output_root = data_root + "_cropWL" 

    if flag == "train":
        img_dir = "imagesTr"
        lab_dir = "labelsTr"
        scri_dir = "scribblesTr"
    elif flag == "test":
        img_dir = "imagesTs"
        lab_dir = "labelsTs"
    elif flag == "valid":
        img_dir = "imagesVal"
        lab_dir = "labelsVal"

    input_img_dir = input_root + "/" + img_dir
    input_lab_dir = input_root + "/" + lab_dir
    seg_dir = input_lab_dir
    output_img_dir = output_root + "/" + img_dir
    output_lab_dir = output_root +"/" + lab_dir
    if flag == "train":
        input_scri_dir = input_root + "/" + scri_dir 
        output_scri_dir = output_root + "/" + scri_dir

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        os.makedirs(output_lab_dir)
        if flag == "train":
            os.makedirs(output_scri_dir)
    
    img_names = sorted(os.listdir(input_img_dir)) 

    crop_location_d = defaultdict(defaultdict)   
    for img_name in img_names:
        print(img_name)

        seg_full_name = seg_dir + "/" + img_name  
        seg_np = Load_LabelConvert_abdomenWORD(label_path = seg_full_name) #seg = label

        idx_min, idx_max = get_3d_bounding_box(seg_np)
        crop_location_d["idx_min"][img_name] = idx_min
        crop_location_d["idx_max"][img_name] = idx_max

        #crop the image
        cropW_single_image(img_name, input_img_dir, idx_min, idx_max, output_img_dir)

        #crop the label
        lab_name = img_name
        cropW_single_image(lab_name, input_lab_dir, idx_min, idx_max, output_lab_dir, convert = True)
       
        #crop the scribble
        if flag =="train":
            scri_name = img_name
            cropW_single_image(scri_name, input_scri_dir, idx_min, idx_max, output_scri_dir, convert = True)

    df = pd.DataFrame.from_dict(crop_location_d)
    df.to_csv("./log/WORD/" + flag + "_cropWL_location.csv")

def deal_trainSet_to_slices():#for 2d
    input_root = data_root + "_cropWL"
    save_root_path = input_root + "_for2D"
    if not os.path.exists(save_root_path + "/Abdomen_training_slices"):
        os.makedirs(save_root_path+ "/Abdomen_training_slices")

    training_slice_num = 0
    imageTr_path = sorted(
        glob.glob(input_root + "/imagesTr/*.nii.gz")) #read image
    # print(imageTr_path)
    
    for case in imageTr_path:
        imageTr_itk = sitk.ReadImage(case)
        imageTr = sitk.GetArrayFromImage(imageTr_itk)
        
        labelTr_path = case.replace("imagesTr","labelsTr")
        labelTr_itk = sitk.ReadImage(labelTr_path)
        labelTr = sitk.GetArrayFromImage(labelTr_itk)
        
        scribbleTr_path = case.replace("imagesTr","scribblesTr")
        scribbleTr_itk = sitk.ReadImage(scribbleTr_path)
        scribbleTr = sitk.GetArrayFromImage(scribbleTr_itk)
     
        # imageTr = MedicalImageDeal(imageTr,percent=0.99).valid_img
        # imageTr = (imageTr - imageTr.min()) / (imageTr.max() - imageTr.min())
        # imageTr = rescale_intensity(imageTr, in_range=(contrast,1),out_range=(0,1))
        # imageTr = imageTr.astype(np.float32)

        item = case.split("/")[-1].split(".")[0] 
        print(item)

        if imageTr.shape != labelTr.shape:
            print("Error, the shape of image is not same as label")

        #切片操作,将切片保存
        for slice_ind in range(imageTr.shape[0]):
            f = h5py.File(
                save_root_path+'/Abdomen_training_slices/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=imageTr[slice_ind], compression="gzip")
            f.create_dataset('label', data=labelTr[slice_ind], compression="gzip")
            f.create_dataset(
                'scribble', data=scribbleTr[slice_ind], compression="gzip")
            f.close()
            training_slice_num += 1 #slice num +1         
    print("Converted all Abdomen training volumes to 2D slices")
    print("Total {} training slices".format(training_slice_num)) #original total 20115
    # after cropping there are totally 17554 training slices

def deal_dataSet_to_volumes(flag = "train"):#falg:  train, valid, test
    input_root = data_root + "_cropWL"
    save_root_path = input_root + "_for3D"

    if flag == "train":
        name = "Tr"
    if flag == "valid":
        name = "Val"
    elif flag == "test":
        name = "Ts"
    # if os.path.exists(save_root_path + "/Abdomen_" + name + "_volumes"):
    #     shutil.rmtree(save_root_path + "/Abdomen_" + name + "_volumes")
    if not os.path.exists(save_root_path + "/Abdomen_" + name + "_volumes"):
        os.makedirs(save_root_path+ "/Abdomen_" + name + "_volumes")

    volume_num = 0

    image_path = sorted(
        glob.glob(input_root + "/imagesTr/*.nii.gz"))
        
    for case in image_path:
        image_itk = sitk.ReadImage(case)
        image = sitk.GetArrayFromImage(image_itk)  

        label_path = case.replace("images","labels")
        label_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_itk)

        if flag == "train":
            scribble_path = case.replace("images", "scribbles")
            scribble_itk = sitk.ReadImage(scribble_path)
            scribble = sitk.GetArrayFromImage(scribble_itk)

        item = case.split("/")[-1].split(".")[0] 
        print(item)
        
        if image.shape != label.shape:
            print("Error, the shape of validating image is not same as label")
 
        f = h5py.File(
            save_root_path+'/Abdomen_' + name + '_volumes/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        if flag =="train":
            f.create_dataset('scribble', data=scribble, compression ="gzip")
        f.close()
        volume_num += 1
    print("Converted all Abdomen " + flag +  " data to  volumes")
    print("Total {}  {} volumes [new contrast]".format(volume_num, flag))


def write_images_nametxt(flag ="train"):
    input_root = data_root + "_cropWL_for3D" 
    # output_info_dir = "../config/data_WORD"
    output_info_dir = input_root
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)

    if flag == "train":
        img_dir = "Abdomen_Tr_volumes"
    elif flag == "test":
        img_dir = "Abdomen_Ts_volumes"
    elif flag == "valid":
        img_dir = "Abdomen_Val_volumes"
    else:
        img_dir = "Abdomen_"+flag+"_volumes"
    info_img_dir = input_root + "/" + img_dir 
    img_names = sorted(os.listdir(info_img_dir)) 
    # ori_img_path = sorted(glob.glob(info_img_dir + "/*.nii.gz")) 

    f = open(output_info_dir + "/" + flag +".txt", "w")
    # with open(output_info_dir + "/" + flag +"_name.txt", "w") as file:
    #         file.write(os.path.join(img_dir,img_name)) 
    for img_name in img_names:
        f.write(os.path.join(img_dir,img_name) + "\n")
    f.close()


def write_images_nametxt_for2D(flag ="train"):
    input_root = data_root + "_cropWL_for2D" 
    output_info_dir = input_root
    if not os.path.exists(output_info_dir):
        os.makedirs(output_info_dir)

    if flag == "train":
        img_dir = "Abdomen_Tr_slices"
    elif flag == "test":
        img_dir = "Abdomen_Ts_volumes"
    elif flag == "valid":
        img_dir = "Abdomen_Val_volumes"
    info_img_dir = input_root + "/" + img_dir 
    img_names = sorted(os.listdir(info_img_dir)) 
    # ori_img_path = sorted(glob.glob(info_img_dir + "/*.nii.gz")) 

    f = open(output_info_dir + "/" + flag +".txt", "w")
    # with open(output_info_dir + "/" + flag +"_name.txt", "w") as file:
    #         file.write(os.path.join(img_dir,img_name)) 
    for img_name in img_names:
        f.write(os.path.join(img_dir,img_name) + "\n")
    f.close()
       
    
def show_rotate():

    # "/mnt/data1/HM/Datasets/Abdomen_word/WORD-V0.1.0-Admin"
    load_data_root =  data_root + "_cropWL/imagesTr/word_0002.nii.gz"
    load_label_root =  data_root + "_cropWL/labelsTr/word_0002.nii.gz"
    img_obj = sitk.ReadImage(load_data_root)
    img = sitk.GetArrayFromImage(img_obj) 

    # image_1 = ndimage.rotate(img, 50, order=0, reshape=False)
    # img_crop_obj = sitk.GetImageFromArray(image_1)
    # img_crop_obj.SetSpacing(img_obj.GetSpacing())
    # img_crop_obj.SetOrigin(img_obj.GetOrigin())
    # img_crop_obj.SetDirection(img_obj.GetDirection())
    # img_out_name = "../../figure/rotate_show/word_0002_2dRotate.nii.gz" 
    # sitk.WriteImage(img_crop_obj, img_out_name)


    # image_2 = ndimage.rotate(img, 50, (-2, -3), reshape = False, order = 1)
    # img_crop_obj = sitk.GetImageFromArray(image_2)
    # img_crop_obj.SetSpacing(img_obj.GetSpacing())
    # img_crop_obj.SetOrigin(img_obj.GetOrigin())
    # img_crop_obj.SetDirection(img_obj.GetDirection())
    # img_out_name = "../../figure/rotate_show/word_0002_3dRotate_w.nii.gz" 
    # sitk.WriteImage(img_crop_obj, img_out_name)

    # image_3 = ndimage.rotate(img, 50, (-1,-2), reshape = False, order = 1)
    # img_crop_obj = sitk.GetImageFromArray(image_3)
    # img_crop_obj.SetSpacing(img_obj.GetSpacing())
    # img_crop_obj.SetOrigin(img_obj.GetOrigin())
    # img_crop_obj.SetDirection(img_obj.GetDirection())
    # img_out_name = "../../figure/rotate_show/word_0002_3dRotate_d.nii.gz" 
    # sitk.WriteImage(img_crop_obj, img_out_name)


    lab_obj = sitk.ReadImage(load_label_root)
    lab = sitk.GetArrayFromImage(lab_obj) 

    # lab_1 = ndimage.rotate(lab, 50, order=0, reshape=False)
    # img_crop_obj = sitk.GetImageFromArray(lab_1)
    # img_crop_obj.SetSpacing(lab_obj.GetSpacing())
    # img_crop_obj.SetOrigin(lab_obj.GetOrigin())
    # img_crop_obj.SetDirection(lab_obj.GetDirection())
    # img_out_name = "../../figure/rotate_show/word_0002_2dRotatelabel.nii.gz" 
    # sitk.WriteImage(img_crop_obj, img_out_name)

    lab_2 = ndimage.rotate(lab, 50, (-2, -3), reshape = False, order = 0)
    img_crop_obj = sitk.GetImageFromArray(lab_2)
    img_crop_obj.SetSpacing(lab_obj.GetSpacing())
    img_crop_obj.SetOrigin(lab_obj.GetOrigin())
    img_crop_obj.SetDirection(lab_obj.GetDirection())
    img_out_name = "../../figure/rotate_show/word_0002_3dRotatelabel_w0.nii.gz" 
    sitk.WriteImage(img_crop_obj, img_out_name)


    lab_3 = ndimage.rotate(lab, 50, (-1,-2), reshape = False, order = 0)
    img_crop_obj = sitk.GetImageFromArray(lab_3)
    img_crop_obj.SetSpacing(lab_obj.GetSpacing())
    img_crop_obj.SetOrigin(lab_obj.GetOrigin())
    img_crop_obj.SetDirection(lab_obj.GetDirection())
    img_out_name = "../../figure/rotate_show/word_0002_3dRotatelabel_d0.nii.gz" 
    sitk.WriteImage(img_crop_obj, img_out_name)


def deal_scribblePerN_to_volumes(flag):#flag: Per1, Per3... default: train
    input_root = data_root + "_cropWL"
    save_root_path = input_root + "_for3D"
    # if os.path.exists(save_root_path + "/Abdomen_" + name + "_volumes"):
    #     shutil.rmtree(save_root_path + "/Abdomen_" + name + "_volumes")
    if not os.path.exists(save_root_path + "/Abdomen_" + flag + "Tr_volumes"):
        os.makedirs(save_root_path+ "/Abdomen_" + flag + "Tr_volumes")

    volume_num = 0
    image_path = sorted(
        glob.glob(input_root + "/imagesTr/*.nii.gz"))
        
    for case in image_path:
        image_itk = sitk.ReadImage(case)
        image = sitk.GetArrayFromImage(image_itk)  

        label_path = case.replace("images","labels")
        label_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_itk)
        
        scribble_path = case.replace("images", "scribbles"+flag)
        scribble_itk = sitk.ReadImage(scribble_path)
        scribble = sitk.GetArrayFromImage(scribble_itk)

        item = case.split("/")[-1].split(".")[0] 
        print(item)
        
        if image.shape != label.shape:
            print("Error, the shape of validating image is not same as label")
 
        f = h5py.File(
            save_root_path+'/Abdomen_' + flag + 'Tr_volumes/{}.h5'.format(item), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('scribble', data=scribble, compression ="gzip")
        f.close()
        volume_num += 1
    print("Converted all Abdomen " + flag +  " data to  volumes")
    print("Total {}  {} volumes [new contrast]".format(volume_num, flag))


if __name__ == "__main__":
    func = 1
    flag1 = "Per5" # train, test, valid, ..., Per1, Per3,...

    if func == 1:
        show_images_info(flag = "train")
    elif func == 2:#使用full label
        cropW_3d_images_WORD(flag = "train")
        cropW_3d_images_WORD(flag = "test")
        cropW_3d_images_WORD(flag = "valid")
    elif func == 3:
        deal_trainSet_to_slices() # before train_slices = 17554
        # after label convertthere are only total 9223 training slices
    elif func ==4 :
        deal_dataSet_to_volumes(flag = "valid") # 20 validation volumes
        deal_dataSet_to_volumes(flag = "test") #30 test volumes
    elif func == 5: #select 7 organs
        cropWL_3d_images_dealLabel_WORD(flag = "train")
        cropWL_3d_images_dealLabel_WORD(flag = "test")
        cropWL_3d_images_dealLabel_WORD(flag = "valid")
    elif func == 6: #
        deal_dataSet_to_volumes(flag = "train")
        # deal_dataSet_to_volumes(flag = "test")
        # deal_dataSet_to_volumes(flag = "valid")
    elif func == 7:
        write_images_nametxt(flag1 + "Tr") 
    elif func == 8: 
        show_rotate()
    elif func == 9: #more poor scribble
        deal_scribblePerN_to_volumes(flag1)

    elif func == 10:
        write_images_nametxt_for2D(flag = "train")
        write_images_nametxt_for2D(flag = "test")
        write_images_nametxt_for2D(flag = "valid")
    
