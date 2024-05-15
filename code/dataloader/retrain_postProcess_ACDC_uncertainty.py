import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage
import os 
import argparse
import glob
import h5py
"""
This code mainly processes the expanded train set into slices for 2d task and then write into txt"""
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/ACDC2017/ACDC', help='Name of Experiment')
parser.add_argument('--data_type', type=str,
                    default='Heart', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='ACDC', help='Data name') 
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')
 
parser.add_argument('--model', type=str,
                    default='unet_cct', help='model_name')
parser.add_argument('--exp', type=str,
                    default='A_weakly_SPS_2d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='fold name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times_num') 

parser.add_argument('--func', type=int, default=0,
                    help='function') 
parser.add_argument('--txtName', type=str, default="trainReT01",
                    help='txt name') 
FLAGS = parser.parse_args()

def deal_retrainData_to_slices_uncertainty(pred_data_path, flag, subdir = "train" ):#for retrain 2D
    input_image_root = FLAGS.data_root_path
    save_root_path = FLAGS.data_root_path + "_for2D"
    save_sub_dir = save_root_path + "/" + flag
    
    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)
    slice_num = 0

    image_path = sorted(
        glob.glob(input_image_root + "/" + subdir + "/labels/*.nii.gz"))
        
    for case in image_path:
        item_name = case.split("/")[-1].split(".")[0].replace("_gt","") 
        print(item_name)

        label_itk = sitk.ReadImage(case) #gt
        label = sitk.GetArrayFromImage(label_itk)  
        
        image_path = case.replace("labels","images_N").replace("_gt","")
        image_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_itk)

        pred_un_name = case.split("/")[-1].replace("_gt", "_pred_cuM")
        pseudoLab_un_path = pred_data_path+"/"+pred_un_name
        pseudoLab_itk = sitk.ReadImage(pseudoLab_un_path)
        pseudoLab = sitk.GetArrayFromImage(pseudoLab_itk)
  
        if image.shape != label.shape:
            print("Error, the shape of image is not same as label")

        for slice_ind in range(image.shape[0]):
            f = h5py.File(save_sub_dir + '/{}_slice_{}.h5'.format(item_name, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=label[slice_ind], compression="gzip")
            if subdir =="train":
                f.create_dataset('pseudoLab', data=pseudoLab[slice_ind], compression ="gzip")
            f.close()
            slice_num += 1
     
    print("Converted data re" + subdir +  " data to  slices")
    print("Total {}  {} slices for retrain".format(slice_num, subdir))

def write_images_nametxt(flag):
    input_root = FLAGS.data_root_path  + "_for2D"
    output_info_dir = input_root

    img_dir = flag 
    info_img_dir = input_root + "/" + img_dir 
    img_names = sorted(os.listdir(info_img_dir))
    name = flag.replace("_slices","")

    f = open(output_info_dir + "/" + name +".txt", "w")
    for img_name in img_names:
        f.write(os.path.join(img_dir,img_name) + "\n")
    f.close()


if __name__ == "__main__":

    input_test_save_path = "../../result/{}_{}/{}_{}_{}_{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.fold, FLAGS.savedir, FLAGS.tt_num)

    func = FLAGS.func
    flag = FLAGS.txtName + "_slices" # is also the filename
  
    if func == 0:
        deal_retrainData_to_slices_uncertainty(input_test_save_path, flag)
    elif func == 1:
        write_images_nametxt(flag)
