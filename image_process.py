import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage
from PIL import Image
import os 
import sys
import argparse
import shutil
import h5py

ACDC_DIR = "/home/disk16t/data/heart/ACDC_DMSPS"

def copy_test_images(data_root):
    for subdir in ["images_N", "labels"]:
        source_dir = ACDC_DIR + "/TestSet/" + subdir
        target_dir = data_root + "/../ACDC/TestSet/" + subdir
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True) 

def write_3dh5_files(data_root, stage = 'train'):
    # data_root = "./data/ACDC2017/ACDC_for2D"
    volume_dir = ACDC_DIR + "/" + stage + "/images_N"
    volume_names = os.listdir(volume_dir)
    for volume_name in volume_names:
        img_obj = sitk.ReadImage(volume_dir + "/" + volume_name)
        img = sitk.GetArrayFromImage(img_obj)
        print(volume_name, img.shape)

        # write 3D volumes
        out_3dh5_name = volume_name.replace(".nii.gz", ".h5")
        out_3dh5_dir = data_root + "/{0:}_volumes/".format(stage)
        if(not os.path.exists(out_3dh5_dir)):
            os.mkdir(out_3dh5_dir)
        h5file = h5py.File(out_3dh5_dir + "/" + out_3dh5_name, 'w')
        h5file.create_dataset('image', data = img, compression="gzip")
        label_folders = ["labels"]
        key_names = ["label"]
        postfixes = ["gt"] 
        for i in range(len(label_folders)):
            lab_name_i = volume_name.replace(".nii.gz", "_{0:}.nii.gz".format(postfixes[i]))
            lab_dir_i = ACDC_DIR + "/" + stage + "/" + label_folders[i]
            lab_obj_i = sitk.ReadImage(lab_dir_i + "/" + lab_name_i)
            lab_i = sitk.GetArrayFromImage(lab_obj_i)
            h5file.create_dataset(key_names[i], data = lab_i, compression="gzip")
        h5file.close()

def write_2dh5_files_for_training(data_root):
    volume_dir = ACDC_DIR + "/train/images_N"
    h5_dir  = data_root + "/train_slices"
    if(not os.path.exists(h5_dir)):
        os.mkdir(h5_dir)
    volume_names = os.listdir(volume_dir)
    for volume_name in volume_names:
        key_folders = ["images_N","labels", "scribbles", "scribbles_rless2"] + \
            ["scribbles_rlessd4", "scribbles_rlessd8", "scribbles_rlessd16"]
        key_names = ["image","label", "scribble", "scribble_rless2"] + \
            ["scribble_rlessd4", "scribble_rlessd8", "scribble_rlessd16"]
        postfixes = [".nii.gz", "_gt.nii.gz"] + ["_scribble.nii.gz"] * 5
        images = []
        for i in range(len(key_folders)):
            img_name_i = volume_name.replace(".nii.gz", postfixes[i])
            img_dir_i = ACDC_DIR + "/train/" +  key_folders[i]
            img_obj_i = sitk.ReadImage(img_dir_i + "/" + img_name_i)
            img_i = sitk.GetArrayFromImage(img_obj_i)
            images.append(img_i)
        slice_num = images[0].shape[0]
        for idx in range(slice_num):
            patient_name = volume_name.split(".")[0]
            h5_name = patient_name + "_slice_{0:}.h5".format(idx)
            h5file = h5py.File(h5_dir + "/" + h5_name, 'w')
            for i in range(len(key_folders)):
                key  = key_names[i]
                data = images[i][idx]
                h5file.create_dataset(key, data = data, compression="gzip")
            h5file.close()

def write_2dh5_files_for_training2(data_root, pred_data_path):
    volume_dir = ACDC_DIR + "/train/images_N"
    stage2_path = data_root + "/train_slices_stage2"
    if(not os.path.exists(stage2_path)):
        os.mkdir(stage2_path)
    volume_names = os.listdir(volume_dir)
    for volume_name in volume_names:
        patient_name = volume_name[:18]
        img_obj = sitk.ReadImage(volume_dir + "/" + volume_name)
        img = sitk.GetArrayFromImage(img_obj)
        seed_name = volume_name.replace(".nii.gz", "_seeds_expand.nii.gz")
        seed_obj = sitk.ReadImage(pred_data_path + "/" + seed_name)
        seed = sitk.GetArrayFromImage(seed_obj)
        # scrb_name = volume_name.replace(".nii.gz", "_scribble.nii.gz")
        # scrb_obj = sitk.ReadImage(volume_dir + "/../scribbles/" + scrb_name)
        # scrb = sitk.GetArrayFromImage(scrb_obj)

        D, H, W = img.shape
        for idx in range(D):
            h5_name = patient_name + "_slice_{0:}.h5".format(idx)
            h5file = h5py.File(stage2_path + "/" + h5_name, 'w')
            h5file.create_dataset('image', data = img[idx], compression="gzip")
            h5file.create_dataset('expanded_seeds', data = seed[idx], compression="gzip")
            h5file.close()



if __name__ == "__main__":
    data_root = "data/ACDC2017/ACDC_for2D"
    if(len(sys.argv) < 2):
        print('Number of arguments should be at least 2. e.g.')
        print('   python image_process.py 0')
        exit()
    if(sys.argv[1] == '0'):
        print(sys.argv[1])
        write_3dh5_files(data_root, "train")
        write_3dh5_files(data_root, "val")
        write_2dh5_files_for_training(data_root)
        copy_test_images(data_root)
    else:
        pred_data_path = "result/acdc_dmsps_train"
        write_2dh5_files_for_training2(data_root, pred_data_path)

