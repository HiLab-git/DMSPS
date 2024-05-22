import os
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import logging
import SimpleITK as sitk
import cv2
from matplotlib.colors import ListedColormap
import math
from skimage.measure import label, regionprops
from scipy import ndimage

data_root = "/mnt/data/HM/Datasets/ACDC2017/ACDC"
test_scri_dir = "TestSet/scribbles"#30
train_scri_dir = "train/scribbles"#140
valid_scri_dir = "val/scribbles"#30

def get_largest_connectted_component(binary_map):
    s = ndimage.generate_binary_structure(2,1) #4
    # s = ndimage.generate_binary_structure(2,2) # 8
    labeled_array,num_features = ndimage.label(binary_map, s)
    if num_features == 0:
        return np.zeros_like(binary_map)
    labels = np.arange(1, num_features+1)
    counts = np.bincount(labeled_array.flatten())[1:]
    if len(counts) == 0:
        return np.zeros_like(binary_map)
    max_label = labels[np.argmax(counts)] 
    max_component = np.zeros_like(labeled_array)
    max_component[labeled_array == max_label] = 1
    return max_component


def reduce_volume_judgeByAbsoluteLength(binary_matrix, reduce_num = 1/2): 
    # Find connected regions in the binary matrix
    regions = label(binary_matrix)
    # Iterate through each connected region
    for region in regionprops(regions):
        print("********************")
        # Get the coordinates of the region
        min_row, min_col, max_row, max_col = region.bbox
        print("min_row:{}, min_col:{}, max_row:{}, max_col:{}".format(min_row, min_col, max_row, max_col))
        # Get the center of the region
        l_row = abs(min_row-max_row)
        l_col = abs(min_col-max_col)
        center_row = int (min_row + l_row * (1-reduce_num))
        center_col = int(min_col + l_col* (1-reduce_num))
        # print(center_col, center_col)
        print("center_row:{}, center_col:{}".format(center_row, center_col))

        if abs(min_row-max_row)>abs(min_col-max_col):
            binary_matrix[min_row:center_row, :] = 0 #leave the right area
        else:
            binary_matrix[:, min_col:center_col] = 0 #leave the bottom area
        binary_matrix = get_largest_connectted_component(binary_matrix)
    return binary_matrix   


def scan(subdir, input_index, save_index):
    input_scri_path = os.path.join(data_root, subdir, input_index)
    input_scri_list = sorted(glob.glob(input_scri_path + "/*.nii.gz"))
    case_num = 0

    save_scri_path = subdir + save_index 
    if not os.path.exists(save_scri_path):
        os.makedirs(save_scri_path)
    
    for case_path in input_scri_list:
        case_num += 1
        case_name = case_path.split("/")[-1].split(".")[0]
        print("case_name:{}".format(case_name))

        scri_obj = sitk.ReadImage(case_path) 
        scri_np = sitk.GetArrayFromImage(scri_obj)

        new_scribble = np.zeros_like(scri_np)

        for i in range(np.shape(scri_np)[0]): #per slice
            slice = scri_np[i,:,:]
            new_slice = np.ones_like(slice) * 4
            for j in np.unique(scri_np):
                binary_map = slice == j

                binary_map = binary_map.astype(np.uint8)
                if j != 4:
                    binary_map = reduce_volume_judgeByAbsoluteLength(binary_map)
                    new_slice = np.where(binary_map==1, j, new_slice)
            new_scribble[i] = new_slice  
        new_scribble_itk = sitk.GetImageFromArray(new_scribble.astype(np.float32))
        new_scribble_itk.CopyInformation(scri_obj)
        sitk.WriteImage(new_scribble_itk, save_scri_path + "/" + case_name + ".nii.gz")
         
    print("deal case num:{}".format(case_num))


if __name__ == "__main__":

    flag = ["", "rless2", "rlessd4", "rlessd8", "rlessd16"] #proceed recursively
    index = 0 #index = [0,1,2,3]
    scan(train_scri_dir, flag[index], flag[index+1])





