import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage
import os 
import argparse
import glob
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/Abdomen_word/WORD-V0.1.0-Admin_cropWL', help='Name of Experiment')
parser.add_argument('--data_type', type=str,
                    default='AbdomenM', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='word_3d', help='Data name') 
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')
 
parser.add_argument('--model', type=str,
                    default='unet_cct_dropout_3D', help='model_name')
parser.add_argument('--exp', type=str,
                    default='A3_weakly_PLS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times num') 

parser.add_argument('--func', type=int, default=0,
                    help='function') 
parser.add_argument('--txtName', type=str, default="trainReT03",
                    help='txt name') 

FLAGS = parser.parse_args()


def deal_retrainData_to_volumes(pred_data_path, flag, subdir = "Tr" ):#for retrain word; subdir: Tr, Ts, Val
    # resample 1,1,1: 12106  train slices for training 
    # original: Total 1356  train slices

    input_image_root = FLAGS.data_root_path 
    save_root_path = FLAGS.data_root_path + "_for3D"
    save_sub_dir = save_root_path + "/" + flag

    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)
    volume_num = 0

    image_path = sorted(
        glob.glob(input_image_root + "/labels" + subdir + "/*.nii.gz"))
        
    for case in image_path:
        item_name = case.split("/")[-1].split(".")[0]
        print(item_name)

        label_itk = sitk.ReadImage(case) #gt
        label = sitk.GetArrayFromImage(label_itk)  
        
        image_path = case.replace("labels","images")
        image_itk = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(image_itk)

        pred_name = item_name + "_pred_cuM.nii.gz"
        pseudoLab_path = pred_data_path+"/"+pred_name
        pseudoLab_itk = sitk.ReadImage(pseudoLab_path)
        pseudoLab = sitk.GetArrayFromImage(pseudoLab_itk)
        print(pred_name)

        if image.shape != label.shape:
            print("Error, the shape of validating image is not same as label")
 
        f = h5py.File(save_sub_dir + '/{}.h5'.format(item_name), 'w')
        f.create_dataset(
            'image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
       
        f.create_dataset('pseudoLab', data=pseudoLab, compression ="gzip")
        f.close()
        volume_num += 1
     
    print("Converted data re" + subdir +  " data to  volumes")
    print("Total {}  re{} volumes".format(volume_num, subdir))

def write_images_nametxt(flag, subdir = "Tr"):
    input_root = FLAGS.data_root_path  + "_for3D"
    output_info_dir = input_root
    
    img_dir = subdir + flag
    info_img_dir = input_root + "/" + img_dir
    img_names = sorted(os.listdir(info_img_dir)) 
    txtname = flag.replace("_volumes","") 

    f = open(output_info_dir + "/" + txtname +".txt", "w")
    for img_name in img_names:
        f.write(os.path.join(img_dir,img_name) + "\n")
    f.close()



if __name__ == "__main__":

    input_test_save_path = "../../result/{}_{}/{}_{}_{}_{}_{}".format(
        FLAGS.data_type, FLAGS.data_name, FLAGS.exp, FLAGS.model, FLAGS.fold, FLAGS.savedir, FLAGS.tt_num)
   
    func =FLAGS.func
    flag = FLAGS.txtName+"_volumes"

    if func == 0:
        deal_retrainData_to_volumes(input_test_save_path, flag)
    elif func == 1:
        write_images_nametxt(flag)
