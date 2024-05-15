import SimpleITK as sitk 
import numpy as np 
from scipy import ndimage
import os 
import argparse
import glob
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root_path', type=str,
                    default='/mnt/data/HM/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingPre', help='Name of Experiment')
parser.add_argument('--data_type', type=str,
                    default='BraTS', help='Data category')
parser.add_argument('--data_name', type=str,
                    default='brats2020_3d', help='Data name') 
parser.add_argument('--savedir', type=str,
                    default='TrResult', help='TsResult for testSet, ValResult for valSet, TrResult for trainSet')
 
parser.add_argument('--model', type=str,
                    default='unet_cct_dropout_3D', help='model_name')
parser.add_argument('--exp', type=str,
                    default='T_weakly_SPS_3d', help='experiment_name')
parser.add_argument('--fold', type=str,
                    default='stage1', help='')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network') 
parser.add_argument('--tt_num', type=int, default=3,
                    help='test times_num') 

parser.add_argument('--func', type=int, default=0,
                    help='function') 
parser.add_argument('--txtName', type=str, default="trainReT04",
                    help='txt name') 

FLAGS = parser.parse_args()


def deal_retrainData_to_volumes(pred_data_path, flag, subdir = "Tr" ):#for retrain word; subdir: Tr, Ts, Val
    input_image_root = FLAGS.data_root_path
    save_sub_dir = input_image_root + "/" + subdir + flag

    if not os.path.exists(save_sub_dir):
        os.makedirs(save_sub_dir)
    volume_num = 0

    pred_path = sorted(
        glob.glob(pred_data_path  + "/*_pred_cuM.nii.gz"))
        
    for pseudo_case in pred_path:
        item_pred_name = pseudo_case.split("/")[-1].split(".")[0]
        # print("item_pred_name:{}".format(item_pred_name))
        item_name = item_pred_name.replace("_pred_cuM", "")
        print("item_name:{}".format(item_name))

        pseudoLab_itk = sitk.ReadImage(pseudo_case)
        pseudoLab = sitk.GetArrayFromImage(pseudoLab_itk)

        image_t1_path =  input_image_root + "/t1/" + item_pred_name.replace("_pred_cuM", "_t1")
        image_t1_itk = sitk.ReadImage(image_t1_path)
        image_t1 = sitk.GetArrayFromImage(image_t1_itk) # image_t1

        image_t1ce_path = image_t1_path.replace("t1", "t1ce")
        image_t1ce_itk = sitk.ReadImage(image_t1ce_path)
        image_t1ce = sitk.GetArrayFromImage(image_t1ce_itk )

        image_t2_path = image_t1_path.replace("t1", "t2")
        image_t2_itk = sitk.ReadImage(image_t2_path)
        image_t2 = sitk.GetArrayFromImage(image_t2_itk)

        image_flair_path = image_t1_path.replace("t1", "flair")
        image_flair_itk = sitk.ReadImage(image_flair_path)
        image_flair = sitk.GetArrayFromImage(image_flair_itk)

        image = np.stack((image_t1, image_t1ce, image_t2, image_flair), axis=0)
        # print("image:{}".format(image.shape))
        
        label_path = image_t1_path.replace("t1", "2label") #gt
        label_itk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_itk)
        # print("label:{}".format(label.shape))

        if image.shape[-3:] != label.shape:
            print("Error, the shape of validating image is not same as label")
 
        f = h5py.File(save_sub_dir + '/{}.h5'.format(item_name), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.create_dataset('pseudoLab', data=pseudoLab, compression ="gzip")
        f.close()
        volume_num += 1
     
    print("Converted data re" + subdir +  " data to volumes")
    print("Total {} re{} volumes".format(volume_num, subdir)) #Total 259  reTr volumes

def write_images_nametxt(flag, subdir = "Tr"):
    input_root = FLAGS.root_path
    output_info_dir = input_root
    
    img_dir = subdir + flag
    info_img_dir = input_root + "/" + img_dir
    img_names = sorted(os.listdir(info_img_dir))
    txtname = "train" + flag.replace("_volumes","") 

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
