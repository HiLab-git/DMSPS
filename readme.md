# DMSPS
official code for: DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation. MedIA 2024 [MedIA](https://www.sciencedirect.com/science/article/pii/S1361841524001993?dgcid=author).
And the previous version is published on the [MICCAI](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_50) 2022.

### Overall Framework
The overall framework of DMSPS：
![overall](https://github.com/HiLab-git/DMSPS/blob/master/imgs/framework.png)

### Citation
If you use this project in your research, please cite the following works:
```
@article{han2024dmsps,
  title={DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation},
  author={Han, Meng and Luo, Xiangde and Xie, Xiangjiang and Liao, Wenjun and Zhang, Shichuan and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  journal={Medical Image Analysis},
  pages={103274},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{luo2022scribble,
  title={Scribble-supervised medical image segmentation via dual-branch network and dynamically mixed pseudo labels supervision},
  author={Luo, Xiangde and Hu, Minhao and Liao, Wenjun and Zhai, Shuwei and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={528--538},
  year={2022},
  organization={Springer}
}
```

# Dataset
* The ACDC dataset with mask annotations can be downloaded from: [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html).
* The Scribble annotations of ACDC can be downloaded from: [Scribble](https://gvalvano.github.io/wss-multiscale-adversarial-attention-gates/data).
* You can also download the ACDC data from the following Baidu disk link, where the same training set, test set and verification set as in DMSPS have been divided: [ACDC](https://pan.baidu.com/s/1Wqcw_qFNezplzdewQMHXsg). The extraction code is：et38 .

# Usage: [Taking the ACDC segmentation task as an example]
### Step0:
1. Clone this project. 
```
git clone https://github.com/HiLab-git/DMSPS
cd DMSPS
```
2. Data pre-processing os used or the processed data.
```
cd code/dataloaders
python preprocess_ACDC.py
```

### Quick test using pre-trained checkpoints:
1. the first stage 
```
cd code/test
python test_2d_forall_fast_txtver.py --data_name Heart_ACDC_Example \
--exp A_weakly_SPS_2d --fold stage1 --model unet_cct --tt_num 1
```
2. the second stage
```
python test_2d_forall_fast_txtver.py --data_name Heart_ACDC_Example \
--exp A_weakly_SPS_2d --fold stage2 --model unet_cct --tt_num 1
```

### Train and test for the first stage 
1. Train the model 
```
cd code/train
python A_train_weaklySup_SPS_2d_soft.py
```
2. Test the model 
```
cd code/test
python test_2d_forall_fast_txtver.py 
```

### Train and test for the second stage 
1. test on trainSet and get the uncertainty-filterd pseudo-label
```
cd code/test
python test_2d_forall_fast_txtver_forTrainSetUncertaintyOnly_Mean.py \
    --data_root_path $yourPath/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage1 --threshold 0.1 --tt_num 3
```
2. deal with the produced confident expanded annotation into h5 file and get the txt
```
cd code/dataloader
python retrain_postProcess_ACDC_uncertainty.py \
    --data_root_path  $yourPath/ACDC2017/ACDC \
    --func 0 --txtName trainReT01
python retrain_postProcess_ACDC_uncertainty.py \
    --data_root_path  $yourPath/ACDC2017/ACDC \
    --func 1 --txtName trainReT01
```
3. train for stage2
```
cd code/train
python A_train_weaklySup_SPS_2d_soft_retrainUncertainty.py \
    --data_root_path  $yourPath/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage2 \
    --sup_type pseudoLab --trainData trainReT01.txt
```
4. test for stage2
```
cd code/test
python test_2d_forall_fast_txtver.py \
    --data_root_path $yourPath/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage2
```

### Acknowledgement
The code of scribble-supervised learning framework is borrowed from [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)