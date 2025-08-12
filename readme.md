# DMSPS
Official code for: DMSPS: Dynamically mixed soft pseudo-label supervision for scribble-supervised medical image segmentation. [MedIA 2024](https://www.sciencedirect.com/science/article/pii/S1361841524001993?dgcid=author).
And the previous version is published on the [MICCAI 2022](https://link.springer.com/chapter/10.1007/978-3-031-16431-6_50).

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
* For ACDC dataset, the images, scribbles and data split can be downloaded from [ACDC BaiduPan](https://pan.baidu.com/s/1Wqcw_qFNezplzdewQMHXsg) (The extraction code is：et38), or [Google Drive](). 
* The WORD dataset can be downloaded from [WORD](https://github.com/HiLab-git/WORD?tab=readme-ov-file).
* The BraTS2020 dataset can be downloaded from [BraTS2020](https://www.med.upenn.edu/cbica/brats2020/data.html). Note that this work aimed to segment two foreground classes: the tumor core and the peritumoral
edema. Scribbles could be genearted by this simulation code: [scribble-generate](https://github.com/HiLab-git/DMSPS/blob/master/code/dataloader/scribble_generater.py)

# Usage with PyMIC
To facilitate the use of code and make it easier to compare with other methods, we have re-implemented DMSPS in PyMIC, a Pytorch-based framework for annotation-efficient segmentation. The core modules of DMSPS in PyMIC can be found [here][pymic_dmsps]. It is suggested to use PyMIC for this experiment. In the following, we take the ACDC dataset as an example for scribble-supervised segmentation.

[pymic_dmsps]: https://github.com/HiLab-git/PyMIC/blob/master/pymic/net_run/weak_sup/wsl_dmsps.py

### Step 0: Preparation
1. Install PyMIC. 
```
pip install pymic==0.5.4
```
2. Dataset convert.
To speed up the training process, we convert the data into h5 files.
```
python image_process.py 0
```
To run this code, you need to set `ACDC_DIR` to the path of the  ACDC dataset after download.

### Step 1: Training the first stage model
1. The configurations including dataset, network, optimizer and hyper-parameters are contained in the configure file
`config/acdc_dmsps.cfg`. Train the first-stage model by running:
```
python run.py train config/acdc_dmsps.cfg
```
2. Obtain predictions for testing images:
```
python run.py test config/acdc_dmsps.cfg
```
3. Obtain quantitative evaluation results:
```
pymic_eval_seg --metric dice --cls_num 4 \
  --gt_dir data/ACDC2017/ACDC/TestSet/labels --seg_dir ./result/acdc_dmsps \
  --name_pair ./config/image_test_gt_seg.csv
```
The average Dice on the test set would be around 88.94%.

4. Then obtain the expanded seeds based on confident predictions from the first stage model.
```
python run.py test config/acdc_dmsps.cfg  --test_csv data/ACDC2017/ACDC_for2D/trainvol.csv \
  --dmsps_test_mode 1 --output_dir result/acdc_dmsps_train
```
5. Create the training set for the second-stage model:
```
python image_process.py 1
```
### Step 2: Training the second stage model
1. Just like the first stage, train the model use:
```
python run.py train config/acdc_dmsps_stage2.cfg
```
Note that to speedup the training process, we finetune the first-stage model here. You may also try to train from scratch. 

2. Test with the second-stage model:
```
python run.py test config/acdc_dmsps_stage2.cfg
```
3. Obtain quantitative evaluation results:
```
pymic_eval_seg --metric dice --cls_num 4 \
  --gt_dir data/ACDC2017/ACDC/TestSet/labels --seg_dir ./result/acdc_dmsps_stage2 \
  --name_pair ./config/image_test_gt_seg.csv
```
The average Dice on the test set would be around 89.51%.
### Step 3: Training with sparser annotations
The original scribbles for the ACDC dataset was quite dense, and to investigate the performance under sparser annotations, we have reduced the scribble length to 1/2, 1/4, 1/8 and 1/16, respectively. For example, to train and test the model with scribbles at length of 1/4,  run:
```
python run.py train config/acdc_dmsps_r4.cfg
python run.py test config/acdc_dmsps_r4.cfg
pymic_eval_seg --metric dice --cls_num 4 \
  --gt_dir data/ACDC2017/ACDC/TestSet/labels --seg_dir ./result/acdc_dmsps_r4 \
  --name_pair ./config/image_test_gt_seg.csv
```
The average Dice on the test set would be around 86.78%.

### Step 4: Compare with other weakly supervised segmentation methods
PyMIC also provides implementation of several other weakly supervised methods (learning from scribbles). Please see [PyMIC_examples/seg_weak_sup/ACDC][PyMIC_example_link] for examples.

[PyMIC_example_link]:https://github.com/HiLab-git/PyMIC_examples/tree/main/seg_weak_sup/ACDC 

# Usage (a backup of the old version)
### Step0:
1. Clone this project. 
```
git clone https://github.com/HiLab-git/DMSPS
cd DMSPS
```
2. Data pre-processing.
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