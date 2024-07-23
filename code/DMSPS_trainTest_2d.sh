export PYTHONPATH==$PYTHONPATH:/mnt/data1/HM/projects/word_hm/code
## example: CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 nohup python main.py &

# train for stage1
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train/A_train_weaklySup_SPS_2d_soft.py\
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage1 --sup_type scribble --trainData train.txt

# test for stage1
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python test/test_2d_forall_fast_txtver.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage1

#test on trainSet and get the uncertainty-filterd pseudo-label [confident expanded annotation]
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python test/test_2d_forall_fast_txtver_forTrainSetUncertaintyOnly_Mean.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage1 --threshold 0.1 --tt_num 3

# deal with the produced confident expanded annotation into h5 file and get the txt
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python dataloader/retrain_postProcess_ACDC_uncertainty.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC \
    --func 0 --txtName trainReT01
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python dataloader/retrain_postProcess_ACDC_uncertainty.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC \
    --func 1 --txtName trainReT01

# train for stage2
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train/A_train_weaklySup_SPS_2d_soft_retrainUncertainty.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage2 --sup_type pseudoLab --trainData trainReT01.txt

# test for stage2
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python test/test_2d_forall_fast_txtver.py \
    --data_root_path /mnt/data/HM/Datasets/ACDC2017/ACDC_for2D \
    --model unet_cct --exp A_weakly_SPS_2d --fold stage2





