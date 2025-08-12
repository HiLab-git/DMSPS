
# PyMICPATH=/home/disk4t/projects/PyMIC_project/PyMIC
# export PYTHONPATH=$PYTHONPATH:/home/disk4t/student_code/DMSPS/code


# python image_process.py 0
# pymic_train config/acdc_dmsps.cfg
python run.py train config/acdc_dmsps_r4.cfg

# pymic_test config/acdc_dmsps.cfg
# pymic_eval_seg --metric dice --cls_num 4 \
#   --gt_dir data/ACDC2017/ACDC/TestSet/labels --seg_dir ./result/acdc_dmsps \
#   --name_pair ./config/image_test_gt_seg.csv
# pymic_test config/acdc_dmsps.cfg  --test_csv data/ACDC2017/ACDC_for2D/trainvol.csv \
#   --dmsps_test_mode 1 --output_dir result/acdc_dmsps_train
# python image_process.py 1
# pymic_train config/acdc_dmsps_stage2.cfg
# pymic_test config/acdc_dmsps_stage2.cfg
# pymic_eval_seg --metric dice --cls_num 4 \
#   --gt_dir data/ACDC2017/ACDC/TestSet/labels --seg_dir ./result/acdc_dmsps_stage2 \
#   --name_pair ./config/image_test_gt_seg.csv
