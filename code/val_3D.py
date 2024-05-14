import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm

def cal_metric(gt, pred): 
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        # return np.array([dice, hd95])
        return dice
    else:
        # return np.zeros(2)
        return 0

def test_single_case_HM(net, net_type,  image, stride_xy, stride_z, patch_size, num_classes):
    if image.ndim ==3:
        w, h, d = image.shape
    elif image.ndim == 4:
        w, h, d = image.shape[-3:]


    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        if image.ndim ==3:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                (dl_pad, dr_pad)], mode='constant', constant_values=0)
        elif image.ndim == 4:
            image = np.pad(image, [(0,0), (wl_pad, wr_pad), (hl_pad, hr_pad),
                                (dl_pad, dr_pad)], mode='constant', constant_values=0)
    if image.ndim ==3:
        ww, hh, dd = image.shape
    elif image.ndim == 4:
        ww, hh, dd = image.shape[-3:]

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros(ww, hh, dd).astype(np.float32)


    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])

                if image.ndim == 3: 
                    test_patch = image[xs:xs+patch_size[0],
                                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(np.expand_dims(
                        test_patch, axis=0), axis=0).astype(np.float32)   
                if image.ndim == 4: 
                    test_patch = image[:, xs:xs+patch_size[0],
                                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                    test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                    
                test_patch = torch.from_numpy(test_patch).cuda()
                with torch.no_grad():
                    if net_type == "unet_3D_dv_semi":
                        y1, _, _, _ = net(test_patch)
                    elif net_type == "unet_cct_dropout_3D" or net_type == "attention_unet_2dual_3d"\
                        or net_type == "unetr_2dual_3d":
                        y1, _ = net(test_patch)
                    else:
                        y1 = net(test_patch)
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map

def test_all_case_3D(valloader, net, args, stride_xy=32, stride_z=24): 
    total_metric = np.zeros((args.num_classes-1, 1))
    
    for i_batch, sampled_batch in enumerate(valloader):
        image = sampled_batch["image"]
        label = sampled_batch["gt"] 
        image = image.squeeze(0).cpu().detach().numpy()
        label = label.squeeze(0).cpu().detach().numpy()
        prediction = test_single_case_HM(
            net, args.model, image, stride_xy, stride_z, args.patch_size, num_classes=args.num_classes)
        for i in range(1, args.num_classes):
            total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    print("Validation end")
    return total_metric / len(valloader)