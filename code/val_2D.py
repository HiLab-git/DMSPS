import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import logging
import math
from tqdm import tqdm
import h5py

def calculate_metric_percase(pred, gt): #2D
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt, voxelspacing=[2, 1, 1])
        # asd = metric.binary.asd(pred, gt, voxelspacing=[2, 1, 1]) 
        return dice
    else:
        return 0

def test_single_case_resizeBlock_2d(net, net_type, image, patch_size, batch_size): #generate prediction
    image = image.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(image)
        ind_x = np.array([i for i in range(image.shape[0])])#下标
        for ind in ind_x[::batch_size]:
            if ind + batch_size < image.shape[0]:
                slice = image[ind:ind + batch_size, ...]
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    if net_type == "unet_cct":
                        out, _ = net(input)
                    else: 
                        out = net(input)
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:ind + batch_size, ...] = pred
            else:
                slice = image[ind:, ...] 
                thickness, x, y = slice.shape[0], slice.shape[1], slice.shape[2]
                slice = zoom(
                    slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
                input = torch.from_numpy(slice).unsqueeze(1).float().cuda()
                net.eval()
                with torch.no_grad():
                    if net_type == "unet_cct":
                        out, _ = net(input)
                    else: 
                        out = net(input)
                    out = torch.argmax(torch.softmax(out, dim=1), dim=1)
                    out = out.cpu().detach().numpy()
                    pred = zoom(
                        out, (1, x / patch_size[0], y / patch_size[1]), order=0)
                    prediction[ind:, ...] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if net_type == "unet_cct":
                out, _ = net(input)
            else: 
                out = net(input)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1)        
            prediction = out.cpu().detach().numpy()   
    return prediction 


# main test 
def test_all_case_2D(valloader, net, args): 
    total_metric = np.zeros((args.num_classes-1, 1))

    for i_batch, sampled_batch in enumerate(valloader):
        image = sampled_batch["image"]
        label = sampled_batch["gt"]
        label = label.squeeze(0).cpu().detach().numpy()
        prediction  = test_single_case_resizeBlock_2d(net, args.model, image, args.patch_size, args.batch_size)
        for i in range(1, args.num_classes):
            total_metric[i-1, :] += calculate_metric_percase(label == i, prediction == i)
    
    print("Validation end")
    return total_metric / len(valloader)
    
    
                      