import torch
import numpy as np
import PIL 
import torchvision.transforms.functional as F
import torch


def Cutout(imgs,labels, n_holes=1, length= 16):
    labels_list = []
    imgs_list = []
    masks_list = []
    num = imgs.shape[0]

    if imgs.dim() == 4: 
        h = imgs.shape[2]
        w = imgs.shape[3]
    elif imgs.dim() == 5:
        d = imgs.shape[2]
        h = imgs.shape[3] 
        w = imgs.shape[4] 
    else:
        raise ValueError("the dimension number should be 4 or 5")


    for i in range(num):
        if imgs.dim() == 4: 
            label = labels[i,:,:,:]
            img = imgs[i,:,:,:]
            mask = np.ones((1, h, w), np.float32)
        elif imgs.dim() == 5: 
            label = labels[i,:,:,:, :]
            img = imgs[i,:,:,:, :]
            mask = np.ones((1, d, h, w), np.float32)
        mask = torch.from_numpy(mask)
        # mask = mask.to(device)
        mask = mask.cuda()

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            if imgs.dim() == 4: 
                mask[0, y1: y2, x1: x2] = 0.
            elif imgs.dim() == 5: 
                z = np.random.randint(d)
                z1 = np.clip(z - length // 2, 0, d)
                z2 = np.clip(z + length // 2, 0, d)
                mask[0, z1:z2, y1: y2, x1: x2] = 0.
        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask
        imgs_list.append(img)
        labels_list.append(label)
        masks_list.append(mask)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)
    masks_out = torch.stack(masks_list)

    return imgs_out, labels_out, masks_out


def rotate_invariant(imgs,labels):
    num = imgs.shape[0]
    imgs_out_list = []
    labels_out_list = []
    angles = []
    
    for i in range(num):
        if imgs.dim() == 4: 
            img = imgs[i, :, :, :]
            label = labels[i, :, :, :]
        elif imgs.dim() == 5:
            img = imgs[i,:, :, :, :]
            label = labels[i, :, :, :, :]

        angle = float(torch.empty(1).uniform_(0.0, 360.0).item())
        
        rotated_img = F.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_label = F.rotate(label, angle, PIL.Image.NEAREST, False, None)
        
        imgs_out_list.append(rotated_img)
        labels_out_list.append(rotated_label)
        
        angles.append(angle)
    
    imgs_out = torch.stack(imgs_out_list)
    labels_out = torch.stack(labels_out_list)
    return imgs_out, labels_out, angles


def rotate_back(imgs,outputs,labels,angles):
    num = imgs.shape[0]
    imgs_out_list = []
    outputs_out_list = []
    labels_out_list = []
    
    for i in range(num):
        if imgs.dim() == 4: 
            img = imgs[i,:,:,:]
            output = outputs[i,:,:,:]
            label = labels[i,:,:,:]
        elif imgs.dim() == 5:
            img = imgs[i,:, :, :, :]
            output = outputs[i,:,:,:, :]
            label = labels[i, :, :, :, :]

        angle = -angles[i]
        
        rotated_img = F.rotate(img, angle, PIL.Image.NEAREST, False, None)
        rotated_output = F.rotate(output, angle, PIL.Image.NEAREST, False, None)
        rotated_label = F.rotate(label, angle, PIL.Image.NEAREST, False, None)
        
        imgs_out_list.append(rotated_img)
        outputs_out_list.append(rotated_output)
        labels_out_list.append(rotated_label)
    
    imgs_out = torch.stack(imgs_out_list)
    outputs_out = torch.stack(outputs_out_list) 
    labels_out = torch.stack(labels_out_list)
    return imgs_out, outputs_out, labels_out