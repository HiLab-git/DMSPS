# 从网上复制的画cam图的代码，还没有改。好像原本是给分类做的
#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
 
def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    draw Class Activation Map
    :param model: load Pytorch model
    :param img_path: test iamge path
    :param save_path: save path
    :param transform: Input image preprocessing method
    :param visual_heatmap: Whether to visualize the original heatmap (calling matplotlib)
    :return:
    '''
    img = Image.open(img_path).convert('RGB')
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
 
    model.eval()
    features = model.features(img)
    output = model.classifier(features)
 
    def extract(g):
        global features_grad
        features_grad = g
 

    pred = torch.argmax(output).item()
    pred_class = output[:, pred]
 
    features.register_hook(extract)
    pred_class.backward() 
 
    grads = features_grad   
 
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
 
    pooled_grads = pooled_grads[0]
    features = features[0]

    for i in range(512):
        features[i, ...] *= pooled_grads[i, ...]
 

    heatmap = features.detach().numpy()
    heatmap = np.mean(heatmap, axis=0)
 
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
 
  
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()
 
    img = cv2.imread(img_path)  
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) 
    heatmap = np.uint8(255 * heatmap)  
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  
    superimposed_img = heatmap * 0.4 + img  
    cv2.imwrite(save_path, superimposed_img)  
