import numpy as np

def pseudo_label_generator_abdomen(data, seed, unlabeled_index = 255, beta=100, class_num = 8, mode='bf'):
    from skimage.segmentation import random_walker
    labeled_organs_num = 0 
    index_flag = [0] * class_num 
    for index in range(1,class_num):
        if index in np.unique(seed):
            labeled_organs_num += 1
            index_flag[index]=1
    if labeled_organs_num == 0:
        pseudo_label = np.zeros_like(seed)
        print("seed:{}".format(np.unique(seed)))
    else:
        markers = np.ones_like(seed)
        markers[seed == unlabeled_index] = 0 
        for i in range(0,class_num):
            markers[seed == i] = i+1 

        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label_un =  segmentation - 1 
    
        pseudo_label = np.zeros_like(pseudo_label_un)
        organ_num = 1
        for i in range(class_num):
            if organ_num>labeled_organs_num:
                break 
            if index_flag[i]==1:
                pseudo_label[pseudo_label_un==organ_num] = i
                organ_num += 1
    return pseudo_label


def pseudo_label_generator_acdc(data, seed, beta=100, mode='bf'):
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import random_walker
    if 1 not in np.unique(seed) or 2 not in np.unique(seed) or 3 not in np.unique(seed):
        pseudo_label = np.zeros_like(seed)
    else:
        markers = np.ones_like(seed)
        markers[seed == 4] = 0
        markers[seed == 0] = 1
        markers[seed == 1] = 2
        markers[seed == 2] = 3
        markers[seed == 3] = 4
        sigma = 0.35
        data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                                 out_range=(0, 1))
        segmentation = random_walker(data, markers, beta, mode)
        pseudo_label =  segmentation - 1
    return pseudo_label