import SimpleITK as sitk
import numpy as np 
import maxflow

def graph_cut_for_ACDC_volume(img, scrib):
    I = np.asarray(img * 255, np.float32)
    seg = np.zeros_like(scrib)
    for i in range(1, 4):
        fg_scrib = scrib == i 
        bg_scrib = (scrib != i) * (scrib != 4)

        fP =  0.5*np.ones_like(fg_scrib, np.float32) \
            + np.asarray(fg_scrib, np.float32)*0.5 \
            - np.asarray(bg_scrib, np.float32)*0.5
        bP = 0.5*np.ones_like(fg_scrib, np.float32) \
            - np.asarray(fg_scrib, np.float32)*0.5 \
            + np.asarray(bg_scrib, np.float32)*0.5
        Prob = np.asarray([bP, fP])
        print("Prob shape:{}".format(Prob.shape))
        Prob = np.transpose(Prob, [1, 2, 3, 0])

        Seed = np.asarray([bg_scrib, fg_scrib], np.uint8)
        Seed = np.transpose(Seed, [1, 2, 3, 0])

        lamda = 8.0
        sigma = 10.0
        param = (lamda, sigma)
        lab = maxflow.interactive_maxflow3d(I, Prob, Seed, param)
        seg[lab > 0] = lab[lab > 0] * i 
    return seg


def graph_cut_for_ACDC_slice(img, scrib):
    I = np.asarray(img * 255, np.float32)
    seg = np.zeros_like(scrib)
    for i in range(1, 4):
        fg_scrib = scrib == i 
        bg_scrib = (scrib != i) * (scrib != 4)

        fP =  0.5*np.ones_like(fg_scrib, np.float32) \
            + np.asarray(fg_scrib, np.float32)*0.5 \
            - np.asarray(bg_scrib, np.float32)*0.5
        bP = 0.5*np.ones_like(fg_scrib, np.float32) \
            - np.asarray(fg_scrib, np.float32)*0.5 \
            + np.asarray(bg_scrib, np.float32)*0.5
        Prob = np.asarray([bP, fP]) #Prob shape:(2, 256, 216)
        Prob = np.transpose(Prob, [1, 2,  0])

        Seed = np.asarray([bg_scrib, fg_scrib], np.uint8)
        Seed = np.transpose(Seed, [1, 2,  0])

        lamda = 8.0
        sigma = 10.0
        param = (lamda, sigma)
        lab = maxflow.interactive_maxflow2d(I, Prob, Seed, param)
        seg[lab > 0] = lab[lab > 0] * i 
    return seg