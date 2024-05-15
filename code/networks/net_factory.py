from networks.efficientunet import Effi_UNet
from networks.pnet import PNet2D
from networks.unet import UNet, UNet_DS, UNet_CCT
from networks.unet_nest import NestedUNet2D
from networks.unet_nest_2dual import NestedUNet2D_2dual
from networks.swinunet import SwinUNet
from networks.swinunet_2dual import SwinUNet_2dual


def net_factory(net_type="unet", in_chns=1, class_num=3):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type =="NestedUNet2d":
        net = NestedUNet2D(in_chns = in_chns,class_num=class_num).cuda()
    elif net_type == "NestedUNet2d_2dual":
        net = NestedUNet2D_2dual(in_chns = in_chns,class_num=class_num).cuda()
    elif net_type == "swinunet":
        net = SwinUNet(class_num=class_num).cuda() 
    elif net_type == "swinunet_2dual":
        net = SwinUNet_2dual(class_num=class_num).cuda() 
    else:
        net = None
    return net
