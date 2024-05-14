from networks.unet_3D import unet_3D
from networks.vnet import VNet
from networks.VoxResNet import VoxResNet
from networks.attention_unet import Attention_UNet
from networks.attention_unet_2dual import Attention_UNet_2dual
from networks.nnunet import initialize_network
from networks.unet_cct_3D import UNet_CCT_dp_3D
from networks.UNETR_monai import UNETR
from networks.UNETR_monai_2dual import UNETR_2dual

from networks.unet_3D_dv_semi import unet_3D_dv_semi, unet_3D_dv


def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2):
    if net_type == "unet_3D":
        net = unet_3D( in_channels = in_chns, n_classes = class_num).cuda()
    elif net_type == "unet_cct_dp_3D":
        net = UNet_CCT_dp_3D(in_channels = in_chns, n_classes = class_num).cuda()
    elif net_type == "attention_unet_3d":
        net = Attention_UNet(in_channels = in_chns, n_classes = class_num).cuda()
    elif net_type == "attention_unet_2dual_3d":
        net = Attention_UNet_2dual(in_channels = in_chns, n_classes = class_num).cuda()
    elif net_type == "unetr_3d":
        net = UNETR(in_channels =in_chns, out_channels = class_num,  img_size = (80,96,96)).cuda()
    elif net_type == "unetr_2dual_3d":
        net = UNETR_2dual(in_channels =in_chns, out_channels = class_num,  img_size = (80,96,96)).cuda()

    elif net_type == "voxresnet":
        net = VoxResNet(in_chns = in_chns, feature_chns = 64,
                        class_num = class_num).cuda()
    elif net_type == "vnet":
        net = VNet(n_channels = in_chns, n_classes = class_num,
                   normalization='batchnorm', has_dropout=True).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()

    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(in_channels = in_chns, n_classes = class_num).cuda()
    else:
        net = None
    return net
    
