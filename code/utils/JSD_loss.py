import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np


def calc_jsd_multiscale(weight, labels1_a, pred1, pred2, pred3, threshold=0.8, Mask_label255_sign='no',num_classes=17):

    Mask_label255 = (labels1_a < 2).float()  # do not compute the area that is irrelavant[无关紧要的] (dataaug)  b,h,w
    weight_softmax = F.softmax(weight, dim=0)

    criterion = CrossEntropyLoss(ignore_index=num_classes, reduction='none')

    labels1_a = labels1_a.squeeze(axis=1).long()
    loss1 = criterion(pred1 * weight_softmax[0], labels1_a)  # * weight_softmax[0]
    loss2 = criterion(pred2 * weight_softmax[1], labels1_a)  # * weight_softmax[1]
    loss3 = criterion(pred3 * weight_softmax[2], labels1_a)  # * weight_softmax[2]

    loss = (loss1 + loss2 + loss3) #softmx函数是指数归一化函数。 

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]

    weighted_probs = [weight_softmax[i] * prob for i, prob in enumerate(probs)] 
    
    mixture_label = (torch.stack(weighted_probs)).sum(axis=0) #按照权重将三个pred叠起来产生新的pred
    #mixture_label = torch.clamp(mixture_label, 1e-7, 1)  # h,c,h,w
    mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # h,c,h,w

    # add this code block for early torch version where torch.amax is not available
    if torch.__version__=="1.5.0" or torch.__version__=="1.6.0":
        _, max_probs = torch.max(mixture_label*Mask_label255.unsqueeze(1), dim=-3, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-2, keepdim=True)
        _, max_probs = torch.max(max_probs, dim=-1, keepdim=True)
    else:
        max_probs = torch.amax(mixture_label*Mask_label255.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    mask = max_probs.ge(threshold).float()


    logp_mixture = mixture_label.log()

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    if Mask_label255_sign == 'yes':
        consistency = sum(log_probs)*Mask_label255
    else:
        consistency = sum(log_probs)

    return torch.mean(loss), torch.mean(consistency), consistency, mixture_label

def calc_jsd_multiscale_no_weight_3d(label, pred1, pred2, pred3, threshold=0.8,num_classes = 17):
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    loss_ce1 = ce_loss(pred1, label[:].long())
    loss_ce2 = ce_loss(pred2, label[:].long())
    loss_ce3 = ce_loss(pred3, label[:].long())
    loss_ce = (loss_ce1 + loss_ce2 + loss_ce3)
    # print("loss_ce:{}".format(loss_ce)) # loss_ce是数

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    # print("probs_shape_len:{}".format(len(probs))) # list 3 item
    # print("probs_shape:{}".format(probs[0].shape)) # probs_shape:torch.Size([1, 8, 96, 96, 96])
    consistency = 0.0
    #计算kl散度
    consistency += F.kl_div(probs[1].log(), probs[0], reduction='none')
    consistency += F.kl_div(probs[2].log(), probs[0], reduction='none')
    consistency += F.kl_div(probs[0].log(), probs[1], reduction='none')
    consistency += F.kl_div(probs[2].log(), probs[1], reduction='none')
    consistency += F.kl_div(probs[0].log(), probs[2], reduction='none')
    consistency += F.kl_div(probs[1].log(), probs[2], reduction='none')
    #  print("consistency_shape_before:{}".format(consistency.shape)) #consistency_shape_before:torch.Size([1, 8, 96, 96, 96])
    consistency = torch.sum(consistency, dim=1)
    #  print("consistency_shape_after:{}".format(consistency.shape)) # consistency_shape_after:torch.Size([1, 96, 96, 96])
    
    # mixture_label = (torch.stack(probs)).sum(axis=0) #按照batch_size的维度堆起来
    # # print("mixture_label_shape_before:{}".format( mixture_label.shape)) 
    # # #mixture_label_shape_before:torch.Size([1, 8, 96, 96, 96])
    # mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # b,c,d,h,w
    # # print("mixture_label_shape_after:{}".format( mixture_label.shape))
    # # # mixture_label_shape_after:torch.Size([1, 8, 96, 96, 96])
    # max_probs = torch.amax(mixture_label.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    # # print("max_probs_shape:{}".format( max_probs.shape))
    # # # max_probs_shape:torch.Size([1, 1, 8, 1, 1, 1])
    # mask = max_probs.ge(threshold).float() 
    # # print("mask_shape:{}".format( mask.shape))
    # # # mask_shape:torch.Size([1, 1, 8, 1, 1, 1])
    # #torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数。大于等于相应位置返回1，小于返回0.
    # logp_mixture = mixture_label.log()
    # log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    # here can replace [dim=1] into [keepdim=True] and mask no longer use 'unaqueeze' to increase a dim 
    # consistency = sum(log_probs)
    # # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # #consistency_shape_after_mixture:torch.Size([1, 8, 96, 96, 96])
    # # print(consistency)
    
    loss = (torch.mean(loss_ce) + torch.mean(consistency))/3
    return loss

def calc_jsd_multiscale_no_weight_sum_3d(label, pred1, pred2, pred3, threshold=0.8, num_classes = 8):
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    loss_ce1 = ce_loss(pred1, label[:].long())
    loss_ce2 = ce_loss(pred2, label[:].long())
    loss_ce3 = ce_loss(pred3, label[:].long())
    loss_ce = (loss_ce1 + loss_ce2 + loss_ce3)
    # print("loss_ce:{}".format(loss_ce)) # loss_ce是数

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    # print("probs_shape_len:{}".format(len(probs))) # list 3 item
    # print("prob_shape:{}".format(probs[0].shape)) # prob_shape:torch.Size([1, 8, 96, 96, 96])
    # consistency = 0.0
    # #计算kl散度
    # consistency += F.kl_div(probs[1].log(), probs[0], reduction='none')
    # consistency += F.kl_div(probs[2].log(), probs[0], reduction='none')
    # consistency += F.kl_div(probs[0].log(), probs[1], reduction='none')
    # consistency += F.kl_div(probs[2].log(), probs[1], reduction='none')
    # consistency += F.kl_div(probs[0].log(), probs[2], reduction='none')
    # consistency += F.kl_div(probs[1].log(), probs[2], reduction='none')
    # #  print("consistency_shape_before:{}".format(consistency.shape)) #consistency_shape_before:torch.Size([1, 8, 96, 96, 96])
    # consistency = torch.sum(consistency, dim=1)
    #  print("consistency_shape_after:{}".format(consistency.shape)) # consistency_shape_after:torch.Size([1, 96, 96, 96])
    
    mixture_label = (torch.stack(probs)).sum(axis=0) #按照batch_size的维度堆起来
    # print("probs:{}".format(probs[0]))
    # print("\n \n \n")
    # print("mixture_label:{}".format(mixture_label))
    # print("mixture_label_shape_before:{}".format( mixture_label.shape)) 
    # #mixture_label_shape_before:torch.Size([1, 8, 96, 96, 96])
    mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # b,c,d,h,w
    # print("mixture_label_shape_after:{}".format( mixture_label.shape))
    # # mixture_label_shape_after:torch.Size([1, 8, 96, 96, 96])
    max_probs = torch.amax(mixture_label.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)# torch.amax取出最大的数值
    # print("max_probs:{}".format(max_probs))
    # print("max_probs_shape:{}".format( max_probs.shape))
    # # max_probs_shape:torch.Size([1, 1, 8, 1, 1, 1])
    mask = max_probs.ge(threshold).float() 
    # print("mask_shape:{}".format( mask.shape))
    # # mask_shape:torch.Size([1, 1, 8, 1, 1, 1])
    #torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数。大于等于相应位置返回1，小于返回0.
    logp_mixture = mixture_label.log()
    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])

    consistency = sum(log_probs)
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    #consistency_shape_after_mixture:torch.Size([1, 8, 96, 96, 96])
    # print(consistency)
    consistency = torch.sum(consistency, dim=1)#这一步可能是FJ加的
    
    loss = (torch.mean(loss_ce) + torch.mean(consistency))/3#这一步可能是FJ加的
    return loss

def calc_jsd_multiscale_no_weight_mean_3d(label, pred1, pred2, pred3, threshold=0.5, num_classes = 8):
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    loss_ce1 = ce_loss(pred1, label[:].long())
    loss_ce2 = ce_loss(pred2, label[:].long())
    loss_ce3 = ce_loss(pred3, label[:].long())
    loss_ce = (loss_ce1 + loss_ce2 + loss_ce3)
    # print("loss_ce:{}".format(loss_ce)) # loss_ce是数

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    # print("probs_shape_len:{}".format(len(probs))) # list 3 item
    # print("prob_shape:{}".format(probs[0].shape)) # prob_shape:torch.Size([1, 8, 96, 96, 96])
    # consistency = 0.0
    # #计算kl散度
    # consistency += F.kl_div(probs[1].log(), probs[0], reduction='none')
    # consistency += F.kl_div(probs[2].log(), probs[0], reduction='none')
    # consistency += F.kl_div(probs[0].log(), probs[1], reduction='none')
    # consistency += F.kl_div(probs[2].log(), probs[1], reduction='none')
    # consistency += F.kl_div(probs[0].log(), probs[2], reduction='none')
    # consistency += F.kl_div(probs[1].log(), probs[2], reduction='none')
    # #  print("consistency_shape_before:{}".format(consistency.shape)) #consistency_shape_before:torch.Size([1, 8, 96, 96, 96])
    # consistency = torch.sum(consistency, dim=1)
    #  print("consistency_shape_after:{}".format(consistency.shape)) # consistency_shape_after:torch.Size([1, 96, 96, 96])
    
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来
    # print("probs:{}".format(probs[0]))
    # print("\n \n \n")
    # print("mixture_label:{}".format(mixture_label))
    # print("mixture_label_shape_before:{}".format( mixture_label.shape)) 
    # #mixture_label_shape_before:torch.Size([1, 8, 96, 96, 96])
    mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # b,c,d,h,w
    # print("mixture_label_shape_after:{}".format( mixture_label.shape))
    # # mixture_label_shape_after:torch.Size([1, 8, 96, 96, 96])
    max_probs = torch.amax(mixture_label.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)# torch.amax取出最大的数值
    # print("max_probs:{}".format(max_probs))
    # print("max_probs_shape:{}".format(max_probs.shape))
    # # max_probs_shape:torch.Size([1, 1, 8, 1, 1, 1])
    mask = max_probs.ge(threshold).float() 
    # print("mask_shape:{}".format( mask.shape))
    # # mask_shape:torch.Size([1, 1, 8, 1, 1, 1])
    #torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数。大于等于相应位置返回1，小于返回0.
    logp_mixture = mixture_label.log()
    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])

    consistency = sum(log_probs)
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    #consistency_shape_after_mixture:torch.Size([1, 8, 96, 96, 96])
    # print(consistency)
    consistency = torch.sum(consistency, dim=1)
    
    loss = (torch.mean(loss_ce) + torch.mean(consistency))/3
 
    return loss

def calc_jsd_multiscale_no_weight_mean_HM_3d(label, pred1, pred2, pred3, threshold=0.5, num_classes = 8):
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    loss_ce1 = ce_loss(pred1, label[:].long())
    loss_ce2 = ce_loss(pred2, label[:].long())
    loss_ce3 = ce_loss(pred3, label[:].long())
    loss_ce = (loss_ce1 + loss_ce2 + loss_ce3)
    # print("loss_ce_shape:{}".format(loss_ce.shape))
    # loss_ce_shape:torch.Size([])

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来
    logp_mixture = mixture_label.log()
    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])

    consistency = sum(log_probs)#按照list的维度相加
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # consistency_shape_after_mixture:torch.Size([1, 96, 96, 96]) # batch_size x D x H x W
    
    loss = (torch.mean(loss_ce) + torch.mean(consistency))/3# torch.mean不设置参数的时候，是返回所有元素的平均值
    # print("loss_type:{}".format(type(loss)))#loss_type:<class 'torch.Tensor'>
    return loss

def calc_jsd_multiscale_no_weight_2d(label, pred1, pred2, pred3, threshold=0.8,num_classes = 17):#2d和3d代码是一样的
    ce_loss = CrossEntropyLoss(ignore_index=num_classes)
    loss_ce1 = ce_loss(pred1, label[:].long())
    loss_ce2 = ce_loss(pred2, label[:].long())
    loss_ce3 = ce_loss(pred3, label[:].long())
    loss_ce = (loss_ce1 + loss_ce2 + loss_ce3)
    # print("loss_ce:{}".format(loss_ce)) # loss_ce是数

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    # print("probs_shape_len:{}".format(len(probs))) # list 3 item
    # print("probs_shape:{}".format(probs[0].shape)) # probs_shape:torch.Size([12, 8, 256, 256])
    consistency = 0.0
    #计算kl散度
    consistency += F.kl_div(probs[1].log(), probs[0], reduction='none')
    consistency += F.kl_div(probs[2].log(), probs[0], reduction='none')
    consistency += F.kl_div(probs[0].log(), probs[1], reduction='none')
    consistency += F.kl_div(probs[2].log(), probs[1], reduction='none')
    consistency += F.kl_div(probs[0].log(), probs[2], reduction='none')
    consistency += F.kl_div(probs[1].log(), probs[2], reduction='none')
    # print("consistency_shape_before:{}".format(consistency.shape)) #consistency_shape_before:torch.Size([12, 8, 256, 256])
    consistency = torch.sum(consistency, dim=1)
    # print("consistency_shape_after:{}".format(consistency.shape)) # consistency_shape_after:torch.Size([12, 256, 256])
    
    # mixture_label = (torch.stack(probs)).sum(axis=0) #按照batch_size的维度堆起来
    # # print("mixture_label_shape_before:{}".format( mixture_label.shape))  
    # # #mixture_label_shape_before:torch.Size([12, 8, 256, 256])
    # mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # b,c,d,h,w
    # # print("mixture_label_shape_after:{}".format( mixture_label.shape))
    # # # mixture_label_shape_after:torch.Size([12, 8, 256, 256])
    # max_probs = torch.amax(mixture_label.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    # # print("max_probs_shape:{}".format( max_probs.shape))
    # # # max_probs_shape:torch.Size([12, 1, 1, 1, 1])
    # mask = max_probs.ge(threshold).float() 
    # # print("mask_shape:{}".format( mask.shape))
    # # # mask_shape:torch.Size([12, 1, 1, 1, 1])
    # #torch.ge(a,b)比较a，b的大小，a为张量，b可以为和a相同形状的张量，也可以为一个常数。大于等于相应位置返回1，小于返回0.
    # logp_mixture = mixture_label.log()
    # log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
    # # print("F.kl_div(logp_mixture, prob, reduction='none') * mask:{}".format(a.shape))
    # #  F.kl_div(logp_mixture, prob, reduction='none') * mask:torch.Size([12, 12, 8, 256, 256])
    # consistency = sum(log_probs)
    # # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # # consistency_shape_after_mixture:torch.Size([12, 8, 256, 256])
    # # print(consistency)
    
    loss = (torch.mean(loss_ce) + torch.mean(consistency))/3
    return loss

def weight_with_EM(p, C=8):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1, keepdim=True) / torch.tensor(np.log(C)).cuda()
    y1 = torch.exp(-y1) #忘记了
    return y1
def calc_meanKL_3d(pred1, pred2, pred3): #单单的KL_loss罢了，和calc_jsd_multiscale_no_weight_mean_HM_3d一致
    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来并取平均，得到的依然是feature的维度

    logp_mixture = mixture_label.log()
    # print("mixture_label:{}".format(mixture_label)) #都是0~0.2的
    # print("mixture_label_shape:{}".format(mixture_label.shape))  #([1, 8, 80, 96, 96])

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])
    consistency = sum(log_probs)#进行list维度的相加，这里list有三个元素
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # consistency_shape_after_mixture:torch.Size([1, 96, 96, 96]) # batch_size x D x H x W
    
    loss = torch.mean(consistency)/3
    #这里的3指的是除以list的长度,torch.mean是按照矩阵内所有元素的平均直接得到一个数字
    return loss

def calc_meanWKL_3d(pred1, pred2, pred3): #没有数学依据但效果最好
    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来并取平均，得到的依然是feature的维度

    weight = weight_with_EM(mixture_label)
    # print("weight_shape:{}".format(weight.shape))#weight_shape:torch.Size([1, 80, 96, 96])
    # print("weight:{}".format(weight)) #都是0~1的

    weight_mixture_label = mixture_label * weight
    # print("weight_mixture_label:{}".format(weight_mixture_label.shape)) #weight_mixture_label:torch.Size([1, 8, 80, 96, 96])

    logp_mixture = weight_mixture_label.log()
    # print("mixture_label:{}".format(mixture_label)) #都是0~0.2的
    # print("mixture_label_shape:{}".format(mixture_label.shape))  #([1, 8, 80, 96, 96])

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])
    consistency = sum(log_probs)#进行list维度的相加，这里list有三个元素
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # consistency_shape_after_mixture:torch.Size([1, 96, 96, 96]) # batch_size x D x H x W
    
    loss = torch.mean(consistency)/3
    #这里的3指的是除以list的长度,torch.mean是按照矩阵内所有元素的平均直接得到一个数字
    return loss

def calc_meanWKL_3d1(pred1, pred2, pred3): #weight loss
    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来并取平均，得到的依然是feature的维度
    # print("mixture_label_shape:{}".format(mixture_label.shape)) #mixture_label_shape:torch.Size([1, 8, 80, 96, 96])

    weight = weight_with_EM(mixture_label)
    # print("weight_shape:{}".format(weight.shape))#weight_shape:torch.Size([1, 80, 96, 96])
    # print("weight:{}".format(weight)) #都是0~1的

    # weight_mixture_label = mixture_label * weight
    # print("weight_mixture_label:{}".format(weight_mixture_label.shape)) #weight_mixture_label:torch.Size([1, 8, 80, 96, 96])

    logp_mixture = mixture_label.log()
    # print("mixture_label:{}".format(mixture_label)) #都是0~0.2的
    # print("mixture_label_shape:{}".format(mixture_label.shape))  #([1, 8, 80, 96, 96])

    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none')*weight, dim=1) for prob in probs]
    # print("log_prob shape:{}".format(log_probs[0].shape)) ##log_prob shape:torch.Size([2, 8, 96, 96, 96])
    # aa= [torch.sum(F.kl_div(logp_mixture, prob, reduction='none'), dim=1) for prob in probs]
    # print("aa shape:{}".format(aa[0].shape)) # aa shape:torch.Size([2, 96, 96, 96])
    consistency = sum(log_probs)#进行list维度的相加，这里list有三个元素
    # print("consistency_shape_after_mixture:{}".format(consistency.shape)) 
    # consistency_shape_after_mixture:torch.Size([1, 96, 96, 96]) # batch_size x D x H x W
    
    loss = torch.mean(consistency)/3
    #这里的3指的是除以list的长度,torch.mean是按照矩阵内所有元素的平均直接得到一个数字
    return loss

def calc_meanKL_Mask_3d(pred1, pred2, pred3, threshold=0.5):

    probs = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    
    mixture_label = (torch.stack(probs)).mean(axis=0) #按照batch_size的维度堆起来
  
    mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # b,c,d,h,w
   
    max_probs = torch.amax(mixture_label.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)# torch.amax取出最大的数值
  
    mask = max_probs.ge(threshold).float() 
    
    logp_mixture = mixture_label.log()
    log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs] 
    consistency = sum(log_probs) 
    consistency = torch.sum(consistency, dim=1) 
    loss = torch.mean(consistency)/3
 
    return loss

def KLLoss_3d(Q,P):#Q:mix_label_afterSoftmax, P:single_label_afterSoftmax
    log_Q = Q.log()
    log_probs = torch.sum(F.kl_div(log_Q, P, reduction='none'),dim=1)
    return log_probs

def interactive_KLLoss_3d(pred1, pred2, pred3):
    pred_softmax = [F.softmax(logits, dim=1) for i, logits in enumerate([pred1, pred2, pred3])]
    mixture12 = (torch.stack([pred_softmax[0],pred_softmax[1]])).mean(axis=0)
    # print("mixture12:{}, shape:{},type:{}".format(mixture12, mixture12.shape, type(mixture12)))
    # a = (pred_softmax[0] + pred_softmax[1])/2.0
    # print("a:{}, shape:{}, type:{}".format(a, a.shape, type(a)))
    # print("\n******\n") #验证过，a和mixture12是一样的。
    mixture13 = (torch.stack([pred_softmax[0],pred_softmax[2]])).mean(axis=0)
    mixture23 = (torch.stack([pred_softmax[1],pred_softmax[2]])).mean(axis=0)

    KL12 = KLLoss_3d(mixture12, pred_softmax[2])
    KL13 = KLLoss_3d(mixture13, pred_softmax[1])
    KL23 = KLLoss_3d(mixture23, pred_softmax[0])

    # consistency = torch.mean((KL12+KL13+KL23))/3
    # print("consistenct_before:{}".format(consistency))

    consistency = torch.mean((torch.stack([KL12,KL13,KL23])).mean(axis=0))
    # print("consistenct_after:{}".format(consistency))
    """ 好像是一样的
    consistenct_before:0.2933838367462158
    consistenct_after:0.2933838665485382
    consistenct_before:0.2955893278121948
    consistenct_after:0.2955893278121948
    consistenct_before:0.29341286420822144
    consistenct_after:0.29341286420822144"""

    return consistency

   


