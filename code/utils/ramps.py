# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.

用于向上或向下倾斜超参数的函数
每个函数都采用相同格式的当前训练步长或epoch,以及渐变长度,并返回一个介于0和1之间的乘数。
"""


import numpy as np


def sigmoid_rampup(current, rampup_length):#
    """Exponential rampup from https://arxiv.org/abs/1610.02242 指数增长"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)#np.clip()将current数组里的数限定在0和t_max之间
        #rampup_length即t_max，指迭代的最大次数
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))#返回的是，exp[-5*(1-t/t_max)^2]


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length #定义假设空间，超出这个范围会警告
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))
