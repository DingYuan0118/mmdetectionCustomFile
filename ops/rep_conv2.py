from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn

@CONV_LAYERS.register_module("RepConv2d")
class RepConv2d(nn.Module):
    """
    register RepConv2d module for not dcn version.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepConv2d, self).__init__()
        
    

if __name__ == "__main__":

