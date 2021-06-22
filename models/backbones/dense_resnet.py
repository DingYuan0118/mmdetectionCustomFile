import warnings
from mmcv.fileio.handlers import base
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from mmcv.cnn import CONV_LAYERS
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models.backbones.resnet import ResNet, Bottleneck, BasicBlock
from custom.ops.rep_deform_conv import RepDeformConv2dPackBlock


@BACKBONES.register_module()
class DenseResnet(ResNet):
    """
    """

    def __init__(self, depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(depth, in_channels=in_channels, stem_channels=stem_channels, base_channels=base_channels, num_stages=num_stages, strides=strides, dilations=dilations, out_indices=out_indices, style=style, deep_stem=deep_stem, avg_down=avg_down,
                         frozen_stages=frozen_stages, conv_cfg=conv_cfg, norm_cfg=norm_cfg, norm_eval=norm_eval, dcn=dcn, stage_with_dcn=stage_with_dcn, plugins=plugins, with_cp=with_cp, zero_init_residual=zero_init_residual, pretrained=pretrained, init_cfg=init_cfg)
        self.nums_out = len(out_indices)
        out_channelslist = []
        self.residualblocklist = nn.ModuleList()
        for i in out_indices:
            out_channelslist.append(base_channels * 2**i * self.block.expansion)  # bottleneck expansion默认为4
        for i in range(self.nums_out-1):
            residual_module = ResidualBlock(in_channels=out_channelslist[i], out_channels=out_channelslist[i+1], stride=2)
            self.residualblocklist.append(residual_module)

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i >= 1:
                x = x + self.relu(self.residualblocklist[i-1](outs[i-1]))
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

class ResidualBlock(nn.Module):
    """
        增加4层输出之间的Residual模块,供快速反向传递梯度使用
        Args:
            in_channels(int): 指定输入通道数
            out_channels(int): 指定输出通道数
            base_channels(int): 指定内部降维后的通道数
        其余参数见nn.Conv2d
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=256,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 inplace=True,
                 padding_mode='zeros'):
        super().__init__()
        # bias 默认为false, 参照resnet
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=(1,1), stride=1,
                               padding=padding, dilation=dilation, bias=bias, groups=groups, padding_mode=padding_mode)
        self.norm1 = nn.BatchNorm2d(base_channels)
        # padding置1, stride置为2, 下采样
        self.conv2 = nn.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=(3,3), stride=stride,
                               padding=1, dilation=dilation, bias=bias, groups=groups, padding_mode=padding_mode)
        self.norm2 = nn.BatchNorm2d(base_channels)
        self.conv3 = nn.Conv2d(in_channels=base_channels, out_channels=out_channels, kernel_size=(1,1), stride=1,
                               padding=padding, dilation=dilation, bias=bias, groups=groups, padding_mode=padding_mode)
        # self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)
        self.init_weight()

    def init_weight(self):
        """
        conv层初始化为零, norm层默认初始化为weight=1, bias=0,因此不用显式初始化
        """
        # self.conv1.weight.data.zero_()
        # self.conv2.weight.data.zero_()
        self.conv3.weight.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out = self.norm3(out)
        return out

grads = {}
 
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

if __name__ == "__main__":
    block = ResidualBlock(3, 512)
    inputs = torch.randn(1,3,224,224)
    outputs = block(inputs)
    outputs = outputs.sum()
    outputs.backward()
    print()

