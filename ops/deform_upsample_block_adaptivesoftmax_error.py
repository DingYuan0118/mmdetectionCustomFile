from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math
import torch.nn.functional as F

# @CONV_LAYERS.register_module("UPDCN")
class DeformUpsampleBlock(DeformConv2dPack):
    """
    Add Rep structure to DCN module, Block 意味着将batchnorm层融入该模块 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.out_channels == self.in_channels, "in repDCN, out_channel must match in_channels"
        # self.groups = self.in_channels   #分组卷积
        self.scale = 2 # 上采样尺度
        del self.weight
        # only weight, no bias

        # self.weight = nn.Parameter(
        #     torch.Tensor(self.out_channels*self.scale**2, self.in_channels // self.groups,
        #                  *self.kernel_size))
        # self.reset_parameters()


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_weight = nn.Sequential(
            nn.Linear(
                self.out_channels,
                self.out_channels//16,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.out_channels//16,
                self.scale**2*self.kernel_size[0]*self.kernel_size[1],
                bias=False
            ),
        )


        self.PS = torch.nn.PixelShuffle(self.scale)
        
        
    
    def forward(self, x):
        # 弃用
        # 已执行，但是存在问题，self.weight部分reshape方法存在问题，weight_tmp_[:,0].flatten() != self.weight[0].flatten()，说明变形有问题
        # reshape在变形时需谨慎使用
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别

        [B, C, H, W] = x.size()
        # assert B==1, "当前只支持batchsize为1的上采样"
        self.weight = self.conv_weight(self.avgpool(x).view(B, C))
        weight_tmp_ = self.weight.reshape(self.scale**2, -1, self.kernel_size[0]*self.kernel_size[1])
        weight_tmp = F.softmax(weight_tmp_, dim=-1)
        weight_tmp = weight_tmp.reshape(self.scale**2, -1,self.kernel_size[0], self.kernel_size[1])
        weight_repeat = weight_tmp.repeat(self.out_channels, 1, 1, 1)
        output = []
        for i in range(B):
            output.append(deform_conv2d(x[i].unsqueeze(0), offset[i].unsqueeze(0), weight_repeat[:,i,:,:].unsqueeze(1), self.stride, self.padding,
                                self.dilation, self.groups, self.deform_groups))
        output = torch.cat(output, dim=0)
        output = self.PS(output)
        return output

if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,4,4).cuda()
    out3x3 = model(input)
    print(out3x3.shape)
