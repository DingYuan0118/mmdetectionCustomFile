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
        # self.temperature = 10
        del self.weight
        # only weight, no bias

        # self.weight = nn.Parameter(
        #     torch.Tensor(self.out_channels*self.scale**2, self.in_channels // self.groups,
        #                  *self.kernel_size))
        # self.reset_parameters()


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_weight = nn.Sequential(
            nn.Conv2d(
                self.out_channels, # focus 暂时只支持x2上采样
                self.out_channels//4, # 07 06 添加
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=4,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.out_channels//4,
                self.kernel_size[0]*self.kernel_size[1],
                kernel_size=1,
                stride=1,
                padding=0,
                # groups=4,
                bias=False,
            ),
        )


        self.PS = torch.nn.PixelShuffle(self.scale)
        
    def forward(self, x):
        # 正确版本
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别

        [B, C, H, W] = x.size()
        x_pad = x
        if H % 2 != 0:
            x_pad = F.pad(x_pad, (0,0,0,1), "constant", 0)
        if W % 2 != 0:
            x_pad = F.pad(x_pad, (0,1), "constant", 0)

        x_foucs = torch.cat([x_pad[..., ::2, ::2], x_pad[..., 1::2, ::2], x_pad[..., ::2, 1::2], x_pad[..., 1::2, 1::2]], 1)
        x_focus_GAP = self.avgpool(x_foucs).reshape(-1, self.out_channels, 1, 1)
        self.weight = self.conv_weight(x_focus_GAP)
        weight_tmp_ = self.weight.reshape(-1, self.scale**2, 1,self.kernel_size[0]*self.kernel_size[1])
        weight_tmp = F.softmax(weight_tmp_, dim=-1)
        weight_tmp = weight_tmp.reshape(-1, self.scale**2, 1, self.kernel_size[0], self.kernel_size[1])
        weight_repeat = weight_tmp.repeat(1, self.out_channels, 1, 1, 1)
        weight_last = weight_repeat.reshape(-1, 1, self.kernel_size[0], self.kernel_size[1])
        x = x.reshape(-1, B*C, H, W)
        offset = offset.reshape(-1, B*self.kernel_size[0] * self.kernel_size[1] * 2, H, W)
        output = deform_conv2d(x, offset, weight_last, self.stride, self.padding,
                            self.dilation, self.groups*B, self.deform_groups*B)
        output = output.reshape(B, -1 ,H, W)
        output = self.PS(output)
        return output



if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,5,5).cuda()
    out1 = model.forward(input)
    print(out1.shape)
    out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())