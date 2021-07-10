from mmcv.cnn.bricks import padding
from numpy.core.numeric import identity
import torch
from mmcv.ops.deform_conv import DeformConv2dPack, deform_conv2d
from mmcv.cnn import CONV_LAYERS
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.init import calculate_gain

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

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_weight = nn.Sequential(
            nn.Linear(
                self.out_channels,
                self.out_channels//4,
                bias=False
            ),
            nn.ReLU(inplace=True), 
            nn.Linear(
                self.out_channels//4,
                self.scale**2*self.kernel_size[0]*self.kernel_size[1],
                bias=False
            ),
        )
        self.FN = FilterNorm(self.scale**2, self.kernel_size[0], running_std=True)
        self.PS = torch.nn.PixelShuffle(self.scale)
        
        
    
    # def forward(self, x):
    #     # 弃用
    #     # 错误版本，等效于forward2，未执行
    #     offset = self.conv_offset(x) # [batch, 18, rows, cols]
    #     # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别
     
    #     [B, C, H, W] = x.size()
    #     # assert B==1, "当前只支持batchsize为1的上采样"
    #     self.weight = self.conv_weight(self.avgpool(x).view(B, C))
    #     weight_tmp_ = self.weight.reshape(self.scale**2, -1, self.kernel_size[0]*self.kernel_size[1])
    #     weight_tmp = F.softmax(weight_tmp_, dim=-1)
    #     weight_tmp = weight_tmp.reshape(self.scale**2, -1, self.kernel_size[0], self.kernel_size[1])
    #     weight_repeat = weight_tmp.repeat(self.out_channels, 1, 1, 1)
    #     x = x.reshape(-1, B*C, H, W)
    #     offset = offset.reshape(-1, B*self.kernel_size[0] * self.kernel_size[1] * 2, H, W)
    #     # 注意reshape操作与想象中可能不同，此处不方便直接使用reshape
    #     weight_last = weight_repeat[:,0].unsqueeze(1)
    #     for i in range(1, B):
    #         weight_last = torch.cat((weight_last, weight_repeat[:,i].unsqueeze(1)))

    #     output = deform_conv2d(x, offset, weight_last, self.stride, self.padding,
    #                         self.dilation, self.groups*B, self.deform_groups*B)
    #     output = output.reshape(B, -1 ,H, W)
    #     output = self.PS(output)
    #     return output

    # def forward2(self, x):
    #     # 弃用
    #     # 已执行，但是存在问题，self.weight部分reshape方法存在问题，weight_tmp_[:,0].flatten() != self.weight[0].flaten*()，说明变形有问题
    #     # reshape在变形时需谨慎使用
    #     offset = self.conv_offset(x) # [batch, 18, rows, cols]
    #     # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别

    #     [B, C, H, W] = x.size()
    #     # assert B==1, "当前只支持batchsize为1的上采样"
    #     self.weight = self.conv_weight(self.avgpool(x).view(B, C))
    #     weight_tmp_ = self.weight.reshape(self.scale**2, -1, self.kernel_size[0]*self.kernel_size[1])
    #     weight_tmp = F.softmax(weight_tmp_, dim=-1)
    #     weight_tmp = weight_tmp.reshape(self.scale**2, -1,self.kernel_size[0], self.kernel_size[1])
    #     weight_repeat = weight_tmp.repeat(self.out_channels, 1, 1, 1)
    #     output = []
    #     for i in range(B):
    #         output.append(deform_conv2d(x[i].unsqueeze(0), offset[i].unsqueeze(0), weight_repeat[:,i,:,:].unsqueeze(1), self.stride, self.padding,
    #                             self.dilation, self.groups, self.deform_groups))
    #     output = torch.cat(output, dim=0)
    #     output = self.PS(output)
    #     return output

    def forward(self, x):
        # 正确版本
        offset = self.conv_offset(x) # [batch, 18, rows, cols]
        # 对weight使用softmax时会导致梯度为None，需要小心，注意self.weight于self.weight.data的差别

        [B, C, H, W] = x.size()
        # assert B==1, "当前只支持batchsize为1的上采样"
        weight = self.conv_weight(self.avgpool(x).view(B, C))
        # weight_tmp_ = self.weight.reshape(-1, self.scale**2, 1,self.kernel_size[0]*self.kernel_size[1])
        # weight_tmp = F.softmax(weight_tmp_, dim=-1)
        # weight_tmp = weight_tmp.reshape(-1, self.scale**2, 1, self.kernel_size[0], self.kernel_size[1])
        weight_fn = self.FN(weight)
        weight_tmp = weight_fn.reshape(-1, self.scale**2, 1,self.kernel_size[0], self.kernel_size[1])
        weight_repeat = weight_tmp.repeat(1, self.out_channels, 1, 1, 1)
        weight_last = weight_repeat.reshape(-1, 1, self.kernel_size[0], self.kernel_size[1])
        x = x.reshape(-1, B*C, H, W)
        offset = offset.reshape(-1, B*self.kernel_size[0] * self.kernel_size[1] * 2, H, W)
        output = deform_conv2d(x, offset, weight_last, self.stride, self.padding,
                            self.dilation, self.groups*B, self.deform_groups*B)
        output = output.reshape(B, -1 ,H, W)
        output = self.PS(output)
        return output

class FilterNorm(nn.Module):
    """
        Filter Norm layer for predicted weight
    """
    def __init__(self, in_channels, kernel_size,
                 nonlinearity='linear', running_std=False, running_mean=False):
        super().__init__()
        self.in_channels = in_channels
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        b = x.size(0)
        c = self.in_channels
        x = x.view(b, c, -1)
        x = x - x.mean(dim=2).view(b, c, 1)
        x = x / (x.std(dim=2).view(b, c, 1) + 1e-10)
        x = x.view(b, -1)
        if self.runing_std:
            x = x * self.std[None, :]
        else:
            x = x * self.std
        if self.runing_mean:
            x = x + self.mean[None, :]
        return x


if __name__ == "__main__":
    model = DeformUpsampleBlock(256,256,(3,3), padding=1, groups=256).cuda()
    input = torch.randn(2,256,4,4).cuda()
    out1 = model.forward(input)
    print(out1.shape)
    out2 = model.forward2(input)
    # out3 = model.forward3(input)
    # print((out2 == out1).all())
    # print((out2 == out3).all())
