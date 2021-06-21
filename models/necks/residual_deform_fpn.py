import warnings
from mmcv.cnn.bricks import padding
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks import FPN
from custom.ops.deform_upsample_block import DeformUpsampleBlock


@NECKS.register_module()
class ResidualDeformFPN(FPN):
    """
    使用了Residual与Deform的集合
    """

    def __init__(self, in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(in_channels, out_channels, num_outs, start_level=start_level, end_level=end_level, add_extra_convs=add_extra_convs, extra_convs_on_inputs=extra_convs_on_inputs,
                         relu_before_extra_convs=relu_before_extra_convs, no_norm_on_lateral=no_norm_on_lateral, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, upsample_cfg=upsample_cfg, init_cfg=init_cfg)
        # 加入Upsample初始化
        self.deformupsample = nn.ModuleList()
        self.residualblocklist = nn.ModuleList()
        # 标准情况下start_level=0, backbone_end_level=4, 从低到高
        for i in range(self.start_level, self.backbone_end_level):
            upsample_module = DeformUpsampleBlock(
                out_channels, out_channels, kernel_size=3, groups=out_channels, padding=1)
            # 此处好像多了一个deformupsample模块,应该只需要三个,实际上有4个,i=0时的模块用不上
            self.deformupsample.append(upsample_module)

        # TODO:加入Residualblock初始化, 只需要三层
            if i != (self.num_ins-1):
                residual_module = ResidualBlock(in_channels=self.in_channels[i], out_channels=self.in_channels[i+1], stride=2)
                self.residualblocklist.append(residual_module)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        
        # TODO:补全residual前向传播过程
        inputs = list(inputs)
        for i in range(1, self.num_ins):
            inputs[i] = inputs[i] + self.residualblocklist[i-1](inputs[i-1])

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.

            # 修改上采样方法
            laterals[i - 1] += self.deformupsample[i](laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
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
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        return out



if __name__ == "__main__":
    import torch
    # 标准输入
    # torch.Size([2, 256, 200, 272])
    # torch.Size([2, 512, 100, 136])
    # torch.Size([2, 1024, 50, 68])
    # torch.Size([2, 2048, 25, 34])

    in_channels = [256, 512, 1024, 2048]
    myfpn = ResidualDeformFPN(in_channels=in_channels, out_channels=10, num_outs=5).cuda()
    standard_fpn = FPN(in_channels=in_channels, out_channels=10, num_outs=5).cuda()
    total_params_myfpn = sum(p.numel() for p in myfpn.parameters())
    total_params_standart_fpn = sum(p.numel() for p in standard_fpn.parameters())
    print("Myfpn parameters:{}, standard fpn parameters:{}".format(total_params_myfpn, total_params_standart_fpn))
    inputs = []
    for i in range(4):
        input_tensor = torch.randn(
            2, in_channels[i], 10*2**(3-i), 10*2**(3-i)).cuda()
        inputs.append(input_tensor)
    inputs = tuple(inputs)

    outputs = myfpn(inputs)

    for j in outputs:
        print(j.shape)
