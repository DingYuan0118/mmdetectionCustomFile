import warnings
from mmcv.cnn.bricks import padding
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks import FPN
from custom.ops.deform_upsample_block_average_deformgroup import DeformUpsampleBlock

@NECKS.register_module()
class DeformUpFPN(FPN):
    """

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
        # 加入Upsample定义
        self.deformupsample = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level-1): # 不同于laterals,upsampling仅需要3层
            upsample_module = DeformUpsampleBlock(out_channels,out_channels,kernel_size=3,groups=out_channels, padding=1, deform_groups=4) # deform_group 指定offset的维度
            self.deformupsample.append(upsample_module)
    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

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
            laterals[i - 1] += self.deformupsample[i-1](laterals[i])


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

if __name__ == "__main__":
    import torch
    in_channels = [10,20,40,80]
    fpn = DeformUpFPN(in_channels=in_channels, out_channels=10, num_outs=5).cuda()
    inputs = []
    for i in range(4):
        input_tensor = torch.randn(3,in_channels[i],10*2**(3-i), 10*2**(3-i)).cuda()
        inputs.append(input_tensor)

    outputs = fpn(inputs)
    
    for j in range(outputs):
        print(j.shape)