import warnings
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

class RepBottleneck(Bottleneck):
    """
    Add reparameterized branches (1x1, identity) to normal bottlenek
    """

    def __init__(self, inplanes, planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):

        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        if dcn is not None:
            assert dcn["type"] == "RepDCN", "RepBottleNeck only use RepDCN for deform conv"
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        # self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm2_1name, norm2_1 = build_norm_layer(
            norm_cfg, planes, postfix="2_1")
        self.norm2_2name, norm2_2 = build_norm_layer(
            norm_cfg, planes, postfix="2_2")
        self.norm2_3name, norm2_3 = build_norm_layer(
            norm_cfg, planes, postfix="2_3")
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            # RepBottleneck use RepConv2d as default
            self.conv2 = build_conv_layer(
                dict(type="Conv2d"),
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_1name, norm2_1)
        self.add_module(self.norm2_2name, norm2_2)
        self.add_module(self.norm2_3name, norm2_3)

        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    @property
    def norm2_1(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm2_1name)

    @property
    def norm2_2(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm2_2name)

    @property
    def norm2_3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm2_3name)

    def forward(self, x):
        
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out3x3, out1x1, ident = self.conv2(out)
            out3x3 = self.norm2_1(out3x3)
            out1x1 = self.norm2_2(out1x1)
            ident = self.norm2_3(ident)

            out = out3x3 + out1x1 + ident
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out



@BACKBONES.register_module()
class RepResnet(ResNet):
    """
    调用RepBottleneck的RepResnet，实现重参数化
    注意：Deform Conv原文中只是用在C3-C5层，因此，RepBottleneck也只使用在C3-C5层，如需全部替换
    则需要更改make_layer分支条件
    """
    arch_settings = {
        # TODO: 将BasicBlock改为RepBasicBlock
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (RepBottleneck, (3, 4, 6, 3)),
        101: (RepBottleneck, (3, 4, 23, 3)),
        152: (RepBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
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
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        assert pretrained is None, "RepResnet没有预训练模型"
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    # 添加RepBottle
                    elif block is Bottleneck or block is RepBottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            if not dcn:
                # 第一层layer不使用RepBottleneck, 即不使用dcn的层也不使用RepBottleneck
                res_layer = self.make_res_layer(
                    block=Bottleneck,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=stage_plugins,
                init_cfg=block_init_cfg)

            else:
                res_layer = self.make_res_layer(
                    block=self.block,
                    inplanes=self.inplanes,
                    planes=planes,
                    num_blocks=num_blocks,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    dcn=dcn,
                    plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)
        


if __name__ == "__main__":
    model = RepResnet(50, 3,
                      dcn=dict(type='RepDCN', deform_groups=1, fallback_on_stride=False),
                      stage_with_dcn=(False, True, True, True)).cuda()
    model2 = ResNet(50, 3,
                    dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
                    stage_with_dcn=(False, True, True, True)).cuda()
    input = torch.Tensor(1,3,224,224).cuda()
    input2 = torch.Tensor(1,3,224,224).cuda()

    out = model(input)
    out2 = model2(input2)
    print(out.shape)
    print(out2.shape)
