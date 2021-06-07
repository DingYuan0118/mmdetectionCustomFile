custom_imports = dict(
    imports=['custom.ops.rep_deform_conv', 'custom.models.backbones.rep_resnet'],
    allow_failed_imports=False)
_base_ = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        type='RepResnet',
        dcn=dict(type='RepDCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
        # TODO: ADD conv_cfg parameter to add repconv eg: conv_cfg=dict(type="repconv")