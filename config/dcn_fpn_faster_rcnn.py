custom_imports = dict(
    imports=['custom.ops.deform_upsample_block', 'custom.models.necks.Myfpn'],
    allow_failed_imports=False)
_base_ = '../../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    neck=dict(
        type='MyFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
        )