_base_ = [
    '../../configs/_base_/models/faster_rcnn_r50_fpn.py', '../../configs/_base_/datasets/voc0712.py',
    '../../configs/_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)),)
# optimizer
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=5)  # actual epoch = 4 * 3 = 12