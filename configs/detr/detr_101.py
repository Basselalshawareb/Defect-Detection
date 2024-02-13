_base_ = [ 'packages/mmdetection/configs/detr/detr_r101_8xb2-500e_coco.py',
    '../_base_/datasets/NEUDET_detection.py', '../_base_/runtime_all_hooks.py']

num_classes = 6
frozen_stages = 1
model = dict(
        backbone=dict(frozen_stages=frozen_stages),
        bbox_head=dict(num_classes=num_classes))