_base_ = ['./coco_detection.py']

dataset_type = 'NEUDETDataset'
data_root = 'data/NEU_DET/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/'),
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='test/'),
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='GenericMetric',
    dataset_name="NEU_DET",
    ann_file=data_root + 'annotations/instances_test.json',
    )
test_evaluator = val_evaluator