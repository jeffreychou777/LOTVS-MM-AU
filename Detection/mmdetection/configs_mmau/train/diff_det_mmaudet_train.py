_base_ = '/home/work/mmdet/mmdetection-main/projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py'

model = dict(bbox_head = dict(num_classes =7,
            single_head = dict(num_classes =7),
            criterion = dict(num_classes =7)
            ))
            


data_root = '../../data/MM-AU-Detect'
metainfo = {
    'classes': ('motorcycle','truck','bus','traffic light','person','bicycle','car', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'labels/train.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + 'labels/val.json',
        data_prefix=dict(img='images/val/')))
test_dataloader = val_dataloader

default_hooks = dict(
      checkpoint=dict(
        interval=1, max_keep_ckpts=1, save_best='auto',
        type='CheckpointHook')
)


val_evaluator = dict(ann_file=data_root + '/labels/val.json')
test_evaluator = val_evaluator
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=10000)

work_dir = '../../work_dirs/mmaudet_train/diff_det_mmaudet_train'