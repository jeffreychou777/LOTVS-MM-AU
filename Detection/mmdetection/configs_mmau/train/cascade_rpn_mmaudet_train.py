_base_ = "../../configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py"

model = dict(
    roi_head = dict(
        bbox_head = dict(num_classes =7 ))
)


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
        ann_file=data_root + '/labels/train.json',
        data_prefix=dict(img='images/train/')))
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=data_root + '/labels/val.json',
        data_prefix=dict(img='images/val/')))
test_dataloader = val_dataloader

default_hooks = dict(
      checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto',
        type='CheckpointHook')
)

train_cfg = dict(max_epochs=1, val_interval=1)
val_evaluator = dict(ann_file=data_root + '/labels/val.json')
test_evaluator = val_evaluator

work_dir = '../../work_dirs/mmaudet_train/cascade_rcnn_mmaudet_train'