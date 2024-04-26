_base_ = [
    '../../_base_/datasets/TinyPerson/TinyPerson_detection_640x512.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='CenterNet_Decouple',  #CenterNet_Decouple
    pretrained='torchvision://resnet50',  #resnext101_32x8d  resnet101
    backbone=dict(
        type='ResCenter',  #ResCenter
        depth=50,
        out_indices=(0, 1, 2, 3),
        norm_eval=False,   #
        norm_cfg=dict(type='BN'), #requires_grad=False
        style='pytorch'),
    neck=dict(
        type='CTResNetMSNeckv66_SPv5',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,  # or 64
        start_level=0,
        add_extra_convs=False,
        num_outs=4,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterNetMSHeadAF', #CenterNetMSHeadAFSoftAssignment2  CenterNetMSHeadAF  CenterNetMSHeadAFHMWH
        num_classes=1,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
    area_range=[(0,8),(8,16),(16, 32),(32, 1e5)],
                   loss_factor_heatmap = [1., 1., 1., 1.],
                   loss_factor_wh = [1., 1., 1., 1.],
                   scale_factor = [4,4,4,4]),

    test_cfg=dict(topk=600, local_maximum_kernel=3, max_per_img=600))


train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(640, 512),
        ratios= (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=(640, 512), keep_ratio=True),
    # dict(type='Resize', img_scale=(768, 614), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','gt_bboxes_ignore'])
]


test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=[1.0],
        flip=False,   #multi scale test flip test
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],  #31
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border'),
                keys=['img'])
        ])
]

dataset_type = 'CocoFmtDataset'
# data_root = 'data/tiny_set/'
data_root = '/home/disk8t/Dataset/tiny_set/'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
            type=dataset_type,
            ann_file=data_root + 'erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json',
            img_prefix=data_root + 'erase_with_uncertain_dataset/train/',
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/task/tiny_set_test_all.json',
        img_prefix=data_root + 'erase_with_uncertain_dataset/test/',
        pipeline=test_pipeline),

    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/task/tiny_set_test_all.json',
        img_prefix=data_root + 'erase_with_uncertain_dataset/test/',
        pipeline=test_pipeline)
)


# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0001)  # 2 GPU

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 1000,
    step=[100, 120])

runner = dict(max_epochs=150)

# Avoid evaluation and saving weights too frequently
# evaluation = dict(interval=1, metric='bbox')
evaluation = dict(
    interval=30, metric='bbox',
    iou_thrs= [0.25, 0.5, 0.75],  # set None mean use 0.5:1.0::0.05  [0.25, 0.5, 0.75]
    proposal_nums=[1000],
    cocofmt_kwargs=dict(
        ignore_uncertain=True,
        use_ignore_attr=True,
        use_iod_for_ignore=True,
        iod_th_of_iou_f="lambda iou: iou",  #"lambda iou: (2*iou)/(1+iou)",
        cocofmt_param=dict(
            evaluate_standard='tiny',  # or 'coco', 'tiny' 'aitod'
            # iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
            # maxDets=[200],              # set this same as set evaluation.proposal_nums
        )

    ),
    # do_final_eval=True
)
checkpoint_config = dict(interval=5)


