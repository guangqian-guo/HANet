dataset_type = 'CocoFmtDataset'
data_root = 'data/tinyset/'


# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         # type='MultiScaleFlipAug',
#         # img_scale=(1333, 800),
#         type='CroppedTilesFlipAug',
#         tile_shape=(640, 512),  # sub image size by cropped
#         tile_overlap=(100, 100),
#         scale_factor=[1.0],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale_factor=[1.0], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        # type='CroppedTilesFlipAug',
        img_scale=(640, 512),  # sub image size by cropped
        # tile_overlap=(100, 100),
        # scale_factor=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='AutoAugment',
#         policies=[[
#             dict(
#                 type='Resize',
#                 img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                            (736, 1333), (768, 1333), (800, 1333)],
#                 multiscale_mode='value',
#                 keep_ratio=True)
#         ],
#                   [
#                       dict(
#                           type='Resize',
#                           img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                           multiscale_mode='value',
#                           keep_ratio=True),
#                       dict(
#                           type='RandomCrop',
#                           crop_type='absolute_range',
#                           crop_size=(384, 600),
#                           allow_negative_crop=True),
#                       dict(
#                           type='Resize',
#                           img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                      (576, 1333), (608, 1333), (640, 1333),
#                                      (672, 1333), (704, 1333), (736, 1333),
#                                      (768, 1333), (800, 1333)],
#                           multiscale_mode='value',
#                           override=True,
#                           keep_ratio=True)
#                   ]]),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=1),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=1),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
            type=dataset_type,
            # ann_file=data_root + 'erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json',
            ann_file=data_root + 'erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json',  # same as last line
            img_prefix=data_root + 'erase_with_uncertain_dataset/train/',
            pipeline=train_pipeline,
            # train_ignore_as_bg=False,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        # ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        merge_after_infer_kwargs=dict(
            merge_gt_file=data_root + 'annotations/task/tiny_set_test_all.json',
            merge_nms_th=0.5
        ),
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/corner/task/tiny_set_test_sw640_sh512_all.json',
        ann_file=data_root + 'mini_annotations/tiny_set_test_all.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline)
)


evaluation = dict(interval=10,
                  metric='bbox',
                  cocofmt_kwargs=dict(
                      ignore_uncertain=True,
                      use_ignore_attr=True,
                      use_iod_for_ignore=True,
                      iod_th_of_iou_f="lambda iou: iou",  # "lambda iou: (2*iou)/(1+iou)",
                      cocofmt_param=dict(
                          evaluate_standard='tiny',  # or 'coco'
                          # iouThrs=[0.25, 0.5, 0.75],  # set this same as set evaluation.iou_thrs
                          # maxDets=[200],              # set this same as set evaluation.proposal_nums
                      )
                  )
                  )
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=150)