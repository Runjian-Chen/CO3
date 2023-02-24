voxel_size = [0.05, 0.05, 0.1]

model = dict(
    type='C2D2',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE'),
    pts_middle_encoder=dict(
        type='SparseEncoder_Without_Convout',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    pts_fusion_layer=dict(
            type='Point_Pixel_Correspondence',
            img_channels=256,
            pts_channels=64,
            mid_channels=128,
            out_channels=128,
            img_levels=[0, 1, 2, 3, 4],
            align_corners=False,
            activate_out=True,
            fuse_out=False),
    # model training and testing settings
    train_cfg=dict(
    self_supervised_strategy='baseline',
    self_supervised_paras = dict(type = 'voxel_pixel_contrastive_loss',
                                 sample_num=1024,
                                 temperature=0.7),

        assigner=[
            dict(  # for Pedestrian
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    )
