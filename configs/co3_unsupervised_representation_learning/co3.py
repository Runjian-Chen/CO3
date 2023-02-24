_base_ = [
    '../_base_/datasets/dair-v2x-c-dataset-pretraining.py',
    '../_base_/schedules/cyclic_10e.py', '../_base_/default_runtime.py'
]

voxel_size = [0.05, 0.05, 0.1]

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

model = dict(
    type='LIDAR_3D_ENCODER',
    pts_voxel_layer_vehicle=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    pts_voxel_layer_fusion=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=voxel_size,
        # point_cloud_range=[-70.4, -40, -3, 70.4, 40, 3],
        # voxel_size=[0.1, 0.05, 0.15],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE'),
    pts_middle_encoder=dict(
        type='SparseEncoder_Without_Convout',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
        with_conv_out=False),

    # model training and testing settings
    train_cfg=dict(
    self_supervised_paras = dict(
                        cooperative_contrastive_loss = dict(type = 'cooperative_contrastive_loss',
                                                         temperature=0.07,
                                                         sample_num=2048,
                                                         fileter_ground_points=True
                                                         ),
                        contextual_shape_prediction_loss = dict(type = 'contextual_shape_prediction_loss',
                                                         sample_num=2048,
                                                         fileter_ground_points=True
                                                         ),
                                    ),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    )

# dataset settings
dataset_type = 'DAIR_V2X_C_pretrain_dataset'
data_root = 'data/DAIR-V2X/cooperative-dataset/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
pcd_limit_range_vehicle = [0, -40, -3, 70.4, 40, 1]
pcd_limit_range_fusion = [0, -40, -3, 70.4, 40, 1]
pcd_limit_range_infrastructure = [0, -40, -3, 70.4, 40, 1]
# pcd_limit_range_fusion = [-70.4, -40, -3, 70.4, 40, 3]
# pcd_limit_range_infrastructure = [-70.4, -40, -3, 70.4, 40, 3]
# point_cloud_range = [-70.4, -40, -3, 70.4, 40, 3]
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadPointsFromFile_DAIR', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.2, 0.2, 0.2]),
    dict(type='RandomFlip3D_LIDAR_ONLY', flip_ratio_bev_horizontal=0.5, sync_2d=False ),
    dict(type='Get_Vehicle_Infrastructure_Points'),
    dict(type='PointsRangeFilter_DAIR', pcd_limit_range_vehicle=pcd_limit_range_vehicle, pcd_limit_range_fusion=pcd_limit_range_fusion),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['fusion_points', 'vehicle_points', 'fusion_points_shuffled_idx']),
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points', 'img'])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'cooperative/updated_data_info.json',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            pcd_limit_range_vehicle=pcd_limit_range_vehicle,
            pcd_limit_range_infrastructure=pcd_limit_range_infrastructure,)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'cooperative-vehicle-infrastructure/cooperative/data_info.json',
        pts_prefix='velodyne_reduced',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=False,
        pcd_limit_range_vehicle=pcd_limit_range_vehicle,
        pcd_limit_range_infrastructure=pcd_limit_range_infrastructure,),
    )
