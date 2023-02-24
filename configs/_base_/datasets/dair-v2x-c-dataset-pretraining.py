# dataset settings
dataset_type = 'DAIR_V2X_C_pretrain_dataset'
data_root = 'data/DAIR-V2X/cooperative-dataset/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
pcd_limit_range_vehicle = [0, -40, -3, 70.4, 40, 1]
pcd_limit_range_fusion = [-70.4, -40, -3, 70.4, 40, 3]
pcd_limit_range_infrastructure = [-70.4, -40, -3, 70.4, 40, 3]
point_cloud_range = [-70.4, -40, -3, 70.4, 40, 3]
input_modality = dict(use_lidar=True, use_camera=False)

file_client_args = dict(backend='disk')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel', path_mapping=dict(data='s3://kitti_data/'))

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
        keys=['infrastructure_points', 'vehicle_points', 'vehicle_to_infrastructure_idx']),
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
    workers_per_gpu=0,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'cooperative/updated_data_info.json',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            pcd_limit_range_vehicle=pcd_limit_range_vehicle,
            pcd_limit_range_infrastructure=pcd_limit_range_infrastructure),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'cooperative/cooperative/data_info.json',
        pts_prefix='velodyne_reduced',
        pipeline=eval_pipeline,
        modality=input_modality,
        test_mode=False,
        pcd_limit_range_vehicle=pcd_limit_range_vehicle,
        pcd_limit_range_infrastructure=pcd_limit_range_infrastructure,),
    )
