# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import warnings
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch.nn import functional as F
import torch.nn as nn

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class LIDAR_3D_ENCODER(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer_vehicle=None,
                 pts_voxel_layer_fusion=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 train_cfg=None,
                 test_cfg=None,
                 feats_3d_encoder_out = 64,
                 final_feat = 256,
                 init_cfg=None,
                 sparse_unet=False,
                fusion_encoder_no_grad = False,
                 ):
        super(LIDAR_3D_ENCODER, self).__init__(init_cfg=init_cfg)

        self.sparse_unet = sparse_unet
        self.fusion_encoder_no_grad = fusion_encoder_no_grad

        if pts_voxel_layer_vehicle:
            self.pts_voxel_layer_vehicle = Voxelization(**pts_voxel_layer_vehicle)
        if pts_voxel_layer_fusion:
            self.pts_voxel_layer_fusion = Voxelization(**pts_voxel_layer_fusion)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._iters = 0

        self.voxel_feats_out = final_feat

        self.pts_transform_spatial_feature = nn.Sequential(
            nn.Linear(feats_3d_encoder_out, final_feat),
            nn.GELU(),
            nn.BatchNorm1d(final_feat, eps=1e-3, momentum=0.01),
            nn.Linear(final_feat, final_feat),
            nn.GELU(),
            nn.BatchNorm1d(final_feat, eps=1e-3, momentum=0.01),
        )

        if self.train_cfg is not None:
            self.loss_funcs = {}
            for key, val in self.train_cfg.self_supervised_paras.items():
                self.loss_funcs[key] = builder.build_loss(val)


    @property
    def with_img_shared_head(self):
        """bool: Whether the detector has a shared head in image branch."""
        return hasattr(self,
                       'img_shared_head') and self.img_shared_head is not None

    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self,
                       'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self,
                       'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer."""
        return hasattr(self,
                       'pts_fusion_layer') and self.fusion_layer is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_img_rpn(self):
        """bool: Whether the detector has a 2D RPN in image detector branch."""
        return hasattr(self, 'img_rpn_head') and self.img_rpn_head is not None

    @property
    def with_img_roi_head(self):
        """bool: Whether the detector has a RoI Head in image branch."""
        return hasattr(self, 'img_roi_head') and self.img_roi_head is not None

    @property
    def with_voxel_encoder(self):
        """bool: Whether the detector has a voxel encoder."""
        return hasattr(self,
                       'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        """bool: Whether the detector has a middle encoder."""
        return hasattr(self,
                       'middle_encoder') and self.middle_encoder is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            return img_feats
        else:
            return [img_feats[0]]

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        voxels, num_points, coors, voxels_coors = self.voxelize(pts)

        self.voxels_coors = voxels_coors
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        dense_voxel_feats, voxel_features, coors = self.pts_middle_encoder(voxel_features, coors, batch_size)

        if self.with_pts_backbone:
            dense_voxel_feats = self.pts_backbone(dense_voxel_feats)
        if self.with_pts_neck:
            dense_voxel_feats = self.pts_neck(dense_voxel_feats)

        N, C, D, H, W = dense_voxel_feats.shape
        dense_voxel_feats = dense_voxel_feats.permute(0,2,3,4,1).contiguous().view(-1, C)

        dense_voxel_feats = self.pts_transform(dense_voxel_feats)
        voxel_features = self.pts_transform(voxel_features)
        dense_voxel_feats = dense_voxel_feats.view(N, D, H, W ,self.voxel_feats_out)
        return dense_voxel_feats, voxel_features, coors

    def extract_vehicle_pts_feat(self, pts):
        """Extract features of points."""
        voxels, num_points, coors, voxels_coors = self.voxelize(pts, type = 'vehicle')

        self.vehicle_voxels_coors = voxels_coors
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        dense_voxel_feats, voxel_features, coors = self.pts_middle_encoder(voxel_features, coors, batch_size)

        if self.with_pts_backbone:
            dense_voxel_feats = self.pts_backbone(dense_voxel_feats)
        if self.with_pts_neck:
            dense_voxel_feats = self.pts_neck(dense_voxel_feats)

        N, C, D, H, W = dense_voxel_feats.shape

        dense_voxel_feats = dense_voxel_feats.permute(0,2,3,4,1).contiguous()#.view(-1, C)

        # dense_voxel_feats = self.pts_transform(dense_voxel_feats)
        # voxel_features = self.pts_transform(voxel_features)
        # dense_voxel_feats = dense_voxel_feats.view(N, D, H, W ,self.voxel_feats_out)
        return dense_voxel_feats, voxel_features, coors

    def extract_fusion_pts_feat(self, pts):
        """Extract features of points."""
        voxels, num_points, coors, voxels_coors = self.voxelize(pts, type = 'fusion')

        self.fusion_voxels_coors = voxels_coors
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        dense_voxel_feats, voxel_features, coors = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            dense_voxel_feats = self.pts_backbone(dense_voxel_feats)
        if self.with_pts_neck:
            dense_voxel_feats = self.pts_neck(dense_voxel_feats)

        N, C, D, H, W = dense_voxel_feats.shape
        dense_voxel_feats = dense_voxel_feats.permute(0,2,3,4,1).contiguous()#.view(-1, C)

        # dense_voxel_feats = self.pts_transform(dense_voxel_feats)
        # voxel_features = self.pts_transform(voxel_features)
        # dense_voxel_feats = dense_voxel_feats.view(N, D, H, W ,self.voxel_feats_out)
        return dense_voxel_feats, voxel_features, coors

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        voxels_feats = self.extract_pts_feat(points)
        return (img_feats, voxels_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, type='vehicle'):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points, voxels_coors  = [], [], [], []
        for res in points:
            if type == 'vehicle':
                res_voxels, res_coors, res_num_points, res_voxels_coors = self.pts_voxel_layer_vehicle(res, return_voxel_idx = True)
            elif type == 'fusion':
                res_voxels, res_coors, res_num_points, res_voxels_coors = self.pts_voxel_layer_fusion(res, return_voxel_idx = True)
            else:
                raise ValueError

            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxels_coors.append(res_voxels_coors)

        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch, voxels_coors

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      vehicle_points=None,
                      vehicle_to_infrastructure_idx=None,
                      fusion_points=None,
                      fusion_points_shuffled_idx=None,
                      infrastructure_points=None,
                      fusion_points_infrastructure_mask=None,
                      vehicle_shape_context_feature=None,
                      noise_rotation = None,
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.


        Returns:
            dict: Losses of different branches.
        """
        if vehicle_points is not None:
            # vehicle_coors [N, 3]: from points to voxel position
            vehicle_spatial_features, vehicle_seg_features, vehicle_seg_indices = self.extract_vehicle_pts_feat(vehicle_points)

        if fusion_points is not None:
            if self.fusion_encoder_no_grad:
                with torch.no_grad():
                    fusion_spatial_features, fusion_seg_features, fusion_seg_indices = self.extract_fusion_pts_feat(fusion_points)
            else:
                fusion_spatial_features, fusion_seg_features, fusion_seg_indices = self.extract_fusion_pts_feat(
                    fusion_points)

        losses = dict()

        for loss_name , loss_func in self.loss_funcs.items():

            if loss_name == 'cooperative_contrastive_loss':

                N, D, H, W, C = vehicle_spatial_features.shape
                transformed_vehicle_spatial_features = vehicle_spatial_features.view(-1, C)
                transformed_vehicle_spatial_features = self.pts_transform_spatial_feature(transformed_vehicle_spatial_features)
                transformed_vehicle_spatial_features = transformed_vehicle_spatial_features.view(N, D, H, W ,self.voxel_feats_out)

                N, D, H, W, C = fusion_spatial_features.shape
                transformed_fusion_spatial_features = fusion_spatial_features.view(-1, C)
                transformed_fusion_spatial_features = self.pts_transform_spatial_feature(
                    transformed_fusion_spatial_features)
                transformed_fusion_spatial_features = transformed_fusion_spatial_features.view(N, D, H, W,
                                                                                self.voxel_feats_out)

                loss = loss_func(dense_vehicle_voxel_feats=transformed_vehicle_spatial_features, vehicle_voxels_coors=self.vehicle_voxels_coors, dense_fusion_voxel_feats=transformed_fusion_spatial_features, fusion_voxels_coors=self.fusion_voxels_coors, fusion_points_shuffled_idx=fusion_points_shuffled_idx, vehicle_points=vehicle_points)
                losses[loss_name] = loss
                for key, val in loss_func.log_var.items():
                    losses[loss_name + '---' + key] = val

            if loss_name == 'contextual_shape_prediction_loss':

                loss = loss_func(vehicle_points=vehicle_points, vehicle_voxels_coors=self.vehicle_voxels_coors,
                                 vehicle_spatial_features=vehicle_spatial_features, vehicle_spatial_feature_voxels_coors=self.vehicle_voxels_coors, noise_rotation=noise_rotation,
                                 fusion_spatial_features=fusion_spatial_features, fusion_spatial_feature_voxels_coors = self.fusion_voxels_coors,
                                 fusion_points = fusion_points)
                losses[loss_name] = loss
                for key, val in loss_func.log_var.items():
                    losses[loss_name + '---' + key] = val

        self._iters += 1

        return losses

    def show_results(self, data, result, out_dir):

        with open('readme.txt', 'w') as f:
            for key, val in result.items():
                result_str = key + ': ' + str(val) + '\n'
                f.write(result_str)


    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, voxels_feats = self.extract_feat(
            [points], img=img, img_metas=[img_metas])

        img_feats_on_pts = self.fusion_layer(img_feats, points, img_metas)

        losses = dict()

        for loss_func in self.loss_funcs:
            return_dict = loss_func(voxels_feats, img_feats_on_pts, self.voxels_coors)
            for key, val in return_dict.items():
                losses[key] = val

        return losses

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    @torch.no_grad()
    @force_fp32()
    def extract_3d_representation(self, points, img = None, img_metas = None):
        # points: list of tensor
        # img: list of tensor
        # img_metas: list of dict
        # Return [N_p, C] per point representation
        img_feats = self.extract_img_feat(img, img_metas)
        img_feats_on_pts = self.fusion_layer(img_feats, points, img_metas)

        return img_feats_on_pts[0]