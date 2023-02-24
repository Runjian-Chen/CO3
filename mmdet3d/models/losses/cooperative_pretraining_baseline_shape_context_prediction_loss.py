import torch
import torch.nn as nn

from ..builder import LOSSES
from ..utils.transformer import TransformerEncoder
import math

def limit_coor_range(voxels_coor, spatial_feature):
    D, H, W, C = spatial_feature.shape
    voxels_coor[voxels_coor[:,0] >= D, 0] = D-1
    voxels_coor[voxels_coor[:,1] >= H, 0] = H-1
    voxels_coor[voxels_coor[:,2] >= W, 0] = W-1
    return voxels_coor

class ShapeContext(object):
    def __init__(self, r1=0.125, r2=2, nbins_xy=2, nbins_zy=2):
        # right-hand rule
        """
        nbins_xy >= 2
        nbins_zy >= 1
        """
        self.r1 = r1
        self.r2 = r2
        self.nbins_xy = nbins_xy
        self.nbins_zy = nbins_zy
        self.partitions = nbins_xy * nbins_zy * 2

    @staticmethod
    def pdist(rel_trans):
        D2 = torch.sum(rel_trans.pow(2), 2)
        return torch.sqrt(D2 + 1e-7)

    @staticmethod
    def pdist_batch(rel_trans):
        D2 = torch.sum(rel_trans.pow(2), 3)
        return torch.sqrt(D2 + 1e-7)

    @staticmethod
    def compute_rel_trans(A, B):
        return A.unsqueeze(0) - B.unsqueeze(1)

    @staticmethod
    def compute_rel_trans_batch(A, B):
        return A.unsqueeze(1) - B.unsqueeze(2)

    @staticmethod
    def hash(A, B, seed):
        '''
        seed = bins of B
        entry < 0 will be ignored
        '''
        mask = (A >= 0) & (B >= 0)
        C = torch.zeros_like(A) - 1
        C[mask] = A[mask] * seed + B[mask]
        return C

    @staticmethod
    def hash_batch(A, B, seed):
        '''
        seed = bins of B
        entry < 0 will be ignored
        '''

        mask = (A >= 0) & (B >= 0)
        C = torch.zeros_like(A) - 1
        C[mask] = A[mask] * seed + B[mask]

        return C

    @staticmethod
    def compute_angles(rel_trans):
        """ compute angles between a set of points """
        angles_xy = torch.atan2(rel_trans[:, :, 1], rel_trans[:, :, 0])
        # angles between 0, 2*pi
        angles_xy = torch.fmod(angles_xy + 2 * math.pi, 2 * math.pi)

        angles_zy = torch.atan2(rel_trans[:, :, 1], rel_trans[:, :, 2])
        # angles between 0, pi
        angles_zy = torch.fmod(angles_zy + 2 * math.pi, math.pi)

        return angles_xy, angles_zy

    @staticmethod
    def compute_angles_batch(rel_trans):
        """ compute angles between a set of points """
        angles_xy = torch.atan2(rel_trans[:, :, :, 1], rel_trans[:, :, :, 0])
        # angles between 0, 2*pi
        angles_xy = torch.fmod(angles_xy + 2 * math.pi, 2 * math.pi)

        angles_zy = torch.atan2(rel_trans[:, :, :, 1], rel_trans[:, :, :, 2])
        # angles between 0, pi
        angles_zy = torch.fmod(angles_zy + 2 * math.pi, math.pi)

        return angles_xy, angles_zy

    def compute_partitions(self, xyz):
        rel_trans = ShapeContext.compute_rel_trans(xyz, xyz)

        # angles
        angles_xy, angles_zy = ShapeContext.compute_angles(rel_trans)
        angles_xy_bins = torch.floor(angles_xy / (2 * math.pi / self.nbins_xy))
        angles_zy_bins = torch.floor(angles_zy / (math.pi / self.nbins_zy))
        angles_bins = ShapeContext.hash(angles_xy_bins, angles_zy_bins, self.nbins_zy)

        # distances
        distance_matrix = ShapeContext.pdist(rel_trans)
        dist_bins = torch.zeros_like(angles_bins) - 1

        # partitions
        mask = (distance_matrix >= self.r1) & (distance_matrix < self.r2)
        dist_bins[mask] = 0
        mask = distance_matrix >= self.r2
        dist_bins[mask] = 1

        bins = ShapeContext.hash(dist_bins, angles_bins, self.nbins_xy * self.nbins_zy)
        return bins

    def compute_partitions_batch(self, xyz_batch):
        rel_trans_batch = ShapeContext.compute_rel_trans_batch(xyz_batch, xyz_batch)

        # angles
        angles_xy_batch, angles_zy_batch = ShapeContext.compute_angles_batch(rel_trans_batch)
        angles_xy_bins_batch = torch.floor(angles_xy_batch / (2 * math.pi / self.nbins_xy))
        angles_zy_bins_batch = torch.floor(angles_zy_batch / (math.pi / self.nbins_zy))
        angles_bins_batch = ShapeContext.hash_batch(angles_xy_bins_batch, angles_zy_bins_batch, self.nbins_zy)

        # distances
        distance_matrix_batch = ShapeContext.pdist_batch(rel_trans_batch)
        dist_bins_batch = torch.zeros_like(angles_bins_batch) - 1

        # partitions
        mask_batch = (distance_matrix_batch >= self.r1) & (distance_matrix_batch < self.r2)
        dist_bins_batch[mask_batch] = 0
        mask_batch = distance_matrix_batch >= self.r2
        dist_bins_batch[mask_batch] = 1
        bins_batch = ShapeContext.hash_batch(dist_bins_batch, angles_bins_batch, self.nbins_xy * self.nbins_zy)

        return bins_batch

    def compute_partitions_fast(self, xyz):
        '''
        fast partitions:  axis-aligned partitions
        '''

        partition_matrix = torch.zeros((xyz.shape[0], xyz.shape[0]))
        partition_matrix = partition_matrix.cuda() - 1e9

        rel_trans = ShapeContext.compute_rel_trans(xyz, xyz)
        maskUp = rel_trans[:, :, 2] > 0.0
        maskDown = rel_trans[:, :, 2] < 0.0

        distance_matrix = ShapeContext.pdist(rel_trans)

        mask = (distance_matrix[:, :] > self.r1) & (distance_matrix[:, :] <= self.r2)
        partition_matrix[mask & maskUp] = 0
        partition_matrix[mask & maskDown] = 1

        mask = distance_matrix[:, :] > self.r2
        partition_matrix[mask & maskUp] = 2
        partition_matrix[mask & maskDown] = 3
        self.partitions = 4

        return partition_matrix

@LOSSES.register_module()
class cooperative_pretraining_baseline_shape_context_prediction_loss(nn.Module):

    def __init__(self, temperature = 0.7, sample_num=2048, fileter_ground_points = False, in_channels = 16, out_channels = 32, topk_nearest=200):
        super(cooperative_pretraining_baseline_shape_context_prediction_loss, self).__init__()

        self.temperature = temperature
        self.sample_num = sample_num
        self.fileter_ground_points = fileter_ground_points
        self.log_var = {}
        self.loss_func_name = 'cooperative_pretraining_baseline_shape_context_prediction_loss'
        self.topk_nearest = topk_nearest
        self.local_feature_extractor = ShapeContext(r1=1.0, r2=4.0, nbins_xy= 4, nbins_zy= 4)
        self.projector_final = nn.Sequential(
                nn.Linear(in_channels * 4, in_channels * 4),
                nn.ELU(inplace=True),
                nn.Linear(in_channels * 4, out_channels),
                nn.ELU(inplace=True),
            ).to(torch.cuda.current_device())

        self.projector_low_res = nn.Sequential(
            nn.Linear(64, in_channels * 4),
            nn.ELU(inplace=True),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.ELU(inplace=True),
        ).to(torch.cuda.current_device())

        self.projector_low_res_fusion = nn.Sequential(
            nn.Linear(64, in_channels * 4),
            nn.ELU(inplace=True),
            nn.Linear(in_channels * 4, in_channels * 4),
            nn.ELU(inplace=True),
        ).to(torch.cuda.current_device())

        self.kld = torch.nn.KLDivLoss(reduction='batchmean')
        self.l1_loss = torch.nn.L1Loss(reduction='sum')
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
    def forward(self,
                **kwargs):
        # dense_vehicle_voxel_feats: list of N_i points
        # vehicle_voxels_coors: list of [N_i, C_img]
        # dense_fusion_voxel_feats: [B, D, H, W, C]
        # fusion_voxels_coors: list of [N_i, 3]
        # fusion_points_shuffled_idx:
        pos_sim = []
        neg_sim = []

        vehicle_points = kwargs['vehicle_points']
        fusion_points = kwargs['fusion_points']
        vehicle_voxels_coors = kwargs['vehicle_voxels_coors']
        vehicle_spatial_features = kwargs['vehicle_spatial_features']
        vehicle_spatial_feature_voxels_coors = kwargs['vehicle_spatial_feature_voxels_coors']
        fusion_spatial_features = kwargs['fusion_spatial_features']
        fusion_spatial_feature_voxels_coors = kwargs['fusion_spatial_feature_voxels_coors']

        batch_size = len(vehicle_points)
        cooperative_pretraining_baseline_shape_context_prediction_loss = torch.zeros(1).to(vehicle_points[0].device)

        for b in range(batch_size):
            vehicle_point = vehicle_points[b]
            vehicle_voxels_coor = vehicle_voxels_coors[b]
            vehicle_spatial_feature = vehicle_spatial_features[b]
            vehicle_spatial_feature_voxels_coor = vehicle_spatial_feature_voxels_coors[b].long() // 8
            vehicle_spatial_feature_voxels_coor = limit_coor_range(vehicle_spatial_feature_voxels_coor, vehicle_spatial_feature)
            fusion_spatial_feature = fusion_spatial_features[b]
            fusion_spatial_feature_voxels_coor = fusion_spatial_feature_voxels_coors[b].long() // 8


            if self.fileter_ground_points:
                none_ground_idx = torch.where(vehicle_point[:,2] > -1.6)[0]
                sample_from_none_ground = torch.randperm(none_ground_idx.shape[0])[:self.sample_num].to(vehicle_point.device)
                sampled_idx = none_ground_idx[sample_from_none_ground]
            else:
                sampled_idx = torch.randperm(vehicle_point.shape[0])[:self.sample_num].to(vehicle_point.device)

            with torch.no_grad():
                sampled_vehicle_points = vehicle_point[sampled_idx]
                fusion_point = fusion_points[b]
                distance_vehicle_to_fusion = torch.sum(
                    (sampled_vehicle_points.unsqueeze(1) - fusion_point.unsqueeze(0)) ** 2,
                    dim=-1)
                sorted, indices = torch.sort(distance_vehicle_to_fusion, dim=1)
                fix_point_number_query_indices = indices[:, :self.topk_nearest]

                queried_points = torch.gather(fusion_point.unsqueeze(0).repeat(fix_point_number_query_indices.shape[0], 1, 1)[:,:,:3], 1,
                                              fix_point_number_query_indices.unsqueeze(-1).repeat(1, 1, 3))
                source_partition = self.local_feature_extractor.compute_partitions_batch(queried_points)
                points_in_partition = []

                for partition_id in range(self.local_feature_extractor.partitions):
                    mask_q = (source_partition == partition_id)
                    points_in_current_partition = torch.sum(mask_q[:, 0, :].float(), dim=-1) + 1
                    points_in_partition.append(points_in_current_partition.unsqueeze(-1))
                points_in_partition = torch.cat(points_in_partition, dim=-1)
                # points_in_partition[points_in_partition > max_point_limit] = max_point_limit
                points_in_partition = torch.nn.functional.normalize(points_in_partition, dim=-1) * 4
                gt_dist = torch.softmax(points_in_partition, dim=-1)

            # Low resolution
            sampled_vehicle_voxels_coor_low_res = vehicle_spatial_feature_voxels_coor[sampled_idx]
            sampled_vehicle_D, sampled_vehicle_H, sampled_vehicle_W = sampled_vehicle_voxels_coor_low_res[:,
                                                                      0], sampled_vehicle_voxels_coor_low_res[:,
                                                                          1], sampled_vehicle_voxels_coor_low_res[:, 2]
            sampled_vehicle_voxel_feat_low_res = vehicle_spatial_feature[sampled_vehicle_D, sampled_vehicle_H,
                                                 sampled_vehicle_W, :]
            projected_feat_low_res = self.projector_low_res(sampled_vehicle_voxel_feat_low_res)
            # projected_feat_low_res += transformed_rotation
            predicted_dist = torch.softmax(self.projector_final(projected_feat_low_res), dim=-1)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 2.0 * self.kld(predicted_dist.log(), gt_dist)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 0.001 * self.l1_loss(predicted_dist, gt_dist)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 0.001 * self.mse_loss(predicted_dist, gt_dist)
            cooperative_pretraining_baseline_shape_context_prediction_loss += 10.0 * self.kld(predicted_dist.log(), gt_dist)

            if self.fileter_ground_points:
                none_ground_idx = torch.where(fusion_points[b][:,2] > -1.6)[0]
                sample_from_none_ground = torch.randperm(none_ground_idx.shape[0])[:self.sample_num].to(fusion_points[b].device)
                sampled_idx = none_ground_idx[sample_from_none_ground]
            else:
                sampled_idx = torch.randperm(fusion_points[b].shape[0])[:self.sample_num].to(fusion_points[b].device)

            with torch.no_grad():
                sampled_fusion_points = fusion_points[b][sampled_idx]
                fusion_point = fusion_points[b]
                distance_sample_to_fusion = torch.sum(
                    (sampled_fusion_points.unsqueeze(1) - fusion_point.unsqueeze(0)) ** 2,
                    dim=-1)
                sorted, indices = torch.sort(distance_sample_to_fusion, dim=1)
                fix_point_number_query_indices = indices[:, :self.topk_nearest]

                queried_points = torch.gather(fusion_point.unsqueeze(0).repeat(fix_point_number_query_indices.shape[0], 1, 1)[:,:,:3], 1,
                                              fix_point_number_query_indices.unsqueeze(-1).repeat(1, 1, 3))
                source_partition = self.local_feature_extractor.compute_partitions_batch(queried_points)
                points_in_partition = []

                for partition_id in range(self.local_feature_extractor.partitions):
                    mask_q = (source_partition == partition_id)
                    points_in_current_partition = torch.sum(mask_q[:, 0, :].float(), dim=-1) + 1
                    points_in_partition.append(points_in_current_partition.unsqueeze(-1))
                points_in_partition = torch.cat(points_in_partition, dim=-1)
                # points_in_partition[points_in_partition > max_point_limit] = max_point_limit
                points_in_partition = torch.nn.functional.normalize(points_in_partition, dim=-1) * 4
                gt_dist = torch.softmax(points_in_partition, dim=-1)

            # Low resolution
            sampled_fusion_voxels_coor_low_res = fusion_spatial_feature_voxels_coor[sampled_idx]
            sampled_fusion_D, sampled_fusion_H, sampled_fusion_W = sampled_fusion_voxels_coor_low_res[:,
                                                                      0], sampled_fusion_voxels_coor_low_res[:,
                                                                          1], sampled_fusion_voxels_coor_low_res[:, 2]
            sampled_fusion_voxel_feat_low_res = fusion_spatial_feature[sampled_fusion_D, sampled_fusion_H,
                                                 sampled_fusion_W, :]
            projected_feat_low_res = self.projector_low_res_fusion(sampled_fusion_voxel_feat_low_res)
            # projected_feat_low_res += transformed_rotation
            predicted_dist = torch.softmax(self.projector_final(projected_feat_low_res), dim=-1)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 2.0 * self.kld(predicted_dist.log(), gt_dist)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 0.001 * self.l1_loss(predicted_dist, gt_dist)
            # cooperative_pretraining_baseline_shape_context_prediction_loss += 0.001 * self.mse_loss(predicted_dist, gt_dist)
            cooperative_pretraining_baseline_shape_context_prediction_loss += 10.0 * self.kld(predicted_dist.log(), gt_dist)


        return cooperative_pretraining_baseline_shape_context_prediction_loss