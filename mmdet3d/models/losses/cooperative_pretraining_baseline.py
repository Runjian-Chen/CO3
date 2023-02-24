import torch
import torch.nn as nn

from ..builder import LOSSES
from ..utils.transformer import TransformerEncoder

@LOSSES.register_module()
class cooperative_pretraining_baseline_loss(nn.Module):

    def __init__(self, temperature = 0.7, sample_num=2048, fileter_ground_points = False):
        super(cooperative_pretraining_baseline_loss, self).__init__()

        self.temperature = temperature
        self.sample_num = sample_num
        self.fileter_ground_points = fileter_ground_points
        self.log_var = {}
        self.loss_func_name = 'cooperative_pretraining_baseline_loss'

    def forward(self,
                **kwargs):
        # dense_vehicle_voxel_feats: list of N_i points
        # vehicle_voxels_coors: list of [N_i, C_img]
        # dense_fusion_voxel_feats: [B, D, H, W, C]
        # fusion_voxels_coors: list of [N_i, 3]
        # fusion_points_shuffled_idx:
        pos_sim = []
        neg_sim = []

        dense_vehicle_voxel_feats = kwargs['dense_vehicle_voxel_feats']
        vehicle_voxels_coors = kwargs['vehicle_voxels_coors']
        dense_fusion_voxel_feats = kwargs['dense_fusion_voxel_feats']
        fusion_voxels_coors = kwargs['fusion_voxels_coors']
        fusion_points_shuffled_idx = kwargs['fusion_points_shuffled_idx']
        vehicle_points = kwargs['vehicle_points']

        batch_size = dense_vehicle_voxel_feats.shape[0]
        cooperative_pretraining_baseline_loss = torch.zeros(1).to(dense_vehicle_voxel_feats.device)

        for b in range(batch_size):
            # dense_vehicle_voxel_feat: [D, H, W, C]
            # vehicle_voxels_coor: [N_v, 3]
            # dense_fusion_voxel_feat: [D, H, W, C]
            # fusion_voxels_coor: [N_f, 3]
            # fusion_points_shuffled_id: [N_f,]
            dense_vehicle_voxel_feat = dense_vehicle_voxel_feats[b]
            vehicle_voxels_coor = vehicle_voxels_coors[b].long() // 8
            dense_fusion_voxel_feat = dense_fusion_voxel_feats[b]
            fusion_voxels_coor = fusion_voxels_coors[b].long() // 8
            fusion_points_shuffled_id = fusion_points_shuffled_idx[b].long()
            vehicle_point = vehicle_points[b]

            if self.fileter_ground_points:
                none_ground_idx = torch.where(vehicle_point[:,2] > -1.6)[0]
                sample_from_none_ground = torch.randperm(none_ground_idx.shape[0])[:self.sample_num].to(vehicle_voxels_coor.device)
                sampled_idx = none_ground_idx[sample_from_none_ground]
            else:
                sampled_idx = torch.randperm(vehicle_voxels_coor.shape[0])[:self.sample_num].to(vehicle_voxels_coor.device)
            # sampled_idx = torch.randint(low=0, high=vehicle_voxels_coor.shape[0],
            #                             size=(self.sample_num,)).to(vehicle_voxels_coor.device)
            sampled_vehicle_voxels_coor = vehicle_voxels_coor[sampled_idx]
            sampled_vehicle_D, sampled_vehicle_H, sampled_vehicle_W = sampled_vehicle_voxels_coor[:, 0], sampled_vehicle_voxels_coor[:,1], sampled_vehicle_voxels_coor[:,2]
            sampled_vehicle_voxel_feat = dense_vehicle_voxel_feat[sampled_vehicle_D, sampled_vehicle_H, sampled_vehicle_W, :]

            sampled_fusion_points_shuffled_id = torch.where((fusion_points_shuffled_id.unsqueeze(0) - sampled_idx.unsqueeze(1))==0)[1]
            sampled_fusion_voxels_coor = fusion_voxels_coor[sampled_fusion_points_shuffled_id]
            sampled_fusion_D, sampled_fusion_H, sampled_fusion_W = sampled_fusion_voxels_coor[:,0], sampled_fusion_voxels_coor[:,1], sampled_fusion_voxels_coor[:, 2]
            sampled_fusion_voxel_feat = dense_fusion_voxel_feat[sampled_fusion_D, sampled_fusion_H, sampled_fusion_W, :]

            # Normalization
            sampled_vehicle_voxel_feat = torch.nn.functional.normalize(sampled_vehicle_voxel_feat, dim=1)
            sampled_fusion_voxel_feat = torch.nn.functional.normalize(sampled_fusion_voxel_feat, dim=1)

            s = torch.matmul(sampled_vehicle_voxel_feat, sampled_fusion_voxel_feat.permute(1, 0).contiguous())

            s_ = s.clone().detach()
            s_pos_out = torch.diag(s_)
            s_sum_out = torch.sum(s_, dim=1)

            s = s / self.temperature
            s = torch.exp(s)

            s_pos = torch.diag(s)
            s_sum = torch.sum(s, dim=1)

            cooperative_pretraining_baseline_loss += torch.mean(-1.0 * torch.log(s_pos / s_sum))

            pos_sim.append(torch.mean(s_pos_out).unsqueeze(0))
            neg_sim.append(torch.mean(s_sum_out - s_pos_out).unsqueeze(0) / (self.sample_num - 1))

        if len(pos_sim) > 0:
            pos_sim = torch.mean(torch.cat(pos_sim))
            neg_sim = torch.mean(torch.cat(neg_sim))

            self.log_var['pos_sim'] = pos_sim
            self.log_var['neg_sim'] = neg_sim
            del pos_sim
            del neg_sim

        return cooperative_pretraining_baseline_loss