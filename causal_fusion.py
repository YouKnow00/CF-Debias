import torch.nn as nn
import torch

class CausalFusion(nn.Module):
    def __init__(self, fusion_layer, input_dim=None, fusion_dim=None):
        super().__init__()
        self.fusion_layer = fusion_layer

        # 如果两个维度不同则投影
        if input_dim is not None and fusion_dim is not None and input_dim != fusion_dim:
            self.project_t = nn.Linear(input_dim, fusion_dim)
            self.project_v = nn.Linear(input_dim, fusion_dim)
            self.project_a = nn.Linear(input_dim, fusion_dim)
        else:
            self.project_t = self.project_v = self.project_a = nn.Identity()

    def forward(self, feat_fusion, fusion_c, feat_t, feat_v, feat_a):
        # 对齐维度
        feat_t = self.project_t(feat_t)
        feat_v = self.project_v(feat_v)
        feat_a = self.project_a(feat_a)

        # 正常计算
        TFfeature = feat_t + feat_fusion
        TFcfeature = feat_t + fusion_c
        AFcfeature = feat_a + fusion_c
        VFcfeature = feat_v + fusion_c
        Finalfeature = 1 * TFfeature - 0 * TFcfeature - 1 * AFcfeature - 1 * VFcfeature
        return Finalfeature
