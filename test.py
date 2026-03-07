# causal_fusion.py

import torch
import torch.nn as nn


class CausalFusion(nn.Module):
    """
    通用的反事实多模态融合模块。

    输入：
        - 三模态原始特征（text/audio/visual）
        - 融合函数（如 GraphSmile 的融合模块）
    操作：
        - 构造三模态 constant 特征（反事实版本）
        - 分别对原始与反事实特征执行融合
        - 应用偏差去除融合策略，输出最终特征

    示例调用：
        fusion_layer = CausalFusion(fusion_func)
        final_feature = fusion_layer(text_feat, audio_feat, video_feat)
    """

    def __init__(self, fusion_func):
        """
        参数：
            fusion_func: 模态融合函数（需支持调用方式 fusion(text, audio, video) -> (feature, aux_output)）
        """
        super(CausalFusion, self).__init__()
        self.fusion_func = fusion_func
        self.constant = nn.Parameter(torch.tensor(0.0))  # 可学习常量，用于生成反事实特征

    def forward(self, text_feat, audio_feat, video_feat):
        """
        输入：
            text_feat: (B, L, D) 文本特征
            audio_feat: (B, L, D) 音频特征
            video_feat: (B, L, D) 视频特征
        输出：
            final_feature: (B, L, D) 去偏差融合特征，可用于后续分类器
        """

        # 构造反事实特征
        text_feat_c = self.constant * torch.ones_like(text_feat)
        audio_feat_c = self.constant * torch.ones_like(audio_feat)
        video_feat_c = self.constant * torch.ones_like(video_feat)

        # 原始融合
        fusion_output, _ = self.fusion_func(text_feat, audio_feat, video_feat)

        # 反事实融合
        fusion_output_c, _ = self.fusion_func(text_feat_c, audio_feat_c, video_feat_c)

        # 加法融合策略
        TFfeature = text_feat + fusion_output
        TFcfeature = text_feat + fusion_output_c
        AFcfeature = audio_feat + fusion_output_c
        VFcfeature = video_feat + fusion_output_c

        # 加权组合，当前只使用 TF 与 TFc
        final_feature = 1 * TFfeature - 1 * TFcfeature - 0 * AFcfeature - 0 * VFcfeature

        return final_feature
