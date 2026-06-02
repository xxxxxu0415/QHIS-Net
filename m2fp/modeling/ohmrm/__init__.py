# -*- coding: utf-8 -*-
"""
OHMRM: Occlusion-aware Human Mask Refinement Module

第一版目标：
1. 不破坏 M2FP 原始结构；
2. 接收 M2FP 的 mask_features 和 pred_masks；
3. 对人体实例 mask 做残差细化；
4. 额外输出 boundary_logits 和 occlusion_logits，方便后续加边界监督与遮挡/接触区域监督。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ohmrm import OHMRM

__all__ = ["OHMRM"]

class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        gn_groups = min(groups, out_channels)
        while out_channels % gn_groups != 0:
            gn_groups -= 1

        self.norm = nn.GroupNorm(gn_groups, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class OHMRM(nn.Module):
    """
    遮挡感知人体掩码细化模块。

    输入：
        mask_features: Tensor, [B, C, Hf, Wf]
            M2FP / Mask2Former pixel decoder 输出的 mask feature。

        pred_masks: Tensor, [B, Q, H, W]
            M2FP transformer decoder 输出的 mask logits。

        query_slice: None or tuple(start, end)
            如果只想细化 human queries，可以传入 human query 的范围。
            如果为 None，则默认细化所有 queries。

    输出：
        dict:
            pred_masks_refined: [B, Q, H, W]
            delta_masks:        [B, Q_ref, H, W]
            boundary_logits:    [B, 1, H, W]
            occlusion_logits:   [B, 1, H, W]
    """

    def __init__(
        self,
        in_channels=256,
        hidden_dim=64,
        delta_scale=0.5,
    ):
        super().__init__()

        self.delta_scale = delta_scale

        # 多尺度第一版先用 mask_features 做轻量细节增强。
        # 后面第二版再接 P2/P3/P4 多尺度特征。
        self.feature_proj = nn.Sequential(
            ConvGNAct(in_channels, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
        )

        # 将每个 query 的初始 mask logits 编码成 mask-aware feature
        self.mask_proj = nn.Sequential(
            ConvGNAct(1, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
        )

        # 残差 mask 预测头
        self.refine_head = nn.Sequential(
            ConvGNAct(hidden_dim * 2, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # 边界预测头：后续用于 boundary loss
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # 遮挡/接触困难区域预测头：后续用于 occlusion/contact loss
        self.occlusion_head = nn.Sequential(
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 关键：最后一层残差初始化为 0，保证刚接入时不明显破坏 baseline
        nn.init.constant_(self.refine_head[-1].weight, 0)
        nn.init.constant_(self.refine_head[-1].bias, 0)

    def forward(self, mask_features, pred_masks, query_slice=None):
        if mask_features.dim() != 4:
            raise ValueError(f"mask_features should be [B,C,H,W], got {mask_features.shape}")

        if pred_masks.dim() != 4:
            raise ValueError(f"pred_masks should be [B,Q,H,W], got {pred_masks.shape}")

        B, Q, H, W = pred_masks.shape

        if query_slice is None:
            q_start, q_end = 0, Q
        else:
            q_start, q_end = query_slice
            q_start = int(q_start)
            q_end = int(q_end)
            if q_start < 0 or q_end > Q or q_start >= q_end:
                raise ValueError(f"Invalid query_slice={query_slice}, Q={Q}")

        target_masks = pred_masks[:, q_start:q_end]  # [B, Q_ref, H, W]
        Q_ref = target_masks.shape[1]

        feat = self.feature_proj(mask_features)  # [B, hidden, Hf, Wf]

        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        # 为每个 query 构造对应的 refinement 输入
        feat_q = feat[:, None, :, :, :].expand(B, Q_ref, -1, -1, -1)
        feat_q = feat_q.reshape(B * Q_ref, feat.shape[1], H, W)

        mask_q = target_masks.reshape(B * Q_ref, 1, H, W)
        mask_feat = self.mask_proj(mask_q)

        fuse = torch.cat([feat_q, mask_feat], dim=1)
        delta = self.refine_head(fuse)
        delta = delta.reshape(B, Q_ref, H, W)

        refined_target = target_masks + self.delta_scale * delta

        refined_masks = pred_masks.clone()
        refined_masks[:, q_start:q_end] = refined_target

        boundary_logits = self.boundary_head(feat)
        occlusion_logits = self.occlusion_head(feat)

        return {
            "pred_masks_refined": refined_masks,
            "delta_masks": delta,
            "boundary_logits": boundary_logits,
            "occlusion_logits": occlusion_logits,
        }