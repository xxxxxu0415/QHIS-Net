"""
OHMRM: Occlusion-aware Human Mask Refinement Module
Version: optional multi-scale gated residual fusion.

设计原则：
1. 不修改 M2FP 原始 mask_features 主路径；
2. pred_masks 之后做残差式 refinement；
3. 多尺度特征默认关闭 use_multiscale=False；
4. 多尺度特征只作为 gate 控制的残差补充；
5. refine_head 最后一层零初始化，初始输出等于 baseline。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        in_channels=256,
        hidden_dim=32,
        delta_scale=0.5,
        query_chunk_size=4,
        use_multiscale=False,
        ms_alpha=0.5,
        ms_gate_init=-6.0,
        max_ms_levels=3,
    ):
        super().__init__()

        self.delta_scale = float(delta_scale)
        self.query_chunk_size = int(query_chunk_size)

        # 多尺度默认关闭
        self.use_multiscale = bool(use_multiscale)
        self.ms_alpha = float(ms_alpha)
        self.max_ms_levels = int(max_ms_levels)

        # 原始 M2FP mask_features 主分支
        self.feature_proj = nn.Sequential(
            ConvGNAct(in_channels, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
        )

        # 多尺度残差分支：只作为辅助，不替换原始特征
        self.ms_proj = nn.Sequential(
            ConvGNAct(in_channels, hidden_dim, 1, 1, 0),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
        )

        # gate 初始接近 0
        # sigmoid(-6) ≈ 0.0025，所以初始多尺度影响极弱
        self.ms_gate = nn.Parameter(torch.tensor(float(ms_gate_init)))

        # 初始 mask-aware 编码
        self.mask_proj = nn.Sequential(
            ConvGNAct(1, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
        )

        # 残差 mask refinement head
        self.refine_head = nn.Sequential(
            ConvGNAct(hidden_dim * 2, hidden_dim, 3, 1, 1),
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # 后续加 boundary loss 时使用
        self.boundary_head = nn.Sequential(
            ConvGNAct(hidden_dim, hidden_dim, 3, 1, 1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

        # 后续加 occlusion/contact loss 时使用
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

        # 关键：初始残差为 0
        # 这样刚接入时 pred_masks_refined == pred_masks
        nn.init.constant_(self.refine_head[-1].weight, 0)
        nn.init.constant_(self.refine_head[-1].bias, 0)

    def _build_multiscale_residual(self, multi_scale_features, target_size):
        """
        multi_scale_features: list/tuple of [B, C, Hi, Wi]
        target_size: (H, W)
        return: [B, hidden_dim, H, W]
        """
        if multi_scale_features is None:
            return None

        if not isinstance(multi_scale_features, (list, tuple)):
            return None

        ms_feats = []
        levels = list(multi_scale_features)[: self.max_ms_levels]

        for feat in levels:
            if feat is None:
                continue

            ms = self.ms_proj(feat)

            if ms.shape[-2:] != target_size:
                ms = F.interpolate(
                    ms,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            ms_feats.append(ms)

        if len(ms_feats) == 0:
            return None

        # 用平均而不是 concat，降低参数和扰动强度
        ms_residual = torch.stack(ms_feats, dim=0).mean(dim=0)
        return ms_residual

    def _refine_chunk(self, feat, masks_chunk):
        """
        feat:         [B, C, H, W]
        masks_chunk: [B, q, H, W]
        return:      [B, q, H, W]
        """
        B, C, H, W = feat.shape
        q = masks_chunk.shape[1]

        feat_q = feat[:, None].expand(B, q, C, H, W).reshape(B * q, C, H, W)
        mask_q = masks_chunk.reshape(B * q, 1, H, W)

        mask_feat = self.mask_proj(mask_q)

        fuse = torch.cat([feat_q, mask_feat], dim=1)
        delta = self.refine_head(fuse)

        return delta.reshape(B, q, H, W)

    def forward(
        self,
        mask_features,
        pred_masks,
        multi_scale_features=None,
        query_slice=None,
    ):
        """
        mask_features:        [B, C, Hf, Wf]
        pred_masks:           [B, Q, H, W]
        multi_scale_features: optional list of [B, C, Hi, Wi]
        query_slice:          optional tuple(start, end)
        """

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

        target_masks = pred_masks[:, q_start:q_end]
        Q_ref = target_masks.shape[1]

        # 主分支：原始 M2FP mask_features
        feat = self.feature_proj(mask_features)

        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(
                feat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # 可选多尺度残差补充
        # 默认 use_multiscale=False，所以不会进入这里
        if self.use_multiscale:
            ms_residual = self._build_multiscale_residual(
                multi_scale_features,
                target_size=(H, W),
            )

            if ms_residual is not None:
                gate = torch.sigmoid(self.ms_gate) * self.ms_alpha
                feat = feat + gate * ms_residual

        delta_chunks = []
        chunk = max(1, int(self.query_chunk_size))

        for s in range(0, Q_ref, chunk):
            e = min(s + chunk, Q_ref)
            delta_chunks.append(
                self._refine_chunk(
                    feat,
                    target_masks[:, s:e],
                )
            )

        delta = torch.cat(delta_chunks, dim=1)

        # local/difficulty residual gate
        # 只在低置信区域和边界区域施加 residual，避免 global residual 扰乱 baseline mask
        prob = target_masks.sigmoid()

        low_conf = 1.0 - (prob - 0.5).abs() * 2.0
        low_conf = low_conf.clamp(0, 1)

        dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1])
        dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        boundary = (dx + dy).clamp(0, 1)

        W_local = (0.5 * low_conf + 0.5 * boundary).clamp(0, 1).detach()

        refined_masks = pred_masks.clone()
        refined_masks[:, q_start:q_end] = target_masks + self.delta_scale * W_local * delta

        boundary_logits = self.boundary_head(feat)
        occlusion_logits = self.occlusion_head(feat)

        return {
            "pred_masks_refined": refined_masks,
            "delta_masks": delta,
            "boundary_logits": boundary_logits,
            "occlusion_logits": occlusion_logits,
            "ms_gate": torch.sigmoid(self.ms_gate).detach(),
        }