from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn

from baseline_model import (
    ArchitectureConfig,
    LayerMixConfig,
    PHACTGatedLayerMixFusionNet,
)


@dataclass
class CandidateAmbiguityConfig:
    max_candidates: int = 3
    candidate_feature_dim: int = 96
    metadata_dim: int = 128
    attention_dim: int = 64
    ambiguity_hidden_dim: int = 128
    ambiguity_feature_dim: int = 96
    fusion_residual_dim: int = 224
    dropout: float = 0.10


class MultiCandidatePHACTFusionNet(PHACTGatedLayerMixFusionNet):
    """Frozen layer-mix classifier plus a zero-initialized candidate residual."""

    CANDIDATE_MODULE_NAMES: Tuple[str, ...] = (
        "candidate_query",
        "candidate_key",
        "candidate_ambiguity_mlp",
        "candidate_residual_adapter",
    )

    def __init__(
        self,
        cfg: ArchitectureConfig,
        layer_cfg: LayerMixConfig,
        candidate_cfg: CandidateAmbiguityConfig,
    ) -> None:
        super().__init__(cfg, layer_cfg)
        self.candidate_cfg = candidate_cfg
        expected_feature_dim = 2 * cfg.mirna_phact_conv2_channels
        if candidate_cfg.candidate_feature_dim != expected_feature_dim:
            raise ValueError(
                "candidate_feature_dim must equal max+mean pooled miRNA branch width: "
                f"{candidate_cfg.candidate_feature_dim} != {expected_feature_dim}"
            )
        if candidate_cfg.metadata_dim != cfg.metadata_hidden_dim:
            raise ValueError("candidate metadata width differs from inherited metadata width")
        if candidate_cfg.fusion_residual_dim != cfg.fusion_hidden_dim:
            raise ValueError("candidate residual width differs from inherited fusion width")

        self.candidate_query = nn.Linear(
            candidate_cfg.metadata_dim, candidate_cfg.attention_dim
        )
        self.candidate_key = nn.Linear(
            candidate_cfg.candidate_feature_dim, candidate_cfg.attention_dim
        )
        self.candidate_ambiguity_mlp = nn.Sequential(
            nn.Linear(2 * candidate_cfg.candidate_feature_dim, candidate_cfg.ambiguity_hidden_dim),
            nn.SiLU(),
            nn.Dropout(candidate_cfg.dropout),
            nn.Linear(candidate_cfg.ambiguity_hidden_dim, candidate_cfg.ambiguity_feature_dim),
            nn.SiLU(),
        )
        self.candidate_residual_adapter = nn.Linear(
            candidate_cfg.ambiguity_feature_dim,
            candidate_cfg.fusion_residual_dim,
            bias=False,
        )
        nn.init.zeros_(self.candidate_residual_adapter.weight)
        self.configure_candidate_training()

    def configure_candidate_training(self) -> None:
        self.requires_grad_(False)
        for name in self.CANDIDATE_MODULE_NAMES:
            getattr(self, name).requires_grad_(True)

    def freeze_inherited_parameters(self) -> None:
        """Compatibility name used by the standalone inference entrypoint."""
        self.configure_candidate_training()

    def candidate_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for name in self.CANDIDATE_MODULE_NAMES:
            params.extend(p for p in getattr(self, name).parameters() if p.requires_grad)
        trainable = [p for p in self.parameters() if p.requires_grad]
        if {id(p) for p in params} != {id(p) for p in trainable}:
            raise RuntimeError("Candidate optimizer parameters do not match trainable parameters")
        return params

    def train(self, mode: bool = True):
        # First put every inherited path into deterministic evaluation mode.
        super().train(False)
        for name in self.CANDIDATE_MODULE_NAMES:
            getattr(self, name).train(mode)
        self.training = bool(mode)
        return self

    def _candidate_features(self, candidate_phact: torch.Tensor) -> torch.Tensor:
        c = self.candidate_cfg
        expected = (
            c.max_candidates,
            self.cfg.mirna_phact_input_channels,
            28,
        )
        if candidate_phact.ndim != 4 or tuple(candidate_phact.shape[1:]) != expected:
            raise ValueError(
                f"candidate_phact must be [B,{expected[0]},{expected[1]},28], "
                f"got {tuple(candidate_phact.shape)}"
            )
        batch = candidate_phact.shape[0]
        flat = candidate_phact.reshape(
            batch * c.max_candidates,
            self.cfg.mirna_phact_input_channels,
            28,
        )
        flat = self._mask_mirna_phact_tensor(flat)
        hidden = self.mirna_phact_branch(flat)
        features = torch.cat([hidden.amax(dim=2), hidden.mean(dim=2)], dim=1)
        return features.reshape(batch, c.max_candidates, c.candidate_feature_dim)

    def _candidate_residual(
        self,
        metadata_state: torch.Tensor,
        candidate_phact: torch.Tensor,
        candidate_profile_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        c = self.candidate_cfg
        if candidate_profile_mask.ndim != 2 or tuple(candidate_profile_mask.shape[1:]) != (c.max_candidates,):
            raise ValueError(
                f"candidate_profile_mask must be [B,{c.max_candidates}], "
                f"got {tuple(candidate_profile_mask.shape)}"
            )
        profile_mask = (candidate_profile_mask > 0).to(metadata_state.dtype)
        features = self._candidate_features(candidate_phact)
        slot0 = features[:, 0, :]

        query = self.candidate_query(metadata_state)
        keys = self.candidate_key(features)
        logits = torch.sum(keys * query.unsqueeze(1), dim=2) / math.sqrt(c.attention_dim)
        logits = logits.masked_fill(profile_mask == 0, -1.0e4)
        attention = torch.softmax(logits, dim=1) * profile_mask
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        attended = torch.sum(features * attention.unsqueeze(2), dim=1)
        attended_minus_slot0 = attended - slot0

        extra_mask = profile_mask[:, 1:].unsqueeze(2)
        extra_difference = torch.abs(features[:, 1:, :] - slot0.unsqueeze(1))
        mean_abs_extra_difference = (
            (extra_difference * extra_mask).sum(dim=1)
            / extra_mask.sum(dim=1).clamp_min(1.0)
        )
        ambiguity = torch.cat(
            [attended_minus_slot0, mean_abs_extra_difference], dim=1
        )
        ambiguity_features = self.candidate_ambiguity_mlp(ambiguity)
        has_extra_profile = (profile_mask[:, 1:].sum(dim=1, keepdim=True) > 0).to(
            metadata_state.dtype
        )
        residual = self.candidate_residual_adapter(ambiguity_features) * has_extra_profile
        return residual, {
            "candidate_attention": attention,
            "candidate_features": features,
            "candidate_ambiguity_features": ambiguity_features,
            "has_extra_profile": has_extra_profile,
            "candidate_residual": residual,
        }

    def _forward_candidate_impl(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        layer_pair_vectors: torch.Tensor,
        candidate_phact: torch.Tensor,
        candidate_profile_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        final_pair = layer_pair_vectors[:, self.layer_cfg.baseline_layer_offset, :]
        fusion_pre, metadata_state = self._base_components(
            pairwise_onehot,
            rc_pairwise_grid,
            aux_numeric,
            target_tensor,
            mirna_phact_tensor,
            feature_idx,
            dominant_region_idx,
            final_pair,
        )
        layer_residual, layer_gates, layer_delta_features = self._layer_residual(
            layer_pair_vectors, metadata_state
        )
        candidate_residual, candidate_diagnostics = self._candidate_residual(
            metadata_state, candidate_phact, candidate_profile_mask
        )
        combined_pre = fusion_pre + layer_residual + candidate_residual
        hidden = self.fusion_head[2](self.fusion_head[1](combined_pre))
        logits = self.fusion_head[3](hidden).squeeze(1)
        diagnostics = {
            "layer_gate_weights": layer_gates,
            "layer_residual": layer_residual,
            "layer_delta_features": layer_delta_features,
            "metadata_state": metadata_state,
            "base_fusion_pre_activation": fusion_pre,
            "combined_fusion_pre_activation": combined_pre,
            **candidate_diagnostics,
        }
        return logits, diagnostics

    def forward(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        layer_pair_vectors: torch.Tensor,
        candidate_phact: torch.Tensor,
        candidate_profile_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self._forward_candidate_impl(
            pairwise_onehot,
            rc_pairwise_grid,
            aux_numeric,
            target_tensor,
            mirna_phact_tensor,
            feature_idx,
            dominant_region_idx,
            layer_pair_vectors,
            candidate_phact,
            candidate_profile_mask,
        )
        return logits

    def forward_with_diagnostics(self, **kwargs):
        return self._forward_candidate_impl(**kwargs)


def load_layer_mix_baseline(
    model: MultiCandidatePHACTFusionNet,
    checkpoint: Mapping[str, object],
) -> Dict[str, object]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Layer-mix checkpoint has no state_dict")
    model_state = model.state_dict()
    unexpected = sorted(set(state) - set(model_state))
    incompatible = {
        key: {"checkpoint": tuple(value.shape), "model": tuple(model_state[key].shape)}
        for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    }
    if unexpected or incompatible:
        raise RuntimeError({"unexpected": unexpected, "incompatible": incompatible})
    result = model.load_state_dict(state, strict=False)
    expected_missing = sorted(
        key
        for key in model_state
        if any(
            key == name or key.startswith(name + ".")
            for name in model.CANDIDATE_MODULE_NAMES
        )
    )
    if sorted(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            {
                "missing": sorted(result.missing_keys),
                "expected_missing": expected_missing,
                "unexpected": list(result.unexpected_keys),
            }
        )
    unequal = [
        key
        for key, value in state.items()
        if not torch.equal(model.state_dict()[key].detach().cpu(), value.detach().cpu())
    ]
    if unequal:
        raise RuntimeError({"baseline_tensors_not_loaded_exactly": unequal})
    return {
        "loaded_tensor_count": len(state),
        "missing_candidate_tensors": expected_missing,
        "strict_compatible_load": True,
    }


def architecture_payload(
    cfg: ArchitectureConfig,
    layer_cfg: LayerMixConfig,
    candidate_cfg: CandidateAmbiguityConfig,
) -> Dict[str, object]:
    return {
        "model_name": "MultiCandidatePHACTFusionNet",
        "architecture_config": asdict(cfg),
        "layer_mix_config": asdict(layer_cfg),
        "candidate_ambiguity_config": asdict(candidate_cfg),
    }


__all__ = [
    "CandidateAmbiguityConfig",
    "MultiCandidatePHACTFusionNet",
    "architecture_payload",
    "load_layer_mix_baseline",
]
