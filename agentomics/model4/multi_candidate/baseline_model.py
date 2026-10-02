from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

MIRNA_LENGTH = 28
TARGET_LENGTH = 50


@dataclass
class ArchitectureConfig:
    mirbind2_num_pairs: int = 17
    mirbind2_onehot_channels: int = 18
    mirbind2_embedding_dim: int = 8
    mirbind2_dropout_rate: float = 0.2
    mirbind2_filter_sizes: Tuple[int, ...] = (128, 64, 32)
    mirbind2_kernel_sizes: Tuple[int, ...] = (6, 3, 3)
    rc_input_channels: int = 20
    rc_main_conv1_channels: int = 32
    rc_main_conv2_channels: int = 48
    rc_main_conv3_channels: int = 64
    rc_seed_conv_channels: int = 32
    rc_feature_dim: int = 160
    target_input_channels: int = 12
    mirna_phact_input_channels: int = 5
    target_conv1_channels: int = 48
    target_conv2_channels: int = 64
    mirna_phact_conv1_channels: int = 48
    mirna_phact_conv2_channels: int = 48
    feature_embedding_dim: int = 8
    dominant_embedding_dim: int = 8
    aux_numeric_dim: int = 104
    metadata_hidden_dim: int = 128
    fusion_hidden_dim: int = 224
    target_dropout: float = 0.05
    mirna_phact_dropout: float = 0.05
    metadata_dropout: float = 0.15
    fusion_dropout: float = 0.30
    feature_vocab_size: int = 2
    dominant_region_vocab_size: int = 2
    rinalmo_hidden_size: int = 480
    rinalmo_concat_dim: int = 1920
    rinalmo_branch_hidden_dim: int = 256
    rinalmo_feature_dim: int = 96
    rinalmo_dropout: float = 0.20

class PairwiseOneHotCNN(nn.Module):
    """Copied supplementary miRBind2 pairwise-onehot CNN with feature/logit access."""

    def __init__(
        self,
        num_pairs: int,
        mirna_length: int,
        target_length: int,
        embedding_dim: int = 8,
        dropout_rate: float = 0.2,
        filter_sizes: Sequence[int] = (128, 64, 32),
        kernel_sizes: Sequence[int] = (6, 3, 3),
    ):
        super().__init__()
        self.num_pairs = int(num_pairs)
        self.mirna_length = int(mirna_length)
        self.target_length = int(target_length)
        self.pair_linear = nn.Linear(self.num_pairs + 1, embedding_dim)
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        in_channels = embedding_dim
        for kernel_size, num_filters in zip(kernel_sizes, filter_sizes):
            padding = (int(kernel_size) - 1) // 2
            self.conv_layers.append(nn.Conv2d(in_channels, int(num_filters), kernel_size=int(kernel_size), padding=padding))
            self.bn_layers.append(nn.BatchNorm2d(int(num_filters)))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_channels = int(num_filters)
        self.flat_features = self._compute_flat_features()
        self.fc1 = nn.Linear(self.flat_features, 30)
        self.bn_fc = nn.BatchNorm1d(30)
        self.dropout_fc = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(30, 1)

    def _compute_flat_features(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.mirna_length, self.target_length, self.pair_linear.in_features)
            x = self.pair_linear(x)
            x = x.permute(0, 3, 1, 2)
            for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
                x = pool(F.leaky_relu(bn(conv(x)), 0.1))
            return int(x.numel())

    def features_and_logit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pair_linear(x)
        x = x.permute(0, 3, 1, 2)
        for conv, bn, pool, dropout in zip(self.conv_layers, self.bn_layers, self.pool_layers, self.dropout_layers):
            x = dropout(pool(F.leaky_relu(bn(conv(x)), 0.1)))
        x = x.contiguous().view(x.size(0), -1)
        features = self.dropout_fc(F.leaky_relu(self.bn_fc(self.fc1(x)), 0.1))
        logit = self.fc2(features).squeeze(1)
        return features, logit

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features, logit = self.features_and_logit(x)
        if return_features:
            return logit, features
        return torch.sigmoid(logit)

class FrozenEmbeddingFusionBackbone(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_encoder = PairwiseOneHotCNN(
            num_pairs=cfg.mirbind2_num_pairs,
            mirna_length=MIRNA_LENGTH,
            target_length=TARGET_LENGTH,
            embedding_dim=cfg.mirbind2_embedding_dim,
            dropout_rate=cfg.mirbind2_dropout_rate,
            filter_sizes=list(cfg.mirbind2_filter_sizes),
            kernel_sizes=list(cfg.mirbind2_kernel_sizes),
        )
        self.sequence_frozen = False
        self.rc_main_branch = nn.Sequential(
            nn.Conv2d(cfg.rc_input_channels, cfg.rc_main_conv1_channels, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.BatchNorm2d(cfg.rc_main_conv1_channels),
            nn.SiLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(cfg.rc_main_conv1_channels, cfg.rc_main_conv2_channels, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(cfg.rc_main_conv2_channels),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(cfg.rc_main_conv2_channels, cfg.rc_main_conv3_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(cfg.rc_main_conv3_channels),
            nn.SiLU(),
        )
        self.rc_seed_branch = nn.Sequential(
            nn.Conv2d(cfg.rc_input_channels, cfg.rc_seed_conv_channels, kernel_size=(7, 8), padding=0, bias=False),
            nn.BatchNorm2d(cfg.rc_seed_conv_channels),
            nn.SiLU(),
        )
        self.target_branch = nn.Sequential(
            nn.Conv1d(cfg.target_input_channels, cfg.target_conv1_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.target_conv1_channels),
            nn.SiLU(),
            nn.Dropout(cfg.target_dropout),
            nn.Conv1d(cfg.target_conv1_channels, cfg.target_conv2_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(cfg.target_conv2_channels),
            nn.SiLU(),
        )
        self.mirna_phact_branch = nn.Sequential(
            nn.Conv1d(cfg.mirna_phact_input_channels, cfg.mirna_phact_conv1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cfg.mirna_phact_conv1_channels),
            nn.SiLU(),
            nn.Dropout(cfg.mirna_phact_dropout),
            nn.Conv1d(cfg.mirna_phact_conv1_channels, cfg.mirna_phact_conv2_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(cfg.mirna_phact_conv2_channels),
            nn.SiLU(),
        )
        self.feature_embedding = nn.Embedding(cfg.feature_vocab_size, cfg.feature_embedding_dim)
        self.dominant_region_embedding = nn.Embedding(cfg.dominant_region_vocab_size, cfg.dominant_embedding_dim)
        aux_concat_dim = (
            cfg.aux_numeric_dim
            + cfg.feature_embedding_dim
            + cfg.dominant_embedding_dim
            + 2 * cfg.target_conv2_channels
            + 2 * cfg.mirna_phact_conv2_channels
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(aux_concat_dim, cfg.metadata_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.metadata_dropout),
        )
        self.rinalmo_branch = nn.Sequential(
            nn.LayerNorm(cfg.rinalmo_concat_dim),
            nn.Linear(cfg.rinalmo_concat_dim, cfg.rinalmo_branch_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.rinalmo_dropout),
            nn.Linear(cfg.rinalmo_branch_hidden_dim, cfg.rinalmo_feature_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
        )
        fusion_dim = 30 + 1 + cfg.rc_feature_dim + cfg.metadata_hidden_dim + cfg.rinalmo_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, cfg.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden_dim, 1),
        )

    def set_sequence_trainable(self, trainable: bool) -> None:
        self.sequence_frozen = not trainable
        for p in self.seq_encoder.parameters():
            p.requires_grad = bool(trainable)

    def sequence_body_parameters(self) -> List[nn.Parameter]:
        return [p for name, p in self.seq_encoder.named_parameters() if not name.startswith("fc2.")]

    def sequence_classifier_parameters(self) -> List[nn.Parameter]:
        return list(self.seq_encoder.fc2.parameters())

    def new_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.rc_main_branch,
            self.rc_seed_branch,
            self.target_branch,
            self.mirna_phact_branch,
            self.feature_embedding,
            self.dominant_region_embedding,
            self.metadata_mlp,
            self.rinalmo_branch,
            self.fusion_head,
        ]
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def rc_features(self, rc_pairwise_grid: torch.Tensor) -> torch.Tensor:
        main_h = self.rc_main_branch(rc_pairwise_grid)
        main_features = torch.cat([main_h.amax(dim=(2, 3)), main_h.mean(dim=(2, 3))], dim=1)
        seed_h = self.rc_seed_branch(rc_pairwise_grid[:, :, 1:8, :])
        seed_features = seed_h.amax(dim=(2, 3))
        return torch.cat([main_features, seed_features], dim=1)

    def forward(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        rinalmo_mirna_embedding: torch.Tensor,
        rinalmo_target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        seq_features, seq_original_logit = self.seq_encoder.features_and_logit(pairwise_onehot)
        rc_features = self.rc_features(rc_pairwise_grid)
        target_h = self.target_branch(target_tensor)
        target_features = torch.cat([target_h.amax(dim=2), target_h.mean(dim=2)], dim=1)
        mirna_h = self.mirna_phact_branch(mirna_phact_tensor)
        mirna_features = torch.cat([mirna_h.amax(dim=2), mirna_h.mean(dim=2)], dim=1)
        feat_emb = self.feature_embedding(feature_idx)
        dom_emb = self.dominant_region_embedding(dominant_region_idx)
        meta_in = torch.cat([aux_numeric, feat_emb, dom_emb, target_features, mirna_features], dim=1)
        meta = self.metadata_mlp(meta_in)
        rinalmo_absdiff = torch.abs(rinalmo_mirna_embedding - rinalmo_target_embedding)
        rinalmo_product = rinalmo_mirna_embedding * rinalmo_target_embedding
        rinalmo_in = torch.cat([rinalmo_mirna_embedding, rinalmo_target_embedding, rinalmo_absdiff, rinalmo_product], dim=1)
        rinalmo_features = self.rinalmo_branch(rinalmo_in)
        fusion_in = torch.cat([seq_features, seq_original_logit.unsqueeze(1), rc_features, meta, rinalmo_features], dim=1)
        return self.fusion_head(fusion_in).squeeze(1)


@dataclass
class LayerMixConfig:
    """Configuration for the PHACT-conditioned selected-layer residual branch."""

    selected_layer_indices: Tuple[int, ...] = (3, 6, 9, 12)
    layer_count: int = 4
    layer_pair_dim: int = 1920
    projected_dim: int = 160
    delta_hidden_dim: int = 96
    fusion_residual_dim: int = 224
    projector_dropout: float = 0.10
    delta_dropout: float = 0.10
    gate_initial_bias: Tuple[float, ...] = (-1.0, -0.5, 0.0, 1.5)
    baseline_layer_offset: int = 3


class PHACTGatedLayerMixFusionNet(FrozenEmbeddingFusionBackbone):
    """One-logit iteration-30 fusion model with a PHACT-conditioned layer residual.

    The pretrained RiNALMo encoder is external to this classifier. The input
    ``layer_pair_vectors`` contains frozen biological-token mean pools from
    RiNALMo blocks 3, 6, 9, and 12, converted per layer to
    ``[miRNA, target, abs-difference, product]``. Layer 12 is also passed through
    the original iteration-30 RiNALMo adapter. The new branch can only alter the
    original 224-dimensional fusion activation through a zero-initialized
    residual, making epoch-zero predictions exactly equal to iteration 30.
    """

    ALWAYS_FROZEN_MODULE_NAMES: Tuple[str, ...] = (
        "seq_encoder",
        "rc_main_branch",
        "rc_seed_branch",
    )
    BASE_FINETUNE_MODULE_NAMES: Tuple[str, ...] = (
        "target_branch",
        "mirna_phact_branch",
        "feature_embedding",
        "dominant_region_embedding",
        "metadata_mlp",
        "rinalmo_branch",
        "fusion_head",
    )
    NEW_BRANCH_MODULE_NAMES: Tuple[str, ...] = (
        "layer_mix_norm",
        "layer_projector",
        "layer_gate",
        "delta_mlp",
        "layer_residual_adapter",
    )

    def __init__(self, cfg: ArchitectureConfig, layer_cfg: Optional[LayerMixConfig] = None):
        super().__init__(cfg)
        self.layer_cfg = layer_cfg if layer_cfg is not None else LayerMixConfig()
        lc = self.layer_cfg
        if tuple(lc.selected_layer_indices) != (3, 6, 9, 12):
            raise ValueError("This architecture requires RiNALMo hidden-state indices [3,6,9,12]")
        if lc.layer_count != 4 or lc.layer_pair_dim != cfg.rinalmo_concat_dim:
            raise ValueError("Layer count/pair dimension conflicts with the representation contract")
        if lc.fusion_residual_dim != cfg.fusion_hidden_dim:
            raise ValueError("Residual width must equal the original fusion hidden width")

        # Shared across all selected layers, preserving a compact and symmetric branch.
        self.layer_mix_norm = nn.LayerNorm(lc.layer_pair_dim)
        self.layer_projector = nn.Sequential(
            nn.Linear(lc.layer_pair_dim, lc.projected_dim),
            nn.SiLU(),
            nn.Dropout(lc.projector_dropout),
        )
        self.layer_gate = nn.Linear(cfg.metadata_hidden_dim, lc.layer_count)
        self.delta_mlp = nn.Sequential(
            nn.Linear(lc.projected_dim, lc.delta_hidden_dim),
            nn.SiLU(),
            nn.Dropout(lc.delta_dropout),
        )
        self.layer_residual_adapter = nn.Linear(
            lc.delta_hidden_dim,
            lc.fusion_residual_dim,
            bias=False,
        )
        self.reset_layer_mix_parameters()
        self.configure_for_stage("finetune")

    def reset_layer_mix_parameters(self) -> None:
        """Use default feature transforms, a final-layer-favoring gate, and a zero residual."""
        nn.init.zeros_(self.layer_gate.weight)
        with torch.no_grad():
            self.layer_gate.bias.copy_(
                torch.tensor(
                    self.layer_cfg.gate_initial_bias,
                    dtype=self.layer_gate.bias.dtype,
                    device=self.layer_gate.bias.device,
                )
            )
        nn.init.zeros_(self.layer_residual_adapter.weight)

    @staticmethod
    def _mask_target_tensor(target_tensor: torch.Tensor) -> torch.Tensor:
        """Enforce score masks inside the model without changing valid iteration-30 inputs.

        Channels 0/1 are phyloP/phastCons values with masks in 2/3. Channels
        4:11 are seven target-PHACT scores with their finite-position mask in 11.
        """
        if target_tensor.ndim != 3 or target_tensor.shape[1] != 12:
            raise ValueError(f"target_tensor must be [B,12,50], got {tuple(target_tensor.shape)}")
        conservation_values = target_tensor[:, 0:2, :] * target_tensor[:, 2:4, :]
        conservation_masks = target_tensor[:, 2:4, :]
        phact_mask = target_tensor[:, 11:12, :]
        phact_values = target_tensor[:, 4:11, :] * phact_mask
        return torch.cat((conservation_values, conservation_masks, phact_values, phact_mask), dim=1)

    def _mask_mirna_phact_tensor(self, mirna_phact_tensor: torch.Tensor) -> torch.Tensor:
        """Enforce the position/profile mask on all miRNA-PHACT score channels."""
        expected_channels = self.cfg.mirna_phact_input_channels
        score_channels = expected_channels - 1
        if mirna_phact_tensor.ndim != 3 or mirna_phact_tensor.shape[1] != expected_channels:
            raise ValueError(
                f"mirna_phact_tensor must be [B,{expected_channels},28], got {tuple(mirna_phact_tensor.shape)}"
            )
        mask = mirna_phact_tensor[:, score_channels:expected_channels, :]
        return torch.cat((mirna_phact_tensor[:, :score_channels, :] * mask, mask), dim=1)

    def configure_for_stage(self, stage: str) -> None:
        """Set the exact warm-up or fine-tuning trainability partition."""
        if stage not in {"warmup", "finetune"}:
            raise ValueError("stage must be 'warmup' or 'finetune'")
        self._training_stage = stage
        self.requires_grad_(False)
        for name in self.NEW_BRANCH_MODULE_NAMES:
            getattr(self, name).requires_grad_(True)
        if stage == "finetune":
            for name in self.BASE_FINETUNE_MODULE_NAMES:
                getattr(self, name).requires_grad_(True)
        # Defensive assertion: the pretrained sequence/CNN branches never open.
        for name in self.ALWAYS_FROZEN_MODULE_NAMES:
            getattr(self, name).requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen BatchNorm and dropout paths must remain deterministic and immutable.
        for name in self.ALWAYS_FROZEN_MODULE_NAMES:
            getattr(self, name).eval()
        if getattr(self, "_training_stage", "finetune") == "warmup":
            for name in self.BASE_FINETUNE_MODULE_NAMES:
                getattr(self, name).eval()
        return self

    def _base_components(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        final_layer_pair_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return original fusion pre-activation and its 128-d PHACT/context state."""
        target_tensor = self._mask_target_tensor(target_tensor)
        mirna_phact_tensor = self._mask_mirna_phact_tensor(mirna_phact_tensor)

        seq_features, seq_original_logit = self.seq_encoder.features_and_logit(pairwise_onehot)
        rc_features = self.rc_features(rc_pairwise_grid)
        target_h = self.target_branch(target_tensor)
        target_features = torch.cat([target_h.amax(dim=2), target_h.mean(dim=2)], dim=1)
        mirna_h = self.mirna_phact_branch(mirna_phact_tensor)
        mirna_features = torch.cat([mirna_h.amax(dim=2), mirna_h.mean(dim=2)], dim=1)
        feat_emb = self.feature_embedding(feature_idx)
        dom_emb = self.dominant_region_embedding(dominant_region_idx)
        meta_in = torch.cat(
            [aux_numeric, feat_emb, dom_emb, target_features, mirna_features], dim=1
        )
        metadata_state = self.metadata_mlp(meta_in)
        rinalmo_features = self.rinalmo_branch(final_layer_pair_vector)
        fusion_in = torch.cat(
            [seq_features, seq_original_logit.unsqueeze(1), rc_features, metadata_state, rinalmo_features],
            dim=1,
        )
        fusion_pre_activation = self.fusion_head[0](fusion_in)
        return fusion_pre_activation, metadata_state

    def _layer_residual(
        self,
        layer_pair_vectors: torch.Tensor,
        metadata_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_pair_vectors.ndim != 3 or tuple(layer_pair_vectors.shape[1:]) != (
            self.layer_cfg.layer_count,
            self.layer_cfg.layer_pair_dim,
        ):
            raise ValueError(
                "layer_pair_vectors must be [B,4,1920], got "
                f"{tuple(layer_pair_vectors.shape)}"
            )
        projected = self.layer_projector(self.layer_mix_norm(layer_pair_vectors))
        gate_weights = torch.softmax(self.layer_gate(metadata_state), dim=1)
        mixed = torch.sum(projected * gate_weights.unsqueeze(-1), dim=1)
        delta = mixed - projected[:, self.layer_cfg.baseline_layer_offset, :]
        delta_features = self.delta_mlp(delta)
        residual = self.layer_residual_adapter(delta_features)
        return residual, gate_weights, delta_features

    def _forward_impl(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        layer_pair_vectors: torch.Tensor,
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
        residual, gate_weights, delta_features = self._layer_residual(
            layer_pair_vectors, metadata_state
        )
        fusion_with_residual = fusion_pre + residual
        hidden = self.fusion_head[2](self.fusion_head[1](fusion_with_residual))
        logits = self.fusion_head[3](hidden).squeeze(1)
        diagnostics = {
            "gate_weights": gate_weights,
            "residual": residual,
            "delta_features": delta_features,
            "metadata_state": metadata_state,
            "base_fusion_pre_activation": fusion_pre,
            "fusion_pre_activation": fusion_with_residual,
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
    ) -> torch.Tensor:
        """Return exactly one raw binary-classification logit per sample."""
        logits, _ = self._forward_impl(
            pairwise_onehot,
            rc_pairwise_grid,
            aux_numeric,
            target_tensor,
            mirna_phact_tensor,
            feature_idx,
            dominant_region_idx,
            layer_pair_vectors,
        )
        return logits

    def forward_with_diagnostics(
        self,
        pairwise_onehot: torch.Tensor,
        rc_pairwise_grid: torch.Tensor,
        aux_numeric: torch.Tensor,
        target_tensor: torch.Tensor,
        mirna_phact_tensor: torch.Tensor,
        feature_idx: torch.Tensor,
        dominant_region_idx: torch.Tensor,
        layer_pair_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return the same sole logit plus non-predictive branch diagnostics."""
        return self._forward_impl(
            pairwise_onehot,
            rc_pairwise_grid,
            aux_numeric,
            target_tensor,
            mirna_phact_tensor,
            feature_idx,
            dominant_region_idx,
            layer_pair_vectors,
        )

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups: Dict[str, List[nn.Parameter]] = {
            "new_branch": [],
            "adapter_and_head": [],
            "phact_and_metadata": [],
        }
        for name in self.NEW_BRANCH_MODULE_NAMES:
            groups["new_branch"].extend(
                p for p in getattr(self, name).parameters() if p.requires_grad
            )
        for name in ("rinalmo_branch", "fusion_head"):
            groups["adapter_and_head"].extend(
                p for p in getattr(self, name).parameters() if p.requires_grad
            )
        for name in (
            "target_branch",
            "mirna_phact_branch",
            "feature_embedding",
            "dominant_region_embedding",
            "metadata_mlp",
        ):
            groups["phact_and_metadata"].extend(
                p for p in getattr(self, name).parameters() if p.requires_grad
            )
        grouped_ids = [id(p) for values in groups.values() for p in values]
        trainable_ids = [id(p) for p in self.parameters() if p.requires_grad]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(trainable_ids):
            raise RuntimeError("Optimizer groups do not exactly partition trainable parameters")
        return groups

    def trainability_report(self) -> Dict[str, object]:
        groups = self.parameter_groups()
        return {
            "stage": self._training_stage,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "group_parameters": {
                name: sum(p.numel() for p in values) for name, values in groups.items()
            },
            "trainable_names": [name for name, p in self.named_parameters() if p.requires_grad],
            "frozen_names": [name for name, p in self.named_parameters() if not p.requires_grad],
        }


def load_iteration30_classifier_weights(
    model: PHACTGatedLayerMixFusionNet,
    checkpoint: Dict[str, object],
) -> Dict[str, object]:
    """Strictly load every iteration-30 classifier tensor and only miss new tensors."""
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    if not isinstance(state, dict):
        raise ValueError("Iteration-30 checkpoint has no classifier state dict")
    model_state = model.state_dict()
    unexpected = sorted(set(state) - set(model_state))
    incompatible_shapes = {
        key: {"checkpoint": tuple(value.shape), "model": tuple(model_state[key].shape)}
        for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    }
    if unexpected or incompatible_shapes:
        raise RuntimeError(
            {"unexpected": unexpected, "incompatible_shapes": incompatible_shapes}
        )
    result = model.load_state_dict(state, strict=False)
    missing = sorted(result.missing_keys)
    expected_missing = sorted(
        key
        for key in model_state
        if any(key == name or key.startswith(name + ".") for name in model.NEW_BRANCH_MODULE_NAMES)
    )
    if list(result.unexpected_keys) or missing != expected_missing:
        raise RuntimeError(
            {
                "missing": missing,
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
        raise RuntimeError({"iteration30_tensors_not_loaded_exactly": unequal})
    return {
        "loaded_tensor_count": len(state),
        "loaded_parameter_and_buffer_numel": int(sum(v.numel() for v in state.values())),
        "missing_new_tensor_count": len(missing),
        "missing_new_tensors": missing,
        "unexpected_keys": [],
        "incompatible_shapes": {},
        "strict_compatible_load": True,
        "all_iteration30_tensors_equal_after_load": True,
    }


def architecture_payload(
    cfg: ArchitectureConfig,
    layer_cfg: LayerMixConfig,
) -> Dict[str, object]:
    return {
        "architecture_config": asdict(cfg),
        "layer_mix_config": asdict(layer_cfg),
    }
