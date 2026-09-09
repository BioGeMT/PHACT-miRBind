"""Fully fine-tuned RiNALMo-micro fused with frozen miRBind and PHACT."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from phact_mirbind.models.seq_only import PairwiseSeqCNN


RINALMO_ENCODING_MODES = ("independent", "cross")


class BackboneOutput(Protocol):
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


class PhactConditionedLayerMixing(nn.Module):
    """Select among the top RiNALMo layers independently at each position."""

    def __init__(self, phact_channels: int, layer_count: int) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        self.phact_norm = nn.LayerNorm(phact_channels)
        self.phact_logits = nn.Linear(phact_channels, layer_count)
        self.base_logits = nn.Parameter(torch.zeros(layer_count))

    def forward(
        self,
        layer_hidden: torch.Tensor,
        phact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_hidden.shape[:2] != phact.shape[:2]:
            raise ValueError("RiNALMo and PHACT positions must align")
        if layer_hidden.shape[2] != self.base_logits.numel():
            raise ValueError("RiNALMo layer count does not match layer mixer")

        logits = self.phact_logits(self.phact_norm(phact)) + self.base_logits
        weights = torch.softmax(logits, dim=-1)
        mixed = torch.sum(layer_hidden * weights.unsqueeze(-1), dim=2)
        return mixed, weights


class PhactConditionedPooling(nn.Module):
    """Use compact positional PHACT features to pool RNA token embeddings."""

    def __init__(self, hidden_size: int, phact_channels: int, attention_dim: int) -> None:
        super().__init__()
        if phact_channels < 1:
            raise ValueError("phact_channels must be positive")
        self.phact_norm = nn.LayerNorm(phact_channels)
        self.hidden_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.phact_projection = nn.Linear(phact_channels, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        phact: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[:2] != phact.shape[:2]:
            raise ValueError("RiNALMo and PHACT positions must align")
        if hidden.shape[:2] != position_mask.shape:
            raise ValueError("position_mask must match RiNALMo positions")
        if not position_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one valid nucleotide")

        attention_hidden = self.hidden_projection(hidden)
        attention_phact = self.phact_projection(self.phact_norm(phact))
        logits = self.attention_score(torch.tanh(attention_hidden + attention_phact))
        logits = logits.squeeze(-1).masked_fill(~position_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class PairwiseRinalmoPhactFusion(nn.Module):
    """Shared RiNALMo pair encoder with PHACT pooling and frozen miRBind."""

    def __init__(
        self,
        *,
        rinalmo: nn.Module,
        mirbind: PairwiseSeqCNN,
        mirna_phact_channels: int,
        target_phact_channels: int,
        base_token_ids: tuple[int, int, int, int],
        pad_token_id: int,
        cls_token_id: int,
        eos_token_id: int,
        layers_to_mix: int = 4,
        attention_dim: int = 64,
        pair_projection_dim: int = 256,
        classifier_dim: int = 128,
        dropout_rate: float = 0.2,
        encoding_mode: str = "independent",
        phact_baseline: nn.Module | None = None,
        phact_baseline_channel_mode: str = "both",
        phact_baseline_score_channels: tuple[int, int] = (4, 4),
        correction_scale: float = 0.25,
    ) -> None:
        super().__init__()
        config = getattr(rinalmo, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if not isinstance(hidden_size, int):
            raise ValueError("RiNALMo backbone config must expose integer hidden_size")
        backbone_layer_count = getattr(config, "num_hidden_layers", None)
        if not isinstance(backbone_layer_count, int):
            raise ValueError("RiNALMo backbone config must expose num_hidden_layers")
        if not 1 <= layers_to_mix <= backbone_layer_count:
            raise ValueError(
                f"layers_to_mix must be between 1 and {backbone_layer_count}"
            )
        if encoding_mode not in RINALMO_ENCODING_MODES:
            choices = ", ".join(RINALMO_ENCODING_MODES)
            raise ValueError(f"encoding_mode must be one of: {choices}")
        if phact_baseline_channel_mode not in ("mirna", "target", "both"):
            raise ValueError("phact_baseline_channel_mode must be mirna, target, or both")
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")

        self.rinalmo = rinalmo
        self.mirbind = mirbind
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.layers_to_mix = layers_to_mix
        self.encoding_mode = encoding_mode
        self.phact_baseline = phact_baseline
        self.phact_baseline_channel_mode = phact_baseline_channel_mode
        self.phact_baseline_score_channels = phact_baseline_score_channels
        self.correction_scale = correction_scale
        self._rinalmo_frozen = False
        self.register_buffer(
            "base_token_ids",
            torch.tensor(base_token_ids, dtype=torch.long),
            persistent=True,
        )

        for parameter in self.mirbind.parameters():
            parameter.requires_grad = False
        self.mirbind.eval()
        if self.phact_baseline is not None:
            for parameter in self.phact_baseline.parameters():
                parameter.requires_grad = False
            self.phact_baseline.eval()

        self.mirna_layer_mix = PhactConditionedLayerMixing(
            mirna_phact_channels,
            layers_to_mix,
        )
        self.target_layer_mix = PhactConditionedLayerMixing(
            target_phact_channels,
            layers_to_mix,
        )
        self.mirna_pool = PhactConditionedPooling(
            hidden_size,
            mirna_phact_channels,
            attention_dim,
        )
        self.target_pool = PhactConditionedPooling(
            hidden_size,
            target_phact_channels,
            attention_dim,
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(hidden_size * 4, pair_projection_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Sequential(
            nn.Linear(pair_projection_dim + 30, classifier_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_dim, 1),
        )
        if self.phact_baseline is not None:
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mirbind.eval()
        if self.phact_baseline is not None:
            self.phact_baseline.eval()
        if self._rinalmo_frozen:
            self.rinalmo.eval()
        return self

    def freeze_rinalmo(self) -> None:
        for parameter in self.rinalmo.parameters():
            parameter.requires_grad = False
        self.rinalmo.eval()
        self._rinalmo_frozen = True

    def forward(
        self,
        pair_indices: torch.Tensor,
        mirna_base_ids: torch.Tensor,
        mirna_mask: torch.Tensor,
        target_base_ids: torch.Tensor,
        target_mask: torch.Tensor,
        mirna_phact: torch.Tensor,
        target_phact: torch.Tensor,
    ) -> torch.Tensor:
        mirna_layers, target_layers = self._encode_pair(
            mirna_base_ids,
            mirna_mask,
            target_base_ids,
            target_mask,
        )
        mirna_hidden, _ = self.mirna_layer_mix(mirna_layers, mirna_phact)
        target_hidden, _ = self.target_layer_mix(target_layers, target_phact)
        mirna_vector, _ = self.mirna_pool(
            mirna_hidden,
            mirna_phact,
            mirna_mask,
        )
        target_vector, _ = self.target_pool(
            target_hidden,
            target_phact,
            target_mask,
        )
        pair_vector = torch.cat(
            [
                mirna_vector,
                target_vector,
                mirna_vector * target_vector,
                torch.abs(mirna_vector - target_vector),
            ],
            dim=1,
        )
        rinalmo_features = self.pair_projection(pair_vector)
        with torch.no_grad():
            mirbind_features = self.mirbind.encode(pair_indices)
        correction = self.classifier(
            torch.cat([rinalmo_features, mirbind_features], dim=1)
        ).squeeze(-1)
        if self.phact_baseline is None:
            return correction
        with torch.no_grad():
            baseline_logits = self._phact_baseline_logits(
                pair_indices,
                mirna_phact,
                target_phact,
            )
        return baseline_logits + self.correction_scale * correction

    def _phact_baseline_logits(
        self,
        pair_indices: torch.Tensor,
        mirna_phact: torch.Tensor,
        target_phact: torch.Tensor,
    ) -> torch.Tensor:
        assert self.phact_baseline is not None
        mirna_score_count, target_score_count = self.phact_baseline_score_channels
        mirna = mirna_phact[:, :, :mirna_score_count]
        target = target_phact[:, :, :target_score_count]
        batch_size, mirna_length = mirna.shape[:2]
        target_length = target.shape[1]
        mirna_channels = (
            mirna.permute(0, 2, 1)
            .unsqueeze(3)
            .expand(batch_size, mirna.shape[2], mirna_length, target_length)
        )
        target_channels = (
            target.permute(0, 2, 1)
            .unsqueeze(2)
            .expand(batch_size, target.shape[2], mirna_length, target_length)
        )
        if self.phact_baseline_channel_mode == "mirna":
            channels = mirna_channels.contiguous()
        elif self.phact_baseline_channel_mode == "target":
            channels = target_channels.contiguous()
        else:
            channels = torch.cat([mirna_channels, target_channels], dim=1).contiguous()
        return self.phact_baseline(pair_indices, channels)

    def _encode_pair(
        self,
        mirna_base_ids: torch.Tensor,
        mirna_mask: torch.Tensor,
        target_base_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoding_mode == "cross":
            return self._encode_pair_cross(
                mirna_base_ids,
                mirna_mask,
                target_base_ids,
                target_mask,
            )
        total_length = max(mirna_base_ids.shape[1], target_base_ids.shape[1]) + 2
        mirna_input_ids, mirna_attention = self._build_inputs(
            mirna_base_ids,
            mirna_mask,
            total_length,
        )
        target_input_ids, target_attention = self._build_inputs(
            target_base_ids,
            target_mask,
            total_length,
        )
        input_ids = torch.cat([mirna_input_ids, target_input_ids], dim=0)
        attention_mask = torch.cat([mirna_attention, target_attention], dim=0)
        output: BackboneOutput = self.rinalmo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        batch_size = mirna_base_ids.shape[0]
        selected_layers = output.hidden_states[-self.layers_to_mix :]
        layer_hidden = torch.stack(selected_layers, dim=2)
        mirna_hidden = layer_hidden[
            :batch_size,
            1 : mirna_base_ids.shape[1] + 1,
        ]
        target_hidden = layer_hidden[
            batch_size:,
            1 : target_base_ids.shape[1] + 1,
        ]
        return mirna_hidden, target_hidden

    def _encode_pair_cross(
        self,
        mirna_base_ids: torch.Tensor,
        mirna_mask: torch.Tensor,
        target_base_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both RNAs jointly so backbone attention can cross molecules."""
        batch_size = mirna_base_ids.shape[0]
        mirna_length = mirna_base_ids.shape[1]
        target_length = target_base_ids.shape[1]
        target_start = mirna_length + 2
        total_length = target_start + target_length + 1
        input_ids = torch.full(
            (batch_size, total_length),
            self.pad_token_id,
            dtype=torch.long,
            device=mirna_base_ids.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1

        self._write_sequence_segment(
            input_ids,
            attention_mask,
            base_ids=mirna_base_ids,
            position_mask=mirna_mask,
            start=1,
        )
        self._write_sequence_segment(
            input_ids,
            attention_mask,
            base_ids=target_base_ids,
            position_mask=target_mask,
            start=target_start,
        )
        output: BackboneOutput = self.rinalmo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        selected_layers = output.hidden_states[-self.layers_to_mix :]
        layer_hidden = torch.stack(selected_layers, dim=2)
        return (
            layer_hidden[:, 1 : mirna_length + 1],
            layer_hidden[:, target_start : target_start + target_length],
        )

    def _write_sequence_segment(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        base_ids: torch.Tensor,
        position_mask: torch.Tensor,
        start: int,
    ) -> None:
        if not position_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one valid nucleotide")
        sequence_length = base_ids.shape[1]
        safe_base_ids = base_ids.masked_fill(~position_mask, 0)
        nucleotide_ids = self.base_token_ids[safe_base_ids]
        input_ids[:, start : start + sequence_length] = torch.where(
            position_mask,
            nucleotide_ids,
            self.pad_token_id,
        )
        attention_mask[:, start : start + sequence_length] = position_mask.long()
        positions = torch.arange(sequence_length, device=base_ids.device)
        last_valid_positions = torch.where(
            position_mask,
            positions.unsqueeze(0),
            -1,
        ).amax(dim=1)
        eos_positions = start + last_valid_positions + 1
        input_ids.scatter_(1, eos_positions.unsqueeze(1), self.eos_token_id)
        attention_mask.scatter_(1, eos_positions.unsqueeze(1), 1)

    def _build_inputs(
        self,
        base_ids: torch.Tensor,
        position_mask: torch.Tensor,
        total_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = base_ids.shape
        if not position_mask.any(dim=1).all():
            raise ValueError("every sequence must contain at least one valid nucleotide")
        input_ids = torch.full(
            (batch_size, total_length),
            self.pad_token_id,
            dtype=torch.long,
            device=base_ids.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        input_ids[:, 0] = self.cls_token_id
        attention_mask[:, 0] = 1

        safe_base_ids = base_ids.masked_fill(~position_mask, 0)
        nucleotide_ids = self.base_token_ids[safe_base_ids]
        input_ids[:, 1 : sequence_length + 1] = torch.where(
            position_mask,
            nucleotide_ids,
            self.pad_token_id,
        )
        attention_mask[:, 1 : sequence_length + 1] = position_mask.long()
        positions = torch.arange(sequence_length, device=base_ids.device)
        last_valid_positions = torch.where(
            position_mask,
            positions.unsqueeze(0),
            -1,
        ).amax(dim=1)
        eos_positions = last_valid_positions + 2
        input_ids.scatter_(1, eos_positions.unsqueeze(1), self.eos_token_id)
        attention_mask.scatter_(1, eos_positions.unsqueeze(1), 1)
        return input_ids, attention_mask
