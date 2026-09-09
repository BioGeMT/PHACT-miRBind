from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from phact_mirbind.data.pair_encoding import (
    decode_pair_base_ids,
    encode_pair_indices,
)
from phact_mirbind.models.rinalmo_phact import PairwiseRinalmoPhactFusion
from phact_mirbind.models.seq_only import PairwiseSeqCNN


class TinyRinalmo(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=1)
        self.embeddings = nn.Embedding(11, hidden_size)
        self.last_input_shape: tuple[int, ...] | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        del attention_mask
        self.last_input_shape = tuple(input_ids.shape)
        hidden = self.embeddings(input_ids)
        assert output_hidden_states
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden, hidden))


class TinyPhactBaseline(nn.Module):
    def forward(self, pair_indices: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        del channels
        return pair_indices[:, 0, 0].float() / 10.0


def make_model(encoding_mode: str = "independent") -> PairwiseRinalmoPhactFusion:
    mirbind = PairwiseSeqCNN(
        num_pair_classes=18,
        target_length=4,
        mirna_length=4,
        embedding_dim=2,
        filter_sizes=(4,),
        kernel_sizes=(1,),
    )
    return PairwiseRinalmoPhactFusion(
        rinalmo=TinyRinalmo(),
        mirbind=mirbind,
        mirna_phact_channels=2,
        target_phact_channels=3,
        base_token_ids=(6, 9, 7, 8),
        pad_token_id=0,
        cls_token_id=1,
        eos_token_id=2,
        layers_to_mix=1,
        attention_dim=4,
        pair_projection_dim=12,
        classifier_dim=6,
        dropout_rate=0.0,
        encoding_mode=encoding_mode,
    )


def test_rinalmo_phact_fusion_backpropagates_only_through_trainable_branches():
    pair_indices = torch.from_numpy(
        np.stack(
            [
                encode_pair_indices(
                    "ATCG", "ATCG", target_length=4, mirna_length=4
                ),
                encode_pair_indices(
                    "ATC", "ATC", target_length=4, mirna_length=4
                ),
            ]
        )
    )
    mirna_ids, mirna_mask, target_ids, target_mask = decode_pair_base_ids(
        pair_indices
    )
    model = make_model()
    model.train()

    logits = model(
        pair_indices,
        mirna_ids,
        mirna_mask,
        target_ids,
        target_mask,
        torch.randn(2, 4, 2),
        torch.randn(2, 4, 3),
    )
    logits.sum().backward()

    assert logits.shape == (2,)
    assert model.rinalmo.embeddings.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.mirbind.parameters())
    assert not model.mirbind.training


def test_rinalmo_input_keeps_internal_mask_holes_before_eos():
    model = make_model()
    base_ids = torch.tensor([[0, 0, 2, 3]])
    mask = torch.tensor([[True, False, True, True]])

    input_ids, attention_mask = model._build_inputs(base_ids, mask, total_length=6)

    assert input_ids.tolist() == [[1, 6, 0, 7, 8, 2]]
    assert attention_mask.tolist() == [[1, 1, 0, 1, 1, 1]]


def test_cross_encoding_jointly_processes_both_rnas():
    pair_indices = torch.from_numpy(
        np.stack(
            [encode_pair_indices("ATCG", "ATC", target_length=4, mirna_length=4)]
        )
    )
    mirna_ids, mirna_mask, target_ids, target_mask = decode_pair_base_ids(pair_indices)
    model = make_model(encoding_mode="cross")
    model.load_state_dict(make_model(encoding_mode="independent").state_dict())

    mirna_layers, target_layers = model._encode_pair(
        mirna_ids,
        mirna_mask,
        target_ids,
        target_mask,
    )

    assert mirna_layers.shape == (1, 4, 1, 8)
    assert target_layers.shape == (1, 4, 1, 8)
    assert model.rinalmo.last_input_shape == (1, 11)


def test_phact_residual_starts_at_frozen_baseline_logits():
    model = make_model(encoding_mode="cross")
    model.phact_baseline = TinyPhactBaseline()
    nn.init.zeros_(model.classifier[-1].weight)
    nn.init.zeros_(model.classifier[-1].bias)
    pair_indices = torch.from_numpy(
        np.stack(
            [encode_pair_indices("ATCG", "ATC", target_length=4, mirna_length=4)]
        )
    )
    mirna_ids, mirna_mask, target_ids, target_mask = decode_pair_base_ids(pair_indices)

    logits = model(
        pair_indices,
        mirna_ids,
        mirna_mask,
        target_ids,
        target_mask,
        torch.randn(1, 4, 2),
        torch.randn(1, 4, 3),
    )

    assert torch.allclose(logits, pair_indices[:, 0, 0].float() / 10.0)
