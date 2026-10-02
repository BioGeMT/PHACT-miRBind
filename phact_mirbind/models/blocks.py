"""Shared blocks for miRBind-style pairwise CNNs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_conv_config(
    filter_sizes: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
) -> None:
    if len(filter_sizes) != len(kernel_sizes):
        raise ValueError("filter_sizes and kernel_sizes must have the same length")


def make_conv_stack(
    *,
    in_channels: int,
    filter_sizes: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    dropout_rate: float,
) -> tuple[nn.ModuleList, nn.ModuleList, nn.ModuleList, nn.ModuleList]:
    conv_layers = nn.ModuleList()
    bn_layers = nn.ModuleList()
    pool_layers = nn.ModuleList()
    dropout_layers = nn.ModuleList()

    current_channels = in_channels
    for kernel_size, filter_size in zip(kernel_sizes, filter_sizes):
        padding = (kernel_size - 1) // 2
        conv_layers.append(
            nn.Conv2d(
                current_channels,
                filter_size,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        bn_layers.append(nn.BatchNorm2d(filter_size))
        pool_layers.append(nn.MaxPool2d(kernel_size=2))
        dropout_layers.append(nn.Dropout(dropout_rate))
        current_channels = filter_size

    return conv_layers, bn_layers, pool_layers, dropout_layers


def run_conv_stack(
    x: torch.Tensor,
    conv_layers: nn.ModuleList,
    bn_layers: nn.ModuleList,
    pool_layers: nn.ModuleList,
    dropout_layers: nn.ModuleList | None = None,
) -> torch.Tensor:
    if dropout_layers is None:
        for conv, bn, pool in zip(conv_layers, bn_layers, pool_layers):
            x = pool(F.leaky_relu(bn(conv(x)), 0.1))
        return x

    for conv, bn, pool, dropout in zip(
        conv_layers,
        bn_layers,
        pool_layers,
        dropout_layers,
    ):
        x = dropout(pool(F.leaky_relu(bn(conv(x)), 0.1)))
    return x
