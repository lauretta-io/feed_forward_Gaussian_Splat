from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

from experiments.resplat_u2net_saliency.common import Timer
from experiments.resplat_u2net_saliency.models.u2net_original import load_u2net_checkpoint


def test_checkpoint_without_matching_weights_is_rejected(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    checkpoint = tmp_path / "incompatible.pth"
    torch.save({"state_dict": {"official_u2net.stage1.weight": torch.ones(1)}}, checkpoint)

    with pytest.raises(ValueError, match="no parameter keys matching"):
        load_u2net_checkpoint(model, checkpoint, strict=False)


def test_checkpoint_matching_only_a_buffer_is_rejected(tmp_path: Path) -> None:
    model = nn.BatchNorm1d(2)
    checkpoint = tmp_path / "buffer-only.pth"
    torch.save({"state_dict": {"num_batches_tracked": torch.tensor(4)}}, checkpoint)

    with pytest.raises(ValueError, match="no parameter keys matching"):
        load_u2net_checkpoint(model, checkpoint, strict=False)


def test_timer_synchronizes_the_requested_cuda_device() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize") as synchronize,
        Timer("cuda:1"),
    ):
        pass

    assert synchronize.call_count == 2
    synchronize.assert_called_with(device=torch.device("cuda:1"))
