# SPDX-License-Identifier: Apache-2.0
"""Distributed checkpoint contract for the CD-owned EMA target."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from fastvideo.train.methods.consistency_model.consistency_distillation import _EMAState


class _DistributedShadow:

    def __init__(self, value: torch.Tensor) -> None:
        self.shadow = {"weight": value.clone().float().cpu()}


def _checkpoint_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"file://{init_file}",
    )
    mesh = init_device_mesh("cpu", (world_size, ))
    local = torch.full((2, 4), float(rank + 1))
    weight = DTensor.from_local(
        local,
        device_mesh=mesh,
        placements=(Shard(0), ),
        shape=(4, 4),
        stride=(4, 1),
    )
    module = torch.nn.Linear(4, 4, bias=False)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)

    method = SimpleNamespace(
        _ema_target_model=SimpleNamespace(transformer=module),
        _target_ema=_DistributedShadow(local),
        _ema_update_count=7,
        _ema_target_dirty=False,
    )
    method._sync_ema_target = lambda: None
    state = _EMAState(method)

    dcp.save({"ema": state}, checkpoint_id=checkpoint_dir)
    dist.barrier()
    if rank == 0:
        metadata = FileSystemReader(checkpoint_dir).read_metadata()
        tensor_metadata = metadata.state_dict_metadata["ema.model.weight"]
        assert tuple(tensor_metadata.size) == (4, 4)
        assert len(tensor_metadata.chunks) == world_size
    dist.barrier()

    module.weight.to_local().zero_()
    method._target_ema.shadow["weight"].zero_()
    method._ema_update_count = 0
    dcp.load({"ema": state}, checkpoint_id=checkpoint_dir)

    expected = torch.full((2, 4), float(rank + 1))
    torch.testing.assert_close(module.weight.to_local(), expected)
    torch.testing.assert_close(method._target_ema.shadow["weight"], expected)
    assert method._ema_update_count == 7
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_ema_checkpoint_preserves_all_distributed_shards(tmp_path: Path) -> None:
    world_size = 2
    mp.spawn(
        _checkpoint_worker,
        args=(world_size, str(tmp_path / "init"), str(tmp_path / "dcp")),
        nprocs=world_size,
        join=True,
    )
