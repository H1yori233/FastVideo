# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v
from fastvideo.dataset.utils import collate_rows_from_parquet_schema


def test_collate_rows_respects_serialized_tensor_dtype() -> None:
    pil_image = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    vae_latent = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    row = {
        "pil_image_bytes": pil_image.tobytes(),
        "pil_image_shape": list(pil_image.shape),
        "pil_image_dtype": str(pil_image.dtype),
        "vae_latent_bytes": vae_latent.tobytes(),
        "vae_latent_shape": list(vae_latent.shape),
        "vae_latent_dtype": str(vae_latent.dtype),
    }

    batch = collate_rows_from_parquet_schema(
        [row],
        pyarrow_schema_i2v,
        text_padding_length=8,
    )

    assert batch["pil_image"].dtype == torch.uint8
    assert torch.equal(batch["pil_image"][0], torch.from_numpy(pil_image))
    assert batch["vae_latent"].dtype == torch.float32
    assert torch.equal(batch["vae_latent"][0], torch.from_numpy(vae_latent))
