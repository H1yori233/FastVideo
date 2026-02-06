import torch
import torch.nn as nn
import torch.nn.functional as F


class WanGameLingbotCamConditioner(nn.Module):
    """Camera-control injector used by Lingbot-style Wan blocks."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.cam_injector_layer1.weight)
        nn.init.zeros_(self.cam_injector_layer1.bias)
        nn.init.xavier_uniform_(self.cam_injector_layer2.weight)
        nn.init.zeros_(self.cam_injector_layer2.bias)
        nn.init.xavier_uniform_(self.cam_scale_layer.weight)
        nn.init.zeros_(self.cam_scale_layer.bias)
        nn.init.xavier_uniform_(self.cam_shift_layer.weight)
        nn.init.zeros_(self.cam_shift_layer.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        c2ws_plucker_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        if c2ws_plucker_emb is None:
            return hidden_states
        c2ws_hidden_states = self.cam_injector_layer2(
            F.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
        c2ws_hidden_states = c2ws_hidden_states + c2ws_plucker_emb
        cam_scale = self.cam_scale_layer(c2ws_hidden_states)
        cam_shift = self.cam_shift_layer(c2ws_hidden_states)
        return (1.0 + cam_scale) * hidden_states + cam_shift