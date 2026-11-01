"""Auxiliary binary heads (README §14): sewing vs non-sewing (idle+handling)
in the shared 32-D machine subspace. One head per modality."""
import torch
import torch.nn as nn


class AuxiliaryBinaryHeads(nn.Module):
    def __init__(self, machine_dim=32):
        super().__init__()
        self.head_v = nn.Linear(machine_dim, 1)
        self.head_s = nn.Linear(machine_dim, 1)

    def forward(self, H_vm_bar, E_sm):         # [B,S,32] each
        p_v = torch.sigmoid(self.head_v(H_vm_bar)).squeeze(-1)   # [B,S]
        p_s = torch.sigmoid(self.head_s(E_sm)).squeeze(-1)       # [B,S]
        return p_v, p_s
