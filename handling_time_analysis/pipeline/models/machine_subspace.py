"""Machine subspace (README §11-§13): 32-D projections of video/vibration,
window-pooled video, and the DIRECTED cosine alignment loss (video → sg(vib)).
Only the machine subspace is aligned; the vibration teacher is stop-gradiented.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MachineSubspace(nn.Module):
    def __init__(self, video_dim=256, token_dim=64, machine_dim=32):
        super().__init__()
        self.w_m = nn.Linear(video_dim, machine_dim)       # video → machine
        self.w_s = nn.Linear(token_dim, machine_dim)       # vib token → machine

    def video_machine(self, H_v):              # [B,T,256] → [B,T,32]
        return self.w_m(H_v)

    def vib_machine(self, E_s):                # [B,S,64] → [B,S,32]
        return self.w_s(E_s)

    @staticmethod
    def window_pool(H_vm, windows, valid_mask=None):
        """Mean-pool video machine features inside each window → [B,S,32]."""
        B, T, D = H_vm.shape
        outs = []
        for a, b in windows:
            if valid_mask is not None:
                m = valid_mask[:, a:b].unsqueeze(-1)               # [B,L,1]
                s = (H_vm[:, a:b] * m).sum(1) / m.sum(1).clamp_min(1e-6)
            else:
                s = H_vm[:, a:b].mean(1)
            outs.append(s)
        return torch.stack(outs, dim=1)         # [B,S,32]

    @staticmethod
    def align_loss(H_vm_bar, E_sm):
        """L_align = mean_j (1 − cos(video_pool_j, sg(vib_j)))  (README §13)."""
        return (1.0 - F.cosine_similarity(H_vm_bar, E_sm.detach(), dim=-1)).mean()
