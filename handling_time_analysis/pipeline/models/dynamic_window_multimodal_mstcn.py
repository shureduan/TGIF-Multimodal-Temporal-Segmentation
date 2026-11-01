"""Checkpoint-matched TGIF 256-D attention model.

The historical attn_masked_seed*.pt files use Linear(1536,256), three layer
normalizations, and LN(LN(Hv) + softplus(alpha) * gate * LN(context)).
This definition restores that version. State loading is strict. Forward parity
is checked against the preserved training-era Python 3.10 bytecode locally.
A later 512-D GatedProjection model is a different architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mstcn_core import MSTCNStages

from .video_preclassifier import VideoPreClassifier
from .vibration_token_encoder import VibrationTokenEncoder
from .one_way_local_attention import OneWayLocalAttention, build_local_mask
from .machine_subspace import MachineSubspace
from .auxiliary_binary_heads import AuxiliaryBinaryHeads
from .js_conflict import js_conflict
from .risk_gate import RiskGate


class DynamicWindowMultimodalMSTCN(nn.Module):
    def __init__(self, in_dim=1536, video_dim=256, token_dim=64, machine_dim=32,
                 attention_dim=64, num_heads=2, neighbors=1, dropout=0.10,
                 n_stages=4, n_layers=8, ch=64, n_classes=3):
        super().__init__()
        # Training-era 1536 -> 256 projection; keys match attn_masked_seed*.pt.
        self.w_video = nn.Sequential(nn.Linear(in_dim, video_dim), nn.LayerNorm(video_dim), nn.ReLU())
        self.pre = VideoPreClassifier(video_dim, 64, n_classes)
        self.vib_enc = VibrationTokenEncoder(2, 32, token_dim)
        self.attn = OneWayLocalAttention(video_dim, token_dim, attention_dim,
                                         num_heads, dropout)
        self.machine = MachineSubspace(video_dim, token_dim, machine_dim)
        self.aux = AuxiliaryBinaryHeads(machine_dim)
        self.gate = RiskGate()
        self.neighbors = neighbors
        # Three learned normalization layers belong to the saved architecture.
        self.video_norm = nn.LayerNorm(video_dim)
        self.context_norm = nn.LayerNorm(video_dim)
        self.output_norm = nn.LayerNorm(video_dim)
        self.alpha_param = nn.Parameter(torch.tensor(-0.4327))      # softplus approximately 0.5 at initialization
        self.mstcn = MSTCNStages(in_dim=video_dim, n_stages=n_stages,
                                 n_layers=n_layers, ch=ch, n_classes=n_classes)

    def forward(self, video_features, vibration_features=None, windows=None,
                valid_mask=None, has_vib=True, allow_mask=None):
        """
        video_features:     [B,T,1536]
        vibration_features: [B,S,2] station-normalized [peak_energy, psd_20_40]
        windows:            list of (start,end) frame windows — S multi-scale tokens
        allow_mask:         [T,S] bool candidate set (build_multiscale_tokens); if
                            None a coverage±neighbors mask is built from `windows`
        Returns dict (README §25 + attention_weights, allow_mask, r_inject).
        """
        H_v = self.w_video(video_features)                          # [B,T,256]
        pre_logits = self.pre(H_v)                                  # [B,T,3]
        B, T, _ = H_v.shape

        if (not has_vib) or vibration_features is None or windows is None \
                or len(windows) == 0:
            fused = self.output_norm(self.video_norm(H_v))
            out = self._classify(fused)
            return dict(logits=out, pre_logits=pre_logits, fused_features=fused,
                        vibration_tokens=None, attention_weights=None, allow_mask=None,
                        video_binary_prob=None, vibration_binary_prob=None,
                        js_conflict=None, risk_gate_window=None, risk_gate_time=None,
                        H_vm_bar=None, E_sm=None, r_inject=None)

        E_s = self.vib_enc(vibration_features)                      # [B,S,64]
        allow = allow_mask if allow_mask is not None else build_local_mask(
            T, windows, self.neighbors)                             # [T,S]
        C_hat, A = self.attn(H_v, E_s, allow)                       # [B,T,256],[B,T,S]

        H_vm = self.machine.video_machine(H_v)                      # [B,T,32]
        E_sm = self.machine.vib_machine(E_s)                        # [B,S,32]
        H_vm_bar = self.machine.window_pool(H_vm, windows, valid_mask)  # [B,S,32]
        p_v, p_s = self.aux(H_vm_bar, E_sm)                         # [B,S]
        D = js_conflict(p_v, p_s)                                   # [B,S]
        g_j = self.gate(D, p_v, p_s)                                # [B,S]
        g_t = (A * g_j.unsqueeze(1)).sum(-1, keepdim=True)          # [B,T,1]

        v_norm = self.video_norm(H_v)
        c_norm = self.context_norm(C_hat)
        alpha = F.softplus(self.alpha_param)
        inj = alpha * g_t * c_norm
        fused = self.output_norm(v_norm + inj)
        r_inject = (inj.norm(dim=-1) / v_norm.norm(dim=-1).clamp_min(1e-6)).mean().detach()
        out = self._classify(fused)
        return dict(logits=out, pre_logits=pre_logits, fused_features=fused,
                    vibration_tokens=E_s, attention_weights=A, allow_mask=allow,
                    video_binary_prob=p_v, vibration_binary_prob=p_s,
                    js_conflict=D, risk_gate_window=g_j, risk_gate_time=g_t,
                    H_vm_bar=H_vm_bar, E_sm=E_sm, r_inject=r_inject)

    def _classify(self, fused):                 # [B,T,256] → [S,B,3,T]
        return self.mstcn(fused.transpose(1, 2))
