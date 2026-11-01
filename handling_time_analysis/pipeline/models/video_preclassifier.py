"""Video pre-classifier head (README §3).

H_v [B,T,256] → Temporal Conv1D → Linear → 3-class pre_logits [B,T,3].
Used only to (a) generate predicted label segments for the dynamic vibration
windows and (b) an auxiliary CE loss. It is NOT the final classifier.
"""
import torch.nn as nn


class VideoPreClassifier(nn.Module):
    def __init__(self, video_dim=256, hidden=64, n_classes=3, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(video_dim, hidden, kernel, padding=kernel // 2)
        self.act = nn.ReLU()
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, H_v):                    # [B,T,256]
        h = self.act(self.conv(H_v.transpose(1, 2))).transpose(1, 2)   # [B,T,hidden]
        return self.fc(h)                      # pre_logits [B,T,3]
