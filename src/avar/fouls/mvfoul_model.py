import torch
import torch.nn as nn


class SimpleVideoFoulNet(nn.Module):
    """
    Lightweight 3D CNN for foul vs non-foul classification.

    Input: B x T x C x H x W
    Output: B x 2 logits
    """

    def __init__(self, num_foul_classes: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc_foul = nn.Linear(64, num_foul_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B,T,C,H,W)
        :return: (B,num_foul_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)  # B,C,T,H,W
        feat = self.backbone(x).flatten(1)
        logits = self.fc_foul(feat)
        return logits
