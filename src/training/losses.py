import torch
import torch.nn.functional as F


class BinaryFocalLossWithLogits(torch.nn.Module):
    pos_weight: torch.Tensor | None

    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t).pow(self.gamma)

        if self.pos_weight is not None:
            alpha_t = self.pos_weight * targets + (1 - targets)
            loss = alpha_t * focal_weight * bce
        else:
            loss = focal_weight * bce

        return loss.mean()
