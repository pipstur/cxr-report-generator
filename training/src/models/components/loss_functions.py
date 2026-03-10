import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float().view(-1, 1)
        logits = logits.view(-1, 1)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultilabelFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        loss = (1 - pt) ** self.gamma * bce

        return loss.mean() if self.reduction == "mean" else loss.sum()


def get_loss_function(loss_function: str, focal_loss_parameters: dict):
    """Selects and returns the appropriate loss function."""
    if loss_function == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_function == "focal":
        alpha, gamma, reduction = (
            focal_loss_parameters.get("alpha", 0.8),
            focal_loss_parameters.get("gamma", 2.0),
            focal_loss_parameters.get("reduction", "mean"),
        )
        return BinaryFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
    elif loss_function == "multilabel_focal":
        gamma, reduction = (
            focal_loss_parameters.get("gamma", 2.0),
            focal_loss_parameters.get("reduction", "mean"),
        )
        return MultilabelFocalLoss(gamma=gamma, reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
