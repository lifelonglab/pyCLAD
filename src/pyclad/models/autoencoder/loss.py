import torch.nn as nn


class VariationalMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super(VariationalMSELoss, self).__init__()
        self.reduction = reduction

    def __call__(self, x, x_hat, mean, var):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction=self.reduction)
        kl_divergence = -0.5 * (1 + var - mean.pow(2) - var.exp())

        if self.reduction == "none":
            return reproduction_loss + kl_divergence
        elif self.reduction == "sum":
            return reproduction_loss + kl_divergence.sum()
        else:
            return reproduction_loss + kl_divergence.mean()
