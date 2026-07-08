import torch
import torch.nn as nn

# MSE: Mean Squared Error Loss Function
class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets)**2)
        return loss
