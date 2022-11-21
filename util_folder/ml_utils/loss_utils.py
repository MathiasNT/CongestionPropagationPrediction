import torch
from torch.nn.modules.loss import _Loss
from torch import Tensor
from torch.nn import functional as F
from typing import Callable, Optional

class UpstreamBCELoss(_Loss):
    """BCE loss w. specific weights for each level of upstream
    """
    def __init__(self, pos_weights: Tensor, size_average=None, reduce=None, reduction: str = 'mean'):
        super(UpstreamBCELoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('pos_weights', pos_weights)

    def forward(self, input: Tensor, target: Tensor, net_inci_info: Tensor) -> Tensor:
        losses = []

        # Not upstream loss
        mask = net_inci_info > 0
        level_loss = F.binary_cross_entropy_with_logits(input[mask], target[mask], reduction='none')
        losses.append(level_loss)

        # Upstream levels        
        for i in range(len(self.pos_weights)):
            mask = net_inci_info == -i
            level_loss = F.binary_cross_entropy_with_logits(input[mask], target[mask],
                                                            pos_weight=self.pos_weights[i], reduction='none')
            losses.append(level_loss)
        losses = torch.cat(losses)
        return losses.mean()


class UpstreamFocalLoss(_Loss):
    """BCE loss w. specific weights for each level of upstream
    """
    def __init__(self, pos_weights: Tensor, gamma: Tensor = 2, size_average=None, reduce=None, reduction: str = 'mean'):
        super(UpstreamFocalLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('pos_weights', pos_weights)
        self.register_buffer('gamma', gamma)

    def forward(self, input: Tensor, target: Tensor, net_inci_info: Tensor) -> Tensor:
        losses = []

        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)

        # Not upstream loss
        mask = net_inci_info > 0
        level_loss = F.binary_cross_entropy_with_logits(input[mask], target[mask], reduction='none')
        level_loss = level_loss * ((1 - p_t[mask]) ** self.gamma)
        losses.append(level_loss)

        # Upstream levels        
        for i in range(len(self.pos_weights)):
            mask = net_inci_info == -i
            level_loss = F.binary_cross_entropy_with_logits(input[mask], target[mask],
                                                            pos_weight=self.pos_weights[i], reduction='none')
            level_loss = level_loss * ((1 - p_t[mask]) ** self.gamma)
            losses.append(level_loss)
        losses = torch.cat(losses)
        return losses.mean()


class FocalLoss(_Loss):
    """BCE loss w. specific weights for each level of upstream
    """
    def __init__(self, alpha: Tensor, gamma: Tensor = 2, size_average=None, reduce=None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('gamma', torch.tensor([gamma]))
        self.register_buffer('alpha', torch.tensor([alpha]))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)

        bce_loss = F.binary_cross_entropy_with_logits(input, target, pos_weight=self.alpha, reduction='none')
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.reduction == 'mean':
            loss = focal_loss.mean()
        elif self.reduction == 'sum':
            loss = focal_loss.sum()

        return loss


class KLCategorical(_Loss):
    """ KL divergence between two categorical distribution.
    """
    def __init__(self, num_atoms, eps=1e-16, size_average=None, reduce=None, reduction: str = 'mean'):
        super(KLCategorical, self).__ini__(size_average, reduce, reduction)
        self.eps = eps
        self.num_atoms = num_atoms

   
    def forward(self, preds, log_prior):
        kl_div = preds * (torch.log(preds + self.eps) - log_prior)
        return kl_div.sum() / (self.num_atoms * preds.size(0))