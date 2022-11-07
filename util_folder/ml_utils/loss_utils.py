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