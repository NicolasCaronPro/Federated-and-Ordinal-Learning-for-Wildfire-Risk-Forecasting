import sys
sys.path.insert(0,'/home/caron/Bureau/Model/HexagonalScale/ST-GNN-for-wildifre-prediction/Prediction/GNN/')

import torch.nn.functional as F
from typing import Optional
from forecasting_models.pytorch.tools_2 import *
from forecasting_models.pytorch.loss_utils import *

###################################### Ordinality ##########################################

class BCELoss(torch.nn.Module):
    """Binomial Cross Entropy loss for ordinal classification.

    In this repository the models return a probability for each class, e.g.
    a tensor of shape ``(N, num_classes)`` when ``num_classes`` is the number of
    ordinal categories.  In order to compute the BCE in an "all-threshold"
    fashion, these class probabilities are converted into ``num_classes - 1``
    binary classification targets representing ``P(y > k)`` for each threshold
    ``k``.
    """
    
    def __init__(self, num_classes: int):
        super(BCELoss, self).__init__()
        self.num_classes = num_classes

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the BCE loss for ordinal targets.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor of shape ``(N, num_classes)`` containing the predicted
            probabilities for each class. These probabilities will be
            converted internally into ``P(y > k)`` for each threshold ``k``.
        y_true : torch.Tensor
            Tensor of shape ``(N,)`` with integer labels in ``[0, num_classes-1]``.
        sample_weights : Optional[torch.Tensor]
            Optional tensor of shape ``(N,)`` with per-sample weights.

        Returns
        -------
        torch.Tensor
            The averaged binomial cross entropy loss.
        """

        y_pred = F.softmax(y_pred, dim=1)

        # Ensure target is one-dimensional and integer
        y_true = y_true.long().view(-1)

        # Convert class probabilities into P(y > k) for each threshold k
        cumulative = torch.cumsum(y_pred, dim=1)
        y_pred_bin = 1 - cumulative[:, :-1]

        # Create binary targets for each threshold
        thresholds = torch.arange(self.num_classes - 1, device=y_true.device)
        y_true_bin = (y_true.unsqueeze(1) > thresholds).float()

        y_pred_bin = torch.abs(y_pred_bin)

        # BCE for each threshold
        loss_per_thresh = F.binary_cross_entropy(y_pred_bin, y_true_bin, reduction="none")

        # Average loss over thresholds for each sample
        loss = loss_per_thresh.mean(dim=1)

        if sample_weights is not None:
            weighted_loss = loss * sample_weights
            return torch.sum(weighted_loss) / torch.sum(sample_weights)
        else:
            return torch.mean(loss)

class ForegroundDiceLoss(nn.Module):
    """
    Multiclass Dice sur les classes > 0 (foreground seulement).
    - pénalise X->0 (FN) et 0->X (FP)
    - n'inclut pas 0->0 (TN) car on retire la classe 0 du calcul
    - supporte ignore_index pour exclure des pixels (ex: 255)
    Optionnel: class_weights (longueur C-1) pour pondérer les classes foreground.
    """
    def __init__(self, smooth=1e-5, ignore_index=None, class_weights=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        if class_weights is not None:
            w = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("w", w / (w.sum() + 1e-12))
        else:
            self.w = None

    def forward(self, probs, target):
        """
        logits: (N, C, H, W)  -- logits multiclasse
        target: (N, H, W)     -- entiers dans [0..C-1], 0 = background
        """
        N, C = probs.shape
        assert C >= 2, "Besoin d'au moins 1 classe foreground (C>=2)."

        probs = probs.view(N, C, -1)           # (N,C,*)
        target  = target.view(N, -1)               # (N,*)

        # Masque des pixels valides (on n'ignore pas la classe 0, seulement ignore_index)
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)       # (N,H,W) bool
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # One-hot du target (on met une valeur sûre pour les pixels ignorés, puis on masquera)
        target_safe = target.clone()
        target_safe[~valid] = 0                         # peu importe: sera masqué
        t_oh = torch.zeros_like(probs).scatter_(1, target_safe.unsqueeze(1), 1.0)  # (N,C,H,W)

        # On retire la classe background (canal 0) -> ne compte pas 0->0
        p_fg = probs[:, 1:]                       # (N,C-1,H,W)
        t_fg = t_oh[:, 1:]                        # (N,C-1,H,W)

        # Appliquer le masque de validité aux deux (exclut totalement les pixels ignore_index)
        valid_ = valid.unsqueeze(1).to(p_fg.dtype)      # (N,1,H,W)
        p = p_fg * valid_
        t = t_fg * valid_

        intersection = (p * t).sum(dim=2)               # (N,C-1)
        denom = p.pow(2).sum(dim=2) + t.sum(dim=2)      # (N,C-1)
        dice_c = (2 * intersection + self.smooth) / (denom + self.smooth)  # (N,C-1)

        # Moyenne sur classes foreground (pondérée éventuelle), puis sur batch
        if self.w is not None:
            # self.w: (C-1,)
            dice_per_n = (dice_c * self.w.view(1, -1)).sum(dim=1)  # (N,)
        else:
            dice_per_n = dice_c.mean(dim=1)                         # (N,)

        loss = 1.0 - dice_per_n.mean()
        return loss

  
class MCEAndWKLoss(torch.nn.modules.loss._WeightedLoss):
    """
    The loss function integrates both MCELoss and WKLoss, concurrently minimising
    error distances while preventing the omission of classes from predictions.

    Parameters
    ----------
    num_classes : int
        Number of classes
    C: float, defaul=0.5
        Weights the WK loss (C) and the MCE loss (1-C). Must be between 0 and 1.
    wk_penalization_type : str, default='quadratic'
        The penalization type of WK loss to use (quadratic or linear).
        See WKLoss for more details.
    weight : Optional[Tensor], default=None
        A manual rescaling weight given to each class. If given, has to be a Tensor
        of size `J`, where `J` is the number of classes.
        Otherwise, it is treated as if having all ones.
    reduction : str, default='mean'
        Specifies the reduction to apply to the output: ``'none'`` | ``'mean'`` |
        ``'sum'``. ``'none'``: no reduction will be applied, ``'mean'``: the sum of
        the output will be divided by the number of elements in the output,
        ``'sum'``: the output will be summed.
    use_logits : bool, default=False
        If True, the input y_pred will be treated as logits.
        If False, the input y_pred will be treated as probabilities.
    """

    def __init__(
        self,
        num_classes: int,
        C: float = 0.5,
        wk_penalization_type: str = "quadratic",
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        use_logits=True,
        learned=False
    ) -> None:
        super().__init__(
            weight=weight, size_average=None, reduce=None, reduction=reduction
        )

        self.num_classes = num_classes
        self.learned = learned
        if learned:
            self.C = torch.nn.Parameter(torch.tensor(C))
        else:
            self.C = torch.nn.Parameter(torch.tensor(C, requires_grad=False))

        self.wk_penalization_type = wk_penalization_type

        if weight is not None and weight.shape != (num_classes,):
            raise ValueError(
                f"Weight shape {weight.shape} is not compatible"
                + "with num_classes {num_classes}"
            )
        
        if C < 0 or C > 1:
            raise ValueError(f"C must be between 0 and 1, but is {C}")

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported."
                + " Please use 'mean', 'sum' or 'none'"
            )

        self.use_logits = use_logits

        self.wk = WKLoss(
            self.num_classes,
            penalization_type=self.wk_penalization_type,
            weight=weight,
            use_logits=self.use_logits,
        )
        self.mce = MCELoss(self.num_classes, weight=weight, use_logits=self.use_logits)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Parameters
        ----------
        y_true : torch.Tensor
            Ground truth labels
        y_pred : torch.Tensor
            Predicted labels

        Returns
        -------
        loss : torch.Tensor
            The weighted sum of MCE and QWK loss
        """

        if self.learned:
            C = torch.nn.functional.sigmoid(self.C)
        else:
            C = self.C

        wk_result = self.wk(y_pred, y_true)
        mce_result = self.mce(y_pred, y_true)

        return C * wk_result + (1 - C) * mce_result
    
    def get_learnable_parameters(self):
        if not self.learned:
            return {}
        return {"C" : self.C}
    
    def get_attribute(self):
        return [('C', torch.nn.functional.sigmoid(self.C))]
    
    def plot_params(self, mcewk_logs, dir_output):

        Cs = [mcewk_log["C"] for mcewk_log in mcewk_logs]
        epochs = [mcewk_log["epoch"] for mcewk_log in mcewk_logs]

        # Sauvegarde pickle
        egpd_to_save = {"epoch": epochs, "C": Cs}
        save_object(egpd_to_save, 'Cs.pkl', dir_output)

        # kappa vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, Cs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cs')
        plt.title('MCEWK Cs over epochs')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(dir_output / 'mcewk_Cs_epochs.png')
        plt.close()

