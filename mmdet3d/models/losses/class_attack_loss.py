"""
ClassAttackLoss — adversarial classification loss framework.

Drop-in replacement for FocalLoss (use_sigmoid=True) in TransFusionHeadV2.
All attack terms are disabled by default so the baseline is reproduced
exactly when only use_original_focal=True.

Tensor conventions (from TransFusionHeadV2 call site)
-------------------------------------------------------
  pred   : (N, C)  raw logits, N = batch × num_proposals, C = num_classes
  target : (N,)    integer class index; foreground ∈ {0…C-1}, background = C
  weight : (N,)    per-proposal scalar weight (label_weights)

Only foreground proposals (target < C) are used in the adversarial terms.

Total loss
----------
  L = lambda_original  * L_original   [standard sigmoid focal loss]
    + lambda_reverse   * L_reverse    [−FocalLoss — maximise classification loss]
    + lambda_complement * L_complement [−log(1−p_y) — push correct class down]
    + lambda_uniform   * L_uniform    [BCE toward uniform wrong-class target]
    + lambda_margin    * L_margin     [relu(z_y − z_wrong + margin)]
    + lambda_hard_wrong * L_hard_wrong [−log(p_wrong) — push hardest wrong class up]

Inactive terms contribute exactly zero and carry no gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.focal_loss import (
    sigmoid_focal_loss,
    py_sigmoid_focal_loss,
)
from mmdet.models.losses.utils import weight_reduce_loss


# ─── helpers ─────────────────────────────────────────────────────────────────

def _fg_mask(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Boolean mask selecting foreground proposals."""
    return target < num_classes


def _get_num_classes(pred: torch.Tensor) -> int:
    return pred.shape[1]


# ─── main loss module ─────────────────────────────────────────────────────────

@LOSSES.register_module()
class ClassAttackLoss(nn.Module):
    """Adversarial classification loss framework for IS-Fusion attack research.

    Args:
        use_sigmoid (bool): Must be True — only sigmoid path is supported.
        gamma, alpha: FocalLoss hyperparameters.
        reduction, loss_weight: standard mmdet conventions.
        use_*: enable / disable each component.
        lambda_*: per-component scale applied before summing.
        margin: hinge margin for L_margin (default 0).
        debug: if True, print component values each forward pass.
    """

    def __init__(
        self,
        # ── FocalLoss passthrough ────────────────────────────────────────
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        # ── component flags ──────────────────────────────────────────────
        use_original_focal: bool = True,
        use_reverse_focal: bool = False,
        use_complement_loss: bool = False,
        use_uniform_confusion: bool = False,
        use_margin_confusion: bool = False,
        use_hard_wrong_class: bool = False,
        # ── lambda weights ───────────────────────────────────────────────
        lambda_original: float = 1.0,
        lambda_reverse: float = 1.0,
        lambda_complement: float = 1.0,
        lambda_uniform: float = 1.0,
        lambda_margin: float = 1.0,
        lambda_hard_wrong: float = 1.0,
        # ── tuning ───────────────────────────────────────────────────────
        margin: float = 0.0,
        # ── misc ─────────────────────────────────────────────────────────
        debug: bool = False,
    ):
        super(ClassAttackLoss, self).__init__()
        assert use_sigmoid, 'ClassAttackLoss only supports use_sigmoid=True.'

        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.use_original_focal = use_original_focal
        self.use_reverse_focal = use_reverse_focal
        self.use_complement_loss = use_complement_loss
        self.use_uniform_confusion = use_uniform_confusion
        self.use_margin_confusion = use_margin_confusion
        self.use_hard_wrong_class = use_hard_wrong_class

        self.lambda_original = lambda_original
        self.lambda_reverse = lambda_reverse
        self.lambda_complement = lambda_complement
        self.lambda_uniform = lambda_uniform
        self.lambda_margin = lambda_margin
        self.lambda_hard_wrong = lambda_hard_wrong

        self.margin = margin
        self.eps = 1e-8
        self.debug = debug

        # Detached per-component values available after each forward call.
        self.last_components: dict = {}

    # ─── internal focal loss call ─────────────────────────────────────────────

    def _focal(self, pred, target, weight, reduction, avg_factor):
        """Call the appropriate (CUDA or Python) sigmoid focal loss."""
        if torch.cuda.is_available() and pred.is_cuda:
            return sigmoid_focal_loss(
                pred, target, weight,
                gamma=self.gamma, alpha=self.alpha,
                reduction=reduction, avg_factor=avg_factor,
            )
        else:
            num_classes = pred.size(1)
            target_oh = F.one_hot(target, num_classes=num_classes + 1)
            target_oh = target_oh[:, :num_classes].float()
            return py_sigmoid_focal_loss(
                pred, target_oh, weight,
                gamma=self.gamma, alpha=self.alpha,
                reduction=reduction, avg_factor=avg_factor,
            )

    # ─── component methods ────────────────────────────────────────────────────

    def _loss_original(self, pred, target, weight, reduction, avg_factor):
        """Standard sigmoid FocalLoss."""
        return self.loss_weight * self._focal(
            pred, target, weight, reduction, avg_factor
        )

    def _loss_reverse(self, pred, target, weight, reduction, avg_factor):
        """Negative FocalLoss — directly maximises the classification loss."""
        return -self.loss_weight * self._focal(
            pred, target, weight, reduction, avg_factor
        )

    def _loss_complement(self, pred, target):
        """
        Complement loss — push probability of the correct class toward zero.

        L_complement = -log(1 - sigma(z_y) + eps)  averaged over fg proposals.
        """
        C = _get_num_classes(pred)
        mask = _fg_mask(target, C)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        pred_fg = pred[mask]                # (N_fg, C)
        tgt_fg  = target[mask].long()      # (N_fg,)
        N_fg = pred_fg.shape[0]

        p_y = pred_fg.sigmoid()[torch.arange(N_fg, device=pred.device), tgt_fg]
        return -torch.log(1.0 - p_y + self.eps).mean()

    def _loss_uniform(self, pred, target):
        """
        Uniform confusion loss.

        Target: correct class → 0, all C-1 wrong classes → 1/(C-1).
        Loss:   BCE(sigmoid(pred), uniform_target) over foreground proposals.
        """
        C = _get_num_classes(pred)
        mask = _fg_mask(target, C)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        pred_fg = pred[mask]            # (N_fg, C)
        tgt_fg  = target[mask].long()  # (N_fg,)
        N_fg = pred_fg.shape[0]

        uniform_val = 1.0 / max(C - 1, 1)
        uniform_tgt = pred_fg.new_full((N_fg, C), uniform_val)
        uniform_tgt[torch.arange(N_fg, device=pred.device), tgt_fg] = 0.0

        return F.binary_cross_entropy_with_logits(pred_fg, uniform_tgt)

    def _loss_margin(self, pred, target):
        """
        Margin confusion loss.

        L_margin = mean relu(z_y − z_wrong_max + margin)
        Minimising pushes the hardest wrong-class logit above the correct one.
        """
        C = _get_num_classes(pred)
        mask = _fg_mask(target, C)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        pred_fg = pred[mask]            # (N_fg, C)
        tgt_fg  = target[mask].long()  # (N_fg,)
        N_fg = pred_fg.shape[0]

        z_y = pred_fg[torch.arange(N_fg, device=pred.device), tgt_fg]  # (N_fg,)

        # Mask out the correct class to find the best wrong-class logit.
        neg_inf = pred_fg.new_full(pred_fg.shape, float('-inf'))
        wrong_mask = torch.ones_like(pred_fg, dtype=torch.bool)
        wrong_mask[torch.arange(N_fg, device=pred.device), tgt_fg] = False
        pred_wrong = torch.where(wrong_mask, pred_fg, neg_inf)
        z_wrong = pred_wrong.max(dim=1).values  # (N_fg,)

        return F.relu(z_y - z_wrong + self.margin).mean()

    def _loss_hard_wrong(self, pred, target):
        """
        Hard wrong-class loss.

        Find argmax of wrong-class logits (non-differentiable selection),
        then minimise −log(sigma(z_wrong_idx)) to boost its probability.
        """
        C = _get_num_classes(pred)
        mask = _fg_mask(target, C)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        pred_fg = pred[mask]            # (N_fg, C)
        tgt_fg  = target[mask].long()  # (N_fg,)
        N_fg = pred_fg.shape[0]

        # Non-differentiable index selection (intentional — grad flows through value)
        neg_inf = pred_fg.new_full(pred_fg.shape, float('-inf'))
        wrong_mask = torch.ones_like(pred_fg, dtype=torch.bool)
        wrong_mask[torch.arange(N_fg, device=pred.device), tgt_fg] = False
        pred_wrong = torch.where(wrong_mask, pred_fg, neg_inf)
        wrong_idx = pred_wrong.max(dim=1).indices.detach()  # (N_fg,)

        p_wrong = pred_fg.sigmoid()[
            torch.arange(N_fg, device=pred.device), wrong_idx
        ]
        return -torch.log(p_wrong + self.eps).mean()

    # ─── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """
        Args:
            pred (Tensor): Raw logits, shape (N, C).
            target (Tensor): Integer class indices, shape (N,).
                Foreground ∈ {0…C-1}, background = C.
            weight (Tensor, optional): Per-proposal weights, shape (N,).
            avg_factor (float, optional): Normalisation denominator.
            reduction_override (str, optional): Override reduction method.

        Returns:
            Tensor: Scalar combined loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        zero = pred.new_zeros(1).squeeze()
        components = {}

        # ── Component 1: Original FocalLoss ──────────────────────────────
        if self.use_original_focal:
            components['original'] = self._loss_original(
                pred, target, weight, reduction, avg_factor
            )
        else:
            components['original'] = zero

        # ── Component 2: Reverse FocalLoss ───────────────────────────────
        if self.use_reverse_focal:
            components['reverse'] = self._loss_reverse(
                pred, target, weight, reduction, avg_factor
            )
        else:
            components['reverse'] = zero

        # ── Component 3: Complement loss ─────────────────────────────────
        if self.use_complement_loss:
            components['complement'] = self._loss_complement(pred, target)
        else:
            components['complement'] = zero

        # ── Component 4: Uniform confusion ───────────────────────────────
        if self.use_uniform_confusion:
            components['uniform'] = self._loss_uniform(pred, target)
        else:
            components['uniform'] = zero

        # ── Component 5: Margin confusion ────────────────────────────────
        if self.use_margin_confusion:
            components['margin'] = self._loss_margin(pred, target)
        else:
            components['margin'] = zero

        # ── Component 6: Hard wrong-class ────────────────────────────────
        if self.use_hard_wrong_class:
            components['hard_wrong'] = self._loss_hard_wrong(pred, target)
        else:
            components['hard_wrong'] = zero

        # ── Combine ───────────────────────────────────────────────────────
        total = (
            self.lambda_original   * components['original']   +
            self.lambda_reverse    * components['reverse']    +
            self.lambda_complement * components['complement'] +
            self.lambda_uniform    * components['uniform']    +
            self.lambda_margin     * components['margin']     +
            self.lambda_hard_wrong * components['hard_wrong']
        )

        self.last_components = {k: v.detach() for k, v in components.items()}

        if self.debug:
            print(
                f'[ClassAttackLoss] '
                f'orig={components["original"].item():.4f} '
                f'rev={components["reverse"].item():.4f} '
                f'comp={components["complement"].item():.4f} '
                f'unif={components["uniform"].item():.4f} '
                f'margin={components["margin"].item():.4f} '
                f'hardwrong={components["hard_wrong"].item():.4f} '
                f'total={total.item():.4f}'
            )

        return total
