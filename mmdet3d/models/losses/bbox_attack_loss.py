"""
BBoxAttackLoss — adversarial bounding-box regression loss framework.

Drop-in replacement for L1Loss in TransFusionHeadV2.
All attack terms are disabled by default so the baseline is reproduced
exactly when only use_original_l1=True.

Tensor conventions (from TransFusionHeadV2 call site)
-------------------------------------------------------
  pred   : (BS, num_proposals, code_size)  encoded predictions
  target : (BS, num_proposals, code_size)  encoded GT targets (0 for BG)
  weight : (BS, num_proposals, code_size)  per-element weights
               — weight = 0 for background proposals (all code dims)
               — weight = code_weight for foreground proposals
  avg_factor: int   max(num_pos, 1)

Encoding layout (code_size=10, from TransFusionBBoxCoder.encode)
----------------------------------------------------------------
  [0]  x_bev   — (x - pc_range[0]) / (out_size_factor * voxel_size[0])
  [1]  y_bev   — (y - pc_range[1]) / (out_size_factor * voxel_size[1])
  [2]  z_grav  — z_bottom + h/2  (gravity centre, metres)
  [3]  log(w)  — log width
  [4]  log(l)  — log length
  [5]  log(h)  — log height
  [6]  sin(yaw)
  [7]  cos(yaw)
  [8]  vx      (code_weight = 0.2)
  [9]  vy      (code_weight = 0.2)

Component slices
----------------
  CENTRE  = [:3]   x_bev, y_bev, z_grav
  DIMS    = [3:6]  log(w), log(l), log(h)
  ROT     = [6:8]  sin(yaw), cos(yaw)
  VEL     = [8:]   vx, vy

Total loss
----------
  L = lambda_original    * L_original
    + lambda_reverse     * L_reverse
    + lambda_translation * L_translation
    + lambda_orbit       * L_orbit
    + lambda_scale       * L_scale
    + lambda_orientation * L_orientation

Inactive terms contribute exactly zero and carry no gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.smooth_l1_loss import l1_loss
from mmdet.models.losses.utils import weight_reduce_loss

# ── component index constants ─────────────────────────────────────────────────
_CENTRE_IDX  = slice(0, 3)   # x_bev, y_bev, z_grav
_DIMS_IDX    = slice(3, 6)   # log(w), log(l), log(h)
_ROT_IDX     = slice(6, 8)   # sin(yaw), cos(yaw)


def _fg_mask(weight: torch.Tensor) -> torch.Tensor:
    """(BS, num_proposals) bool — True where proposal is foreground.

    Background proposals have weight==0 in every code dimension.
    We use dim-0 (x_bev) as the indicator since its code_weight is 1.0.
    """
    return weight[..., 0] > 0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of values over foreground positions; returns 0 if no foreground."""
    n = mask.sum().clamp(min=1)
    return values[mask].sum() / n


# ─── main loss module ─────────────────────────────────────────────────────────

@LOSSES.register_module()
class BBoxAttackLoss(nn.Module):
    """Adversarial bounding-box regression loss framework for IS-Fusion.

    Drop-in replacement for L1Loss as the bbox regression loss in
    TransFusionHeadV2.  All attack components are disabled by default so the
    module is a zero-change baseline when only ``use_original_l1=True``.

    After each ``forward()`` call, individual component values are stored in
    ``self.last_components`` (detached) for external logging.

    Args:
        reduction, loss_weight: standard mmdet conventions.
        use_*: enable / disable each component.
        lambda_*: per-component scale applied before summing.
        orbit_radius: displacement radius in encoded BEV units for the orbit
            attack.  One encoded unit ≈ out_size_factor × voxel_size metres
            (≈ 0.6 m for this repo's default settings).
        debug: if True, print component values each forward pass.
    """

    def __init__(
        self,
        # ── L1Loss passthrough ───────────────────────────────────────────
        reduction: str = 'mean',
        loss_weight: float = 0.25,
        # ── component flags ──────────────────────────────────────────────
        use_original_l1: bool = True,
        use_reverse_l1: bool = False,
        use_translation_attack: bool = False,
        use_orbit_attack: bool = False,
        use_scale_attack: bool = False,
        use_orientation_attack: bool = False,
        # ── lambda weights ───────────────────────────────────────────────
        lambda_original: float = 1.0,
        lambda_reverse: float = 1.0,
        lambda_translation: float = 1.0,
        lambda_orbit: float = 1.0,
        lambda_scale: float = 1.0,
        lambda_orientation: float = 1.0,
        # ── tuning ───────────────────────────────────────────────────────
        orbit_radius: float = 2.0,
        # ── misc ─────────────────────────────────────────────────────────
        debug: bool = False,
    ):
        super(BBoxAttackLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight

        self.use_original_l1 = use_original_l1
        self.use_reverse_l1 = use_reverse_l1
        self.use_translation_attack = use_translation_attack
        self.use_orbit_attack = use_orbit_attack
        self.use_scale_attack = use_scale_attack
        self.use_orientation_attack = use_orientation_attack

        self.lambda_original = lambda_original
        self.lambda_reverse = lambda_reverse
        self.lambda_translation = lambda_translation
        self.lambda_orbit = lambda_orbit
        self.lambda_scale = lambda_scale
        self.lambda_orientation = lambda_orientation

        self.orbit_radius = orbit_radius
        self.debug = debug

        self.last_components: dict = {}

    # ─── component methods ────────────────────────────────────────────────────

    def _loss_original(self, pred, target, weight, reduction, avg_factor):
        """Standard L1 regression loss — identical to existing L1Loss."""
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )

    def _loss_reverse(self, pred, target, weight, reduction, avg_factor):
        """Negative L1 loss — directly maximises regression error."""
        return -self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )

    def _loss_translation(self, pred, target, weight):
        """
        Translation attack — maximise BEV+Z centre displacement.

        Operates on encoded centre slice [:3]: x_bev, y_bev, z_grav.
        Uses L1 distance over centre dims, negated to maximise displacement.
        Only foreground proposals (weight[...,0] > 0) contribute.
        """
        mask = _fg_mask(weight)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        centre_pred = pred[..., _CENTRE_IDX][mask]     # (N_fg, 3)
        centre_gt   = target[..., _CENTRE_IDX][mask]   # (N_fg, 3)
        # L1 distance per-proposal, averaged over foreground
        dist = (centre_pred - centre_gt).abs().sum(dim=-1)   # (N_fg,)
        return -dist.mean()   # minimise → maximise displacement

    def _loss_orbit(self, pred, target, weight):
        """
        Orbit attack — push BEV centre onto a random orbit around GT centre.

        For each foreground proposal we sample a uniformly random direction in
        BEV (x_bev, y_bev) and construct an orbit target at ``orbit_radius``
        encoded units away from GT.  The z_grav target is kept as GT so the
        attack focuses purely on the BEV plane (where detection matters most).

        The random direction is sampled fresh each forward pass (no gradient
        through it).  Gradient flows only through ``pred``.
        """
        mask = _fg_mask(weight)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        centre_pred = pred[..., _CENTRE_IDX][mask]     # (N_fg, 3)
        centre_gt   = target[..., _CENTRE_IDX][mask]   # (N_fg, 3)
        N_fg = centre_pred.shape[0]

        # Random unit vector in BEV (x, y); z unchanged.
        angle = torch.rand(N_fg, device=pred.device) * 2.0 * 3.141592653589793
        bev_dir = torch.stack([angle.cos(), angle.sin()], dim=-1)  # (N_fg, 2)
        z_col   = torch.zeros(N_fg, 1, device=pred.device)
        direction = torch.cat([bev_dir, z_col], dim=-1)             # (N_fg, 3)

        orbit_target = (centre_gt + self.orbit_radius * direction).detach()
        return F.l1_loss(centre_pred, orbit_target)

    def _loss_scale(self, pred, target, weight):
        """
        Scale attack — maximise dimension error on log(w), log(l), log(h).

        Operates on encoded dims slice [3:6].
        Negated L1 so the model is pushed away from the GT log-scale.
        Only foreground proposals contribute.
        """
        mask = _fg_mask(weight)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        dims_pred = pred[..., _DIMS_IDX][mask]      # (N_fg, 3)
        dims_gt   = target[..., _DIMS_IDX][mask]    # (N_fg, 3)
        dist = (dims_pred - dims_gt).abs().sum(dim=-1)  # (N_fg,)
        return -dist.mean()   # minimise → maximise scale error

    def _loss_orientation(self, pred, target, weight):
        """
        Orientation attack — maximise angular error.

        The encoding is (sin(yaw), cos(yaw)) at indices [6:8].

        cos(yaw_pred − yaw_gt) = sin_p*sin_g + cos_p*cos_g

        Minimising this dot-product drives the predicted angle as far from GT
        as possible (target: cos = −1, i.e. 180° error).
        Only foreground proposals contribute.
        """
        mask = _fg_mask(weight)
        if mask.sum() == 0:
            return pred.new_zeros(1).squeeze()
        rot_pred = pred[..., _ROT_IDX][mask]     # (N_fg, 2)  [sin, cos]
        rot_gt   = target[..., _ROT_IDX][mask]   # (N_fg, 2)

        # cos(delta_yaw) = sin_p*sin_g + cos_p*cos_g
        cos_delta = (rot_pred * rot_gt).sum(dim=-1)   # (N_fg,)
        return cos_delta.mean()   # minimise → maximise angular error

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
            pred (Tensor): Encoded bbox predictions, shape (BS, num_proposals, 10).
            target (Tensor): Encoded GT targets, same shape.
                Background proposals have target == 0 in all dims.
            weight (Tensor, optional): Per-element weights, same shape.
                Background proposals have weight == 0.
            avg_factor (float, optional): Normalisation denominator (num_pos).
            reduction_override (str, optional): Override reduction method.

        Returns:
            Tensor: Scalar combined loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction

        zero = pred.new_zeros(1).squeeze()
        components = {}

        # ── Component 1: Original L1 ──────────────────────────────────────
        if self.use_original_l1:
            components['original'] = self._loss_original(
                pred, target, weight, reduction, avg_factor
            )
        else:
            components['original'] = zero

        # ── Component 2: Reverse L1 ──────────────────────────────────────
        if self.use_reverse_l1:
            components['reverse'] = self._loss_reverse(
                pred, target, weight, reduction, avg_factor
            )
        else:
            components['reverse'] = zero

        # ── Component 3: Translation attack ──────────────────────────────
        if self.use_translation_attack:
            components['translation'] = self._loss_translation(
                pred, target, weight
            )
        else:
            components['translation'] = zero

        # ── Component 4: Orbit attack ─────────────────────────────────────
        if self.use_orbit_attack:
            components['orbit'] = self._loss_orbit(pred, target, weight)
        else:
            components['orbit'] = zero

        # ── Component 5: Scale attack ─────────────────────────────────────
        if self.use_scale_attack:
            components['scale'] = self._loss_scale(pred, target, weight)
        else:
            components['scale'] = zero

        # ── Component 6: Orientation attack ───────────────────────────────
        if self.use_orientation_attack:
            components['orientation'] = self._loss_orientation(
                pred, target, weight
            )
        else:
            components['orientation'] = zero

        # ── Combine ───────────────────────────────────────────────────────
        total = (
            self.lambda_original    * components['original']    +
            self.lambda_reverse     * components['reverse']     +
            self.lambda_translation * components['translation'] +
            self.lambda_orbit       * components['orbit']       +
            self.lambda_scale       * components['scale']       +
            self.lambda_orientation * components['orientation']
        )

        self.last_components = {k: v.detach() for k, v in components.items()}

        if self.debug:
            print(
                f'[BBoxAttackLoss] '
                f'orig={components["original"].item():.4f} '
                f'rev={components["reverse"].item():.4f} '
                f'trans={components["translation"].item():.4f} '
                f'orbit={components["orbit"].item():.4f} '
                f'scale={components["scale"].item():.4f} '
                f'orient={components["orientation"].item():.4f} '
                f'total={total.item():.4f}'
            )

        return total
