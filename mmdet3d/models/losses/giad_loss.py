"""
GIAD Loss – Glare-Induced Attention Diffusion Loss

A modular research loss for exploring glare-inspired heatmap attack objectives.
Compatible with the GaussianFocalLoss call signature so it is a drop-in
replacement for loss_heatmap in TransFusionHeadV2 configs.

Total loss
----------
  L = lambda_original * L_original
    + lambda_reverse * L_original
    + lambda_ring      * L_ring
    + lambda_diffusion * L_diffusion
    + lambda_entropy   * L_entropy
    + lambda_contrast  * L_contrast
    + lambda_veiling   * L_veiling

Each term is gated by its use_* flag; disabled terms contribute exactly zero.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import LOSSES
from mmdet.models.losses.gaussian_focal_loss import gaussian_focal_loss
from mmdet.models.losses.utils import weight_reduce_loss


# ─── kernel builders (pure torch, no external deps) ──────────────────────────

def _gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """Return a (1,1,size,size) normalised 2-D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)   # (size, size)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)     # (1, 1, size, size)


def _ring_kernel_2d(ring_radius: float, ring_sigma: float) -> torch.Tensor:
    """Return a (1,1,K,K) ring kernel peaked at radial distance ring_radius."""
    half = int(ring_radius + 4.0 * ring_sigma) + 1
    size = 2 * half + 1
    coords = torch.arange(size, dtype=torch.float32) - half
    yy, xx = torch.meshgrid(coords, coords)   # (size, size)
    r = torch.sqrt(xx ** 2 + yy ** 2)
    kernel = torch.exp(-((r - ring_radius) ** 2) / (2.0 * ring_sigma ** 2))
    return kernel.unsqueeze(0).unsqueeze(0)    # (1, 1, size, size)


def _veiling_kernel_2d(power: float, epsilon: float,
                       size: int = 51) -> torch.Tensor:
    """Return a (1,1,size,size) inverse-power-law spread kernel."""
    size = size | 1      # ensure odd
    half = size // 2
    coords = torch.arange(size, dtype=torch.float32) - half
    yy, xx = torch.meshgrid(coords, coords)
    r = torch.sqrt(xx ** 2 + yy ** 2)
    kernel = 1.0 / (r + epsilon) ** power
    return kernel.unsqueeze(0).unsqueeze(0)    # (1, 1, size, size)


# ─── helpers ─────────────────────────────────────────────────────────────────

def build_ring_target(peaks: torch.Tensor,
                      ring_kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolve a binary peak map with a ring kernel to get the ring target.

    Args:
        peaks: (B, C, H, W) float tensor, 1 where target==1, else 0.
        ring_kernel: (1, 1, K, K) ring kernel.

    Returns:
        ring_target: (B, C, H, W), values in [0, 1].
    """
    B, C, H, W = peaks.shape
    K = ring_kernel.shape[-1]
    pad = K // 2
    # Process all channels together by reshaping to (B*C, 1, H, W)
    p = peaks.view(B * C, 1, H, W)
    k = ring_kernel.to(p.device, p.dtype)
    out = F.conv2d(p, k, padding=pad)            # (B*C, 1, H, W)
    out = out.view(B, C, H, W)
    # Normalise to [0, 1]
    out_max = out.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    return (out / out_max).clamp(0.0, 1.0)


def build_veiling_target(peaks: torch.Tensor,
                         veiling_kernel: torch.Tensor) -> torch.Tensor:
    """
    Build a veiling-luminance target by spreading peaks with an
    inverse-power-law kernel.

    Args:
        peaks: (B, C, H, W) float tensor.
        veiling_kernel: (1, 1, K, K).

    Returns:
        veiling_target: (B, C, H, W), values in [0, 1].
    """
    B, C, H, W = peaks.shape
    K = veiling_kernel.shape[-1]
    pad = K // 2
    p = peaks.view(B * C, 1, H, W)
    k = veiling_kernel.to(p.device, p.dtype)
    out = F.conv2d(p, k, padding=pad)
    out = out.view(B, C, H, W)
    out_max = out.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
    return (out / out_max).clamp(0.0, 1.0)


def diffuse_heatmap(pred: torch.Tensor,
                    diff_kernel: torch.Tensor,
                    iterations: int) -> torch.Tensor:
    """
    Apply repeated Gaussian blur to produce a diffused version of pred.

    Args:
        pred: (B, C, H, W)
        diff_kernel: (1, 1, K, K) Gaussian kernel.
        iterations: number of smoothing passes.

    Returns:
        diffused: (B, C, H, W)
    """
    B, C, H, W = pred.shape
    K = diff_kernel.shape[-1]
    pad = K // 2
    x = pred.view(B * C, 1, H, W)
    k = diff_kernel.to(x.device, x.dtype)
    for _ in range(iterations):
        x = F.conv2d(x, k, padding=pad)
    return x.view(B, C, H, W)


def compute_local_contrast(pred: torch.Tensor,
                           window: int) -> torch.Tensor:
    """
    Local max-min contrast of pred within a sliding window.

    Args:
        pred: (B, C, H, W), values in [0, 1].
        window: neighbourhood size (must be odd).

    Returns:
        contrast: (B, C, H, W)
    """
    pad = window // 2
    local_max = F.max_pool2d(pred, kernel_size=window,
                              stride=1, padding=pad)
    local_min = -F.max_pool2d(-pred, kernel_size=window,
                               stride=1, padding=pad)
    return local_max - local_min


# ─── main loss module ─────────────────────────────────────────────────────────

@LOSSES.register_module()
class GIADLoss(nn.Module):
    """Glare-Induced Attention Diffusion Loss.

    Drop-in replacement for GaussianFocalLoss as the heatmap loss in
    TransFusionHeadV2. All component sub-losses are disabled by default
    except ``use_original_gaussian_loss=True`` which preserves baseline
    behaviour exactly.

    After each ``forward()`` call, individual component values are stored in
    ``self.last_components`` (a dict of detached scalars) so that the head or
    training loop can log or further weight them.

    Args:
        alpha, gamma: GaussianFocalLoss parameters.
        reduction, loss_weight: standard mmdet loss conventions.
        use_*: enable / disable each component.
        lambda_*: per-component scale factor (applied before summing).
        ring_radius, ring_sigma: ring kernel parameters (in heatmap pixels).
        diffusion_iterations, diffusion_kernel_size: diffusion parameters.
        contrast_window: sliding-window size for contrast computation.
        veiling_power, veiling_epsilon: veiling kernel parameters.
        debug: if True, print component values each forward pass.
    """

    def __init__(
        self,
        # ── GaussianFocalLoss passthrough ────────────────────────────────
        alpha: float = 2.0,
        gamma: float = 4.0,
        reduction: str = 'mean',
        loss_weight: float = 1.0,
        # ── component flags ──────────────────────────────────────────────
        use_original_gaussian_loss: bool = True,
        use_reverse_gaussian_loss: bool = False,
        use_ring_loss: bool = False,
        use_attention_diffusion: bool = False,
        use_entropy_loss: bool = False,
        use_contrast_loss: bool = False,
        use_veiling_luminance: bool = False,
        # ── lambda weights ───────────────────────────────────────────────
        lambda_original: float = 1.0,
        lambda_reverse: float = 1.0,
        lambda_ring: float = 1.0,
        lambda_diffusion: float = 1.0,
        lambda_entropy: float = 1.0,
        lambda_contrast: float = 1.0,
        lambda_veiling: float = 1.0,
        # ── ring parameters ──────────────────────────────────────────────
        ring_radius: float = 3.0,
        ring_sigma: float = 1.0,
        # ── diffusion parameters ─────────────────────────────────────────
        diffusion_iterations: int = 3,
        diffusion_kernel_size: int = 5,
        # ── contrast parameters ──────────────────────────────────────────
        contrast_window: int = 11,
        # ── veiling parameters ───────────────────────────────────────────
        veiling_power: float = 2.0,
        veiling_epsilon: float = 1e-6,
        # ── misc ─────────────────────────────────────────────────────────
        debug: bool = False,
    ):
        super(GIADLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.use_original_gaussian_loss = use_original_gaussian_loss
        self.use_reverse_gaussian_loss = use_reverse_gaussian_loss
        self.use_ring_loss = use_ring_loss
        self.use_attention_diffusion = use_attention_diffusion
        self.use_entropy_loss = use_entropy_loss
        self.use_contrast_loss = use_contrast_loss
        self.use_veiling_luminance = use_veiling_luminance

        self.lambda_original = lambda_original
        self.lambda_reverse = lambda_reverse
        self.lambda_ring = lambda_ring
        self.lambda_diffusion = lambda_diffusion
        self.lambda_entropy = lambda_entropy
        self.lambda_contrast = lambda_contrast
        self.lambda_veiling = lambda_veiling

        self.diffusion_iterations = diffusion_iterations
        self.contrast_window = contrast_window | 1   # ensure odd
        self.veiling_power = veiling_power
        self.veiling_epsilon = veiling_epsilon

        self.debug = debug

        # Pre-build fixed kernels and register as buffers so they move
        # automatically with .cuda() / .to(device).
        if use_ring_loss:
            k = _ring_kernel_2d(ring_radius, ring_sigma)
            self.register_buffer('ring_kernel', k)
        else:
            self.ring_kernel = None

        if use_attention_diffusion:
            # Ensure kernel size is odd
            ks = diffusion_kernel_size | 1
            sigma_d = ks / 6.0   # rule-of-thumb: sigma ≈ size/6
            k = _gaussian_kernel_2d(ks, sigma_d)
            self.register_buffer('diff_kernel', k)
        else:
            self.diff_kernel = None

        if use_veiling_luminance:
            # Kernel size: span ~5 * ring_radius or at least 51 pixels
            veiling_size = max(51, int(ring_radius * 10 + 1)) | 1
            k = _veiling_kernel_2d(veiling_power, veiling_epsilon,
                                   size=veiling_size)
            self.register_buffer('veiling_kernel', k)
        else:
            self.veiling_kernel = None

        # After each forward, individual losses are stored here (detached)
        # for external logging without breaking the graph.
        self.last_components: dict = {}

    # ─── component methods ────────────────────────────────────────────────────

    def _loss_original(self, pred, target, weight, avg_factor,
                       reduction_override):
        """Standard GaussianFocalLoss."""
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * gaussian_focal_loss(
            pred, target, weight,
            alpha=self.alpha, gamma=self.gamma,
            reduction=reduction, avg_factor=avg_factor,
        )

    def _loss_ring(self, pred, target):
        """Ring-shaped target loss: push attention to a halo around centres."""
        peaks = (target == 1).float()
        ring_target = build_ring_target(peaks, self.ring_kernel)
        return F.mse_loss(pred, ring_target)

    def _loss_diffusion(self, pred):
        """Attention diffusion loss: push pred to resemble its blurred self."""
        with torch.no_grad():
            diffused = diffuse_heatmap(pred.detach(), self.diff_kernel,
                                       self.diffusion_iterations)
        return F.mse_loss(pred, diffused)

    def _loss_entropy(self, pred):
        """Entropy maximisation loss: spread attention uniformly."""
        eps = 1e-8
        # Normalise spatially per (batch, class) channel
        flat = pred.view(pred.shape[0], pred.shape[1], -1)     # (B, C, HW)
        p = flat / (flat.sum(dim=-1, keepdim=True) + eps)
        H = -(p * (p + eps).log()).sum(dim=-1)                  # (B, C)
        return -H.mean()   # minimise → maximise entropy

    def _loss_contrast(self, pred):
        """Contrast reduction loss: discourage local high-contrast responses."""
        contrast = compute_local_contrast(pred, self.contrast_window)
        return contrast.mean()

    def _loss_veiling(self, pred, target):
        """Veiling luminance loss: spread responses with a 1/r^power kernel."""
        peaks = (target == 1).float()
        veiling_target = build_veiling_target(peaks, self.veiling_kernel)
        return F.mse_loss(pred, veiling_target)

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
            pred (Tensor): Clipped-sigmoid heatmap prediction, shape
                (B, num_classes, H, W), values in (0, 1).
            target (Tensor): Gaussian heatmap ground truth, same shape.
                Values are 1 at object centres and decay toward 0.
            weight (Tensor, optional): Per-element weight.
            avg_factor (float, optional): Normalisation factor.
            reduction_override (str, optional): Override reduction method.

        Returns:
            Tensor: Scalar total loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        zero = pred.new_zeros(1).squeeze()
        components = {}

        # ── Component 1: Original GaussianFocalLoss ──────────────────────
        if self.use_original_gaussian_loss:
            l = self._loss_original(pred, target, weight,
                                    avg_factor, reduction_override)
            components['original'] = l
        else:
            components['original'] = zero

        # ── Component 1: Original GaussianFocalLoss ──────────────────────
        if self.use_reverse_gaussian_loss:
            l = self._loss_original(pred, target, weight,
                                    avg_factor, reduction_override)
            components['reverse'] = -1*l
        else:
            components['reverse'] = zero

        # ── Component 2: Ring loss ────────────────────────────────────────
        if self.use_ring_loss:
            l = self._loss_ring(pred, target)
            components['ring'] = l
        else:
            components['ring'] = zero

        # ── Component 3: Attention diffusion ─────────────────────────────
        if self.use_attention_diffusion:
            l = self._loss_diffusion(pred)
            components['diffusion'] = l
        else:
            components['diffusion'] = zero

        # ── Component 4: Entropy maximisation ────────────────────────────
        if self.use_entropy_loss:
            l = self._loss_entropy(pred)
            components['entropy'] = l
        else:
            components['entropy'] = zero

        # ── Component 5: Contrast reduction ──────────────────────────────
        if self.use_contrast_loss:
            l = self._loss_contrast(pred)
            components['contrast'] = l
        else:
            components['contrast'] = zero

        # ── Component 6: Veiling luminance ───────────────────────────────
        if self.use_veiling_luminance:
            l = self._loss_veiling(pred, target)
            components['veiling'] = l
        else:
            components['veiling'] = zero

        # ── Combine ───────────────────────────────────────────────────────
        total = (
            self.lambda_original  * components['original']  +
            self.lambda_reverse  * components['reverse']  +
            self.lambda_ring      * components['ring']       +
            self.lambda_diffusion * components['diffusion']  +
            self.lambda_entropy   * components['entropy']    +
            self.lambda_contrast  * components['contrast']   +
            self.lambda_veiling   * components['veiling']
        )

        # Store detached values for external logging (no graph retained)
        self.last_components = {k: v.detach() for k, v in components.items()}

        if self.debug:
            print(
                f'[GIADLoss] '
                f'orig={components["original"].item():.4f} '
                f'ring={components["ring"].item():.4f} '
                f'diff={components["diffusion"].item():.4f} '
                f'entr={components["entropy"].item():.4f} '
                f'cont={components["contrast"].item():.4f} '
                f'veil={components["veiling"].item():.4f} '
                f'total={total.item():.4f}'
            )

        return total
