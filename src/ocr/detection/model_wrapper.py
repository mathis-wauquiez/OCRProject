import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import replace

from .craft.craft import CRAFT

from ... import utils
from ..params import craftParams


class craftWrapper(nn.Module):

    def __init__(self, params=craftParams(), **kwargs):
        super().__init__()

        params = replace(params, **kwargs)

        self.model = CRAFT()
        self.params = params
        self.model.load_state_dict(utils.copyStateDict(torch.load(params.chckpt)))

    def _scaling_ratio(self, H, W):
        """Ratio used to map between original and preprocessed coordinates."""
        return min(self.params.mag_ratio * max(H, W), self.params.canvas_size) / max(H, W)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor with 0-255 range
        Returns:
            preprocessed, score_text, score_link
        """
        B, C, H, W = x.shape

        ratio = self._scaling_ratio(H, W)
        target_h, target_w = int(H * ratio), int(W * ratio)
        target_h_32, target_w_32 = utils.nearest_32(target_h), utils.nearest_32(target_w)

        processed = torch.zeros((B, C, target_h_32, target_w_32), device=x.device)
        processed[..., :target_h, :target_w] = F.interpolate(
            x, size=(target_h, target_w),
            mode=self.params.interpolation, align_corners=False,
        )

        # The authors normalize after zero-padding
        mean = self.params.mean.view(1, -1, 1, 1).to(x.device)
        std = self.params.std.view(1, -1, 1, 1).to(x.device)
        processed = (processed - mean) / std

        y, _ = self.model(processed)
        score_text = y[..., 0]
        score_link = y[..., 1]

        return processed, score_text, score_link

    def map_original_to_preprocessed(self, coords_original, original_shape):
        """Scale coordinates from original image space to preprocessed space."""
        H, W = original_shape
        return coords_original * self._scaling_ratio(H, W)

