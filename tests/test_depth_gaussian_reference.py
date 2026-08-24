"""Depth preprocessing parity with InstinctMJ's torchvision pipeline."""

import torch
import torch.nn.functional as F

from instinctlab.mdp.observations import _gaussian_blur


def test_depth_gaussian_blur_uses_instinctmj_reflect_padding() -> None:
    """The reference torchvision GaussianBlur reflects image edges, rather than repeating them."""
    image = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    coords = torch.arange(3, dtype=image.dtype) - 1
    gauss = torch.exp(-0.5 * coords.square())
    gauss /= gauss.sum()
    kernel = (gauss[:, None] * gauss[None, :]).view(1, 1, 3, 3)
    reflected = F.conv2d(F.pad(image.unsqueeze(1), (1, 1, 1, 1), mode="reflect"), kernel).squeeze(1)
    replicated = F.conv2d(F.pad(image.unsqueeze(1), (1, 1, 1, 1), mode="replicate"), kernel).squeeze(1)

    actual = _gaussian_blur(image, kernel_size=3, sigma=1.0)

    assert torch.allclose(actual, reflected)
    assert not torch.allclose(actual, replicated)
