from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable

__all__ = ["DSSIMLoss"]


class DSSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image, gt_image):
        # size: B V 3 H W
        pred = rearrange(image, "B V C H W -> (B V) C H W")
        target = rearrange(gt_image, "B V C H W -> (B V) C H W")
        return 1.0 - ssim(pred, target)


def gaussian(window_size, sigma):
    """Return a 1-D Gaussian kernel of length `window_size` with std `sigma`."""
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Create a 2-D Gaussian smoothing kernel replicated across `channel` channels.

    The 2-D kernel is the outer product of two identical 1-D Gaussians
    (sigma=1.5), giving a separable isotropic filter.  It is used as a
    depthwise convolution kernel (groups=channel) so each channel is
    smoothed independently.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    SSIM formula:
        SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2)
                    / (mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)

    where mu and sigma are local statistics estimated with a Gaussian window.
    C1 = (0.01*L)^2, C2 = (0.03*L)^2 (L=1.0 for [0,1]-scaled images) are
    small stabilization constants that prevent division by near-zero variances.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Local variance: E[X^2] - E[X]^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    # Local covariance: E[XY] - E[X]*E[Y]
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # Stabilization constants for L=1 (images in [0,1])
    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


if __name__ == "__main__":
    images = torch.rand(16, 4, 3, 128, 128)  # Batch of 3 RGB images
    gt_images = torch.rand(16, 4, 3, 128, 128)  # Ground truth images

    loss_fn = DSSIMLoss()
    loss = loss_fn(images, gt_images)
    print(loss)  # Prints the DSSIM loss value
