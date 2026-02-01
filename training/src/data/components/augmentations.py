import numpy as np
import torch


class SaltAndPepperNoise:
    """Applies Salt and Pepper noise to an image."""

    def __init__(self, p=0.02):
        """
        Args:
            p (float): Probability of a pixel being affected by noise.
        """
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor (C, H, W) in range [0, 1].
        Returns:
            Tensor: Noisy image.
        """
        np_img = img.numpy()
        rand_mask = np.random.rand(*np_img.shape)  # Generate random mask
        np_img[rand_mask < self.p / 2] = 0  # Salt (black)
        np_img[rand_mask > 1 - self.p / 2] = 1  # Pepper (white)
        return torch.tensor(np_img)

    def __repr__(self):
        return f"SaltAndPepperNoise(p={self.p})"


class GaussianNoise:
    """Applies Gaussian noise to an image."""

    def __init__(self, mean=0.0, std=0.05):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
        """
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image tensor (C, H, W) in range [0, 1].
        Returns:
            Tensor: Noisy image.
        """
        noise = torch.randn(img.shape) * self.std + self.mean
        return torch.clamp(img + noise, 0, 1)

    def __repr__(self):
        return f"GaussianNoise(mean={self.mean}, std={self.std})"
