import math
import random

import torch
from torch.nn import functional as F


class MixUP:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    @torch.inference_mode()
    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        batch = batch * (1-ratio) + batch_roll * ratio
        target = target * (1-ratio) + target_roll * ratio

        return batch, target


class CutMix:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    @torch.inference_mode()
    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        B, C, H, W = batch.shape
        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        height_half = int(0.5 * math.sqrt(ratio) * H)
        width_half = int(0.5 * math.sqrt(ratio) * W)
        r = int(random.random() * H)
        c = int(random.random() * W)

        start_x = max(r - height_half, 0)
        end_x = min(r + height_half, H)
        start_y = max(c - width_half, 0)
        end_y = min(c + width_half, W)

        ratio = (end_x - start_x) * (end_y - start_y) / (H * W)

        batch[:, :, start_x:end_x, start_y:end_y] = batch_roll[:, :, start_x:end_x, start_y:end_y]
        target = target * (1-ratio) + target_roll * ratio

        return batch, target