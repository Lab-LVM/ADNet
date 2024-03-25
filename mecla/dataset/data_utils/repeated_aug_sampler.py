import math

import torch
from torch import distributed as dist
from torch.utils.data import Sampler


class RepeatAugSampler(Sampler):
    """RASampler ported from timm.
    This is introduced in https://arxiv.org/pdf/1902.05509.pdf.
    Repeated Sampler only samples subset of train dataset.
    Subset of train dataset is duplicated with strong augmentation.
    By learning differently augmented image, we can regularize model with large batch size.
    This technique is very important for larger batch size, but decrease performance for smaller batch size.
    Original description:
    Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler
    This sampler was taken from https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    Used in
    Copyright (c) 2015-present, Facebook, Inc.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            num_repeats=3,
            selected_round=256,
            selected_ratio=0,
    ):
        if num_replicas is None:
            num_replicas = 1
            # if not dist.is_available():
            #     raise RuntimeError("Requires distributed package to be available")
            # num_replicas = dist.get_world_size()
        if rank is None:
            rank = 0
            # if not dist.is_available():
            #     raise RuntimeError("Requires distributed package to be available")
            # rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # Determine the number of samples to select per epoch for each rank.
        # num_selected logic defaults to be the same as original RASampler impl, but this one can be tweaked
        # via selected_ratio and selected_round args.
        selected_ratio = selected_ratio or num_replicas  # ratio to reduce selected samples by, num_replicas if 0
        if selected_round:
            self.num_selected_samples = int(math.floor(
                 len(self.dataset) // selected_round * selected_round / selected_ratio))
        else:
            self.num_selected_samples = int(math.ceil(len(self.dataset) / selected_ratio))

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # produce repeats e.g. [0, 0, 0, 1, 1, 1, 2, 2, 2....]
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample per rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # return up to num selected samples
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch