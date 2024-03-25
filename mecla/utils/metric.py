import os

import numpy as np
import torch
from torch import distributed as dist
import torchmetrics.functional as TMF


class Metric:
    def __init__(self, reduce_every_n_step=50, reduce_on_compute=True, header='', fmt='{val:.4f} ({avg:.4f})'):
        """Base Metric Class supporting ddp setup
        :arg
            reduce_ever_n_step(int): call all_reduce every n step in ddp mode
            reduce_on_compute(bool): call all_reduce in compute() method
            fmt(str): format representing metric in string
        """
        self.dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ

        if self.dist:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.reduce_every_n_step = reduce_every_n_step
            self.reduce_on_compute = reduce_on_compute
        else:
            self.world_size = None
            self.reduce_every_n_step = self.reduce_on_compute = False

        self.val = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
        self.header = header
        self.fmt = fmt

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().clone()
        elif self.reduce_every_n_step and not isinstance(val, torch.Tensor):
            raise ValueError('reduce operation is allowed for only tensor')

        self.val = val
        self.sum += val * n
        self.n += n
        self.avg = self.sum / self.n

        if self.reduce_every_n_step and self.n % self.reduce_every_n_step == 0:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

    def compute(self):
        if self.reduce_on_compute:
            self.sum = all_reduce_mean(self.sum, self.world_size)
            self.avg = self.sum / self.n

        return self.avg

    def __str__(self):
        return self.header + ' ' + self.fmt.format(**self.__dict__)


def Accuracy(y_hat, y, top_k=(1,)):
    """Compute top-k accuracy
    :arg
        y_hat(tensor): prediction shaped as (B, C)
        y(tensor): label shaped as (B)
        top_k(tuple): how exactly model should predict in each metric
    :return
        list of metric scores
    """
    prediction = torch.argsort(y_hat, dim=-1, descending=True)
    accuracy = [(prediction[:, :min(k, y_hat.size(1))] == y.unsqueeze(-1)).float().sum(dim=-1).mean() * 100 for k in top_k]
    return accuracy


def compute_metrics(preds, labels, args):
    if isinstance(preds, (list, tuple)):
        preds = torch.concat(preds, dim=0).detach().clone()
        labels = torch.concat(labels, dim=0).detach().clone().to(torch.long)

    if args.mode != 'knn':
        preds = all_gather_with_different_size(preds)
        labels = all_gather_with_different_size(labels)

    if args.task == 'binary':
        preds = preds.squeeze(1)

    metrics = list(TMF.__dict__[m](
        preds,
        labels,
        args.task,
        num_classes=args.num_classes,
        num_labels=args.num_labels,
        average=None if args.mode == 'valid' else 'macro',
    ) for m in args.metric_names)

    return metrics


def all_reduce_mean(val, world_size):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_reduce_sum(val):
    """Collect value to each gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.all_reduce(val, dist.ReduceOp.SUM)
    return val


def reduce_mean(val, world_size):
    """Collect value to local zero gpu
    :arg
        val(tensor): target
        world_size(int): the number of process in each group
    """
    val = val.clone()
    dist.reduce(val, 0, dist.ReduceOp.SUM)
    val = val / world_size
    return val


def all_gather(x):
    """Collect value to local rank zero gpu
    :arg
        x(tensor): target
    """
    if dist.is_initialized():
        dest = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(dest, x)
        return torch.cat(dest, dim=0)
    else:
        return x


def all_gather_with_different_size(x):
    """all gather operation with different sized tensor
    :arg
        x(tensor): target
    (reference) https://stackoverflow.com/a/71433508/17670380
    """
    if dist.is_initialized():
        local_size = torch.tensor([x.size(0)], device=x.device)
        all_sizes = all_gather(local_size)
        max_size = max(all_sizes)

        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros([size_diff]+list(x.shape[1:]), device=x.device, dtype=x.dtype)
            x = torch.cat((x, padding))

        all_gathered_with_pad = all_gather(x)
        all_gathered = []
        ws = dist.get_world_size()
        for vector, size in zip(all_gathered_with_pad.chunk(ws), all_sizes.chunk(ws)):
            all_gathered.append(vector[:size])

        return torch.cat(all_gathered, dim=0)
    else:
        return x


def knn_classifier(feature, feature_bank, feature_bank_label, num_classes, multi_label=False, k=200, t=0.1):
    # 1. get k-nearest-neighbor distance
    all_dist = torch.mm(feature, feature_bank) # B x I
    knn_dist, knn_idx = all_dist.topk(k=k, dim=-1) # B x k
    knn_dist = (knn_dist / t).exp() # B x k

    # 2. get C class probability by summing up same class distance
    if multi_label:
        knn_idx = knn_idx[:, :, None].expand(-1, -1, num_classes) # B x k x C
        one_hot_label = feature_bank_label.expand(feature.size(0), -1, -1) # B x I x C
        one_hot_label = torch.gather(one_hot_label, dim=1, index=knn_idx) # B x k x C
        one_hot_label = one_hot_label * knn_dist.unsqueeze(dim=-1) # B x k x C
        pred_scores = torch.sum(one_hot_label, dim=1) # B x C
    else:
        train_label = torch.gather(feature_bank_label.expand(feature.size(0), -1), dim=-1, index=knn_idx) # B x k
        one_hot_label = torch.zeros(feature.size(0) * k, num_classes, device=train_label.device) # (B x k) x C
        one_hot_label = one_hot_label.scatter(dim=-1, index=train_label.view(-1, 1), value=1.0) # (B x k) x C
        one_hot_label = one_hot_label.view(feature.size(0), -1, num_classes) * knn_dist.unsqueeze(dim=-1) # B x k x C
        pred_scores = torch.sum(one_hot_label, dim=1) # B x C

    return pred_scores


if __name__ == '__main__':
    feature = torch.rand((5, 10))
    feature_bank = torch.rand((20, 10)).t()
    feature_bank_label = torch.randint(2, (20, 5))
    num_classes = 5
    multi_label = True
    score = knn_classifier(feature, feature_bank, feature_bank_label, num_classes, multi_label, k=2)
    print(score)