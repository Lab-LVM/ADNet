import torch
from timm.loss import AsymmetricLossMultiLabel
from torch import nn
from torch.nn import functional as F, BCEWithLogitsLoss, MSELoss
from torch.cuda.amp import GradScaler


class NativeScalerWithGradAccum:
    def __init__(self):
        """NativeScalerWithGradAccum (timm)
        Native(pytorch) f16 scaler
        """
        self._scaler = GradScaler()

    def __call__(self, loss, optimizer, model_param, scheduler=None, grad_norm=None, update=True):
        self._scaler.scale(loss).backward()
        if update:
            if grad_norm:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_param, grad_norm)
            self._scaler.step(optimizer)
            self._scaler.update()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class BCEWithLogitsLossWithTypeCasting(BCEWithLogitsLoss):
    def __init__(self, *args, label_smoothing=0.0, els=None, class_weight=None, batch_rebalance=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = label_smoothing
        self.els = els
        if class_weight is not None:
            self.register_buffer('class_weight', torch.from_numpy(class_weight)[None, ...])
        else:
            self.class_weight = None
        self.batch_rebalance = batch_rebalance

    def forward(self, y_hat, y):
        y = y.detach().clone()

        if y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(1)

        if self.els is not None:
            num_classes = y_hat.shape[-1]
            p = torch.sigmoid(y_hat.clamp(min=1e-8))
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean(dim=1)
            ten_low = torch.topk(entropy, k=int(y.size(0)//10), largest=False)[1]
            class_dist = torch.tensor(self.els).to(y_hat.device)[None, :].expand(int(y.size(0)//10), -1)
            y[ten_low] = torch.where(y[ten_low]==1, y[ten_low] - class_dist, class_dist)
        else:
            num_classes = y_hat.shape[-1]
            off_value = self.smoothing / num_classes
            on_value = 1. - self.smoothing + off_value
            y = torch.where(y == 1, on_value, off_value)

        weight = 1.0
        if self.class_weight is not None:
            weight = self.class_weight.expand(y.size(0), -1, -1)
            weight = torch.where(y==0, weight[:, :, 0], weight[:, :, 1])

        if self.batch_rebalance:
            pos = y.float().mean()
            neg = 1 - pos
            weight = torch.where(y==0, pos, neg)

        loss = super().forward(y_hat, y)
        loss = (loss * weight).mean()

        return loss


class AsymmetricLossMultiLabelWithTypeCasting(AsymmetricLossMultiLabel):
    def forward(self, y_hat, y):
        if y_hat.shape != y.shape:
            y = F.one_hot(y.long(), num_classes=y_hat.shape[-1])
        return super().forward(y_hat, y)


def get_criterion_scaler(args):
    """Get Criterion(Loss) function and scaler
    Criterion functions are divided depending on usage of mixup
    - w/ mixup - you don't need to add smoothing loss, because mixup will add smoothing loss.
    - w/o mixup - you should need to add smoothing loss
    """
    if args.criterion in ['ce', 'crossentropy']:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
        val_criterion = nn.CrossEntropyLoss()
    elif args.criterion in ['bce', 'binarycrossentropy']:
        criterion = BCEWithLogitsLossWithTypeCasting(class_weight=args.weight if args.wce else None,
                                                     label_smoothing=args.smoothing,
                                                     els=args.norm_weight if args.els else None,
                                                     batch_rebalance=args.batch_rebalance,
                                                     reduction='none').to(args.device)
        val_criterion = BCEWithLogitsLossWithTypeCasting()
    elif args.criterion in ['mse', 'l2']:
        criterion = MSELoss()
        val_criterion = MSELoss()
    elif args.criterion in ['asym', 'assymetric']:
        criterion = AsymmetricLossMultiLabelWithTypeCasting()
        val_criterion = BCEWithLogitsLossWithTypeCasting()

    if args.amp:
        scaler = NativeScalerWithGradAccum()
    else:
        scaler = None

    return criterion, val_criterion, scaler


if __name__ == '__main__':
    x = torch.rand(2, 14)
    y = torch.randint(2, (2, 14))
    critic = BCEWithLogitsLossWithTypeCasting(label_smoothing=0.1)
    print(critic(x, y))