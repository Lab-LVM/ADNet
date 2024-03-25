import os
from copy import deepcopy
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F

import mecla.model.res_unet
import mecla.model.arcface
import mecla.model.uit
import mecla.model.convnext
import mecla.model.maxvit


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2
    copy from timm
    """
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def get_model(args):
    if args.model_type == 'torchvision':
        model = torchvision.models.__dict__[args.model_name](
            num_classes=args.num_classes,
            pretrained=args.pretrained
        ).cuda(args.device)
    elif args.model_type == 'timm':
        if args.model_name.startswith('densenet'):
            model = timm.create_model(
                args.model_name,
                in_chans=args.in_channels,
                num_classes=args.num_classes,
                drop_rate=args.drop_rate,
                # drop_path_rate=args.drop_path_rate,
                pretrained=args.pretrained
            ).cuda(args.device)
        elif args.model_name.startswith('deit') or args.model_name.startswith('convit'):
            model = timm.create_model(
                args.model_name,
                in_chans=args.in_channels,
                img_size=args.train_size,
                num_classes=args.num_classes,
                pretrained=args.pretrained
            ).cuda(args.device)
        else:
            model = timm.create_model(
                args.model_name,
                # img_size=args.train_size,
                in_chans=args.in_channels,
                num_classes=args.num_classes,
                pretrained=args.pretrained,
            ).cuda(args.device)
    elif args.model_type == 'custom':
        if args.model_name.startswith('resnetrs'):
            model = ResnetRS.create_pretrained(
                args.model_name,
                in_ch=3,
                num_classes=args.num_classes,
            ).cuda(args.device)
    else:
        raise Exception(f"{args.model_type} is not supported yet")

    if args.checkpoint_path and Path(args.checkpoint_path).exists():
        args.log(f"load model weight from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        for key in ['model', 'state_dict']:
            if key in state_dict:
                state_dict = state_dict[key]
            for fc_name in ['fc', 'head', 'head.fc', 'classifier', 'adv_classifier']:
                fc_weight = f"{fc_name}.weight"
                fc_bias = f"{fc_name}.bias"
                if fc_weight in state_dict and model.num_classes != state_dict[fc_weight].shape[0]:
                    args.log('popping out head')
                    state_dict.pop(fc_weight)
                    state_dict.pop(fc_bias)
            for pos_embed_name in ['pos_embed']:
                if pos_embed_name in state_dict:
                    inner_shape = getattr(model, pos_embed_name).shape[-1]
                    if state_dict[pos_embed_name].shape[-1] != inner_shape:
                        state_dict[pos_embed_name] = F.interpolate(state_dict[pos_embed_name], inner_shape)
        model.load_state_dict(state_dict, strict=False)

    if args.mode == 'knn':
        for fc_name in ['fc', 'head', 'classifier', 'adv_classifier']:
            if hasattr(model, fc_name):
                if fc_name == 'head' and hasattr(model.head, 'fc'):
                    fc_name = 'head.fc'
                exec(f"args.feat_dim = model.{fc_name}.weight.shape[1]")
                exec(f"model.{fc_name} = nn.Identity()")
                break

    return model

def get_ema_ddp_model(model, args):
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.ema:
        ema_model = ModelEmaV2(model, args.ema_decay, None)
    else:
        ema_model = None

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ema_model, ddp_model