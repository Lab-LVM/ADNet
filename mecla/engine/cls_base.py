import os
import wandb

import pandas as pd
import torch
import torch.nn.functional as F
import time, datetime
from mecla.utils import compute_metrics, Metric, reduce_mean


global_step = 0

@torch.inference_mode()
def test(valid_dataloader, valid_dataset, model, args):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    predictions = list()
    labels = list()
    start_time = time.time()
    args.log(f'start validation of {args.model_name}...')

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        if args.ten_crop or args.multi_crop:
            b, ncrop, c, h, w = x.shape
            x = x.reshape(b * ncrop, c, h, w)

        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)

            if args.ten_crop or args.multi_crop:
                y_hat = torch.sigmoid(y_hat).reshape(b, ncrop, -1).mean(dim=1)

        predictions.append(y_hat)
        labels.append(y)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"{args.mode.upper()}: [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. move prediction & label to cpu and normalize prediction to probability.
    predictions = torch.concat(predictions, dim=0).detach().float().cpu()
    labels = torch.concat(labels, dim=0).detach().long().cpu()
    if args.task in ['binary', 'multilabel']:
        predictions = torch.sigmoid(predictions)
    else:
        predictions = torch.softmax(predictions, dim=-1)

    # 4. save inference result or compute metrics
    if args.mode == 'test' and args.dataset_type in ['isic2018', 'isic2019', 'isic2020']:
        metrics = []
        save_path = os.path.join(args.log_dir, f"{args.model_name}.csv")
        if args.dataset_type in ['isic2018', 'isic2019']:
            df = {"image":valid_dataset.id_list}
            df.update({c: predictions[:, i].tolist() for i, c in enumerate(valid_dataset.classes)})
        else:
            df = {"image":valid_dataset.id_list, 'malignant': predictions.squeeze().tolist()}

        pd.DataFrame(df).to_csv(save_path, index=False)
        args.log(f'saved prediction to {save_path}')

    else:
        metrics = compute_metrics(predictions, labels, args)
        metrics_dict = {k: v.detach().cpu().tolist() for k, v in zip(args.metric_names, metrics)}
        metrics_dict['pathology'] = args.classes
        save_path = os.path.join(args.log_dir, f"{args.model_name}.csv")
        metrics_csv = pd.DataFrame(metrics_dict)

        if args.dataset_type == 'nihchest':
            metrics_csv = metrics_csv.reindex([0,1,4,8,9,10,12,13,2,3,5,6,11,7])
        elif args.dataset_type == 'chexpert':
            metrics_csv = metrics_csv.reindex([3,0,2,1,4])

        metrics_csv['auroc'] = metrics_csv['auroc'] * 100
        metrics_csv = metrics_csv.T
        metrics_csv.to_csv(save_path, index=False)
        args.log(f'validation result\n' + metrics_csv.to_string())

        metrics_mean = [x.mean(dim=0).detach().float().cpu().item() for x in metrics]
        space = 12
        num_metric = 1 + len(metrics_mean)
        args.log('-'*space*num_metric)
        args.log(("{:>12}"*num_metric).format('Stage', *args.metric_names))
        args.log('-'*space*num_metric)
        args.log(f"{f'{args.mode.upper()}':>{space}}" + "".join([f"{m:{space}.4f}" for m in metrics_mean]))
        args.log('-'*space*num_metric)

    return predictions, labels, metrics


@torch.inference_mode()
def validate(valid_dataloader, model, criterion, args, epoch, train_loss=None, ema=False):
    # 1. create metric
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='Loss:')
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    ema_str = '(EMA)' if ema else ''

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    predictions = list()
    labels = list()
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(valid_dataloader):
        if args.ten_crop:
            b, ncrop, c, h, w = x.shape
            x = x.reshape(b * ncrop, c, h, w)

        batch_size = x.size(0)

        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)

            if args.ten_crop:
                y_hat = torch.sigmoid(y_hat).reshape(b, ncrop, -1).mean(dim=1)

            loss = criterion(y_hat, y)

        predictions.append(y_hat)
        labels.append(y)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"{ema_str}VALID({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    loss = loss_m.compute()
    metrics = compute_metrics(predictions, labels, args)
    metric_dict = {f"{ema_str}{m}": v for m, v in zip(args.metric_names, metrics)}
    if not ema:
        metric_dict.update({'train_loss': train_loss, 'val_loss': loss})

    # 4. print metric
    space = 12
    num_metric = 2 + len(metrics)
    args.log('-'*space*num_metric)
    args.log(("{:>12}"*num_metric).format('Stage', 'Loss', *args.metric_names))
    args.log('-'*space*num_metric)
    args.log(f"{f'{ema_str}VALID({epoch})':>{space}}" + "".join([f"{m:{space}.4f}" for m in [loss]+metrics]))
    args.log('-'*space*num_metric)

    is_best = False
    if args.best < metric_dict[ema_str+args.save_metric]:
        is_best = True
        args.best = metric_dict[ema_str+args.save_metric]

    if args.is_rank_zero and args.use_wandb:
        args.log(metric_dict, metric=True, step=epoch)
        if is_best:
            wandb.run.summary[f"best_{args.save_metric}"] = args.best

    if args.save_weight and args.is_rank_zero and is_best:
        torch.save(model.state_dict(), args.best_weight_path)

    return loss


def train_one_epoch_with_valid(
        train_dataloader, valid_dataloader, model, optimizer, criterion, val_criterion, args,
        scheduler=None, scaler=None, epoch=None, ema_model=None, kd_model=None,
    ):
    global global_step
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)

        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)
            loss = criterion(y_hat, y)

            if args.pseudo_label or args.distill_distribution:
                if args.ema:
                    with torch.no_grad():
                        prob = torch.sigmoid(ema_model.module(x))
                else:
                    prob = torch.sigmoid(y_hat)

                if args.distill_distribution:
                    loss += F.binary_cross_entropy_with_logits(y_hat, prob)

                elif args.pseudo_label:
                    target = prob.ge(0.5).float()
                    mask = torch.logical_or(prob.ge(0.95), prob.le(0.05)).float()
                    loss += F.binary_cross_entropy_with_logits(y_hat, target, mask)

            if kd_model is not None:
                with torch.no_grad():
                    prob = torch.sigmoid(kd_model(x))
                loss += F.binary_cross_entropy_with_logits(y_hat, prob)

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        if batch_idx and ema_model and batch_idx % args.ema_update == 0:
            ema_model.update(model)

        batch_m.update(time.time() - start_time)

        if args.valid_freq and batch_idx % args.valid_freq == 0:
            val_loss = validate(valid_dataloader, model, val_criterion, args, epoch=global_step, train_loss=loss_m.compute(), ema=False)
            if ema_model:
                ema_val_loss = validate(valid_dataloader, ema_model.module, val_criterion, args, epoch=global_step,
                                    train_loss=loss_m.compute(), ema=True)
            global_step += 1

        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 12
    num_metric = 4 + 1
    args.log('-'*space*num_metric)
    args.log(("{:>12}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    args.log('-'*space*num_metric)

    if args.valid_freq is None:
        val_loss = validate(valid_dataloader, model, val_criterion, args, epoch=epoch, train_loss=loss, ema=False)
        if ema_model:
            ema_val_loss = validate(valid_dataloader, ema_model.module, val_criterion, args, epoch=epoch, train_loss=loss, ema=True)

    return val_loss