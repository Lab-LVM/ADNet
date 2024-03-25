import os
from tqdm import tqdm
import pandas as pd

import torch
from torch import distributed as dist
from torch.nn import functional as F

from mecla.utils import all_reduce_sum, knn_classifier, all_gather_with_different_size, compute_metrics


@torch.inference_mode()
def extract_features(dataloader, model, args, split):
    dataset_len = len(dataloader.dataset)
    data_type = torch.half if args.amp else torch.float
    whole_features = torch.zeros([dataset_len, args.feat_dim], device=args.device, dtype=data_type)

    if args.num_labels > 1:
        whole_labels = torch.zeros([dataset_len, args.num_labels], device=args.device, dtype=torch.float)
    else:
        whole_labels = torch.zeros([dataset_len], device=args.device, dtype=torch.long)

    is_single = os.environ.get('LOCAL_RANK', None) is None
    is_master = is_single or int(os.environ['LOCAL_RANK']) == 0

    model.eval()
    for (x, y), idx in tqdm(dataloader, desc='extract features', disable=not is_master):
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            features = model(x)
            whole_features[idx] = features
            whole_labels[idx] = y

    if args.distributed:
        whole_features = all_reduce_sum(whole_features)
        whole_labels = all_reduce_sum(whole_labels)

    if is_master:
        torch.save(whole_features.cpu(), os.path.join(args.log_dir, f"{split}_features.pth"))
        torch.save(whole_labels.cpu(), os.path.join(args.log_dir, f"{split}_labels.pth"))
        args.log(f"{split} features and labels are saved into {args.log_dir}")

    dist.barrier()



def run_test(valid_feature, valid_label, train_feature, train_label, k, t, num_classes, args):
    is_single = os.environ.get('LOCAL_RANK', None) is None
    is_master = is_single or int(os.environ['LOCAL_RANK']) == 0
    prog_bar = tqdm(zip(valid_feature, valid_label), total=len(valid_feature), disable=not is_master)

    predictions = list()
    labels = list()
    for x, y in prog_bar:
        y_hat = knn_classifier(x, train_feature, train_label, num_classes, args.num_labels > 1, k, t)
        predictions.append(y_hat)
        labels.append(y)

    metrics = compute_metrics(predictions, labels, args)

    return metrics


def test_with_knn_classifier(train_dataloader, test_dataloader, model, args):
    # 1. save features if not saved before
    if args.feature_path is None:
        extract_features(train_dataloader, model, args, 'train')
        extract_features(test_dataloader, model, args, 'val')
        args.feature_path = args.log_dir

    # 2. load features
    args.log(f"load saved features from {args.feature_path}")
    train_feature = torch.load(os.path.join(args.feature_path, f'train_features.pth')).to(args.device)
    train_label = torch.load(os.path.join(args.feature_path, f'train_labels.pth')).to(args.device)
    valid_feature = torch.load(os.path.join(args.feature_path, f'val_features.pth')).to(args.device)
    valid_label = torch.load(os.path.join(args.feature_path, f'val_labels.pth')).to(args.device)

    chunk_size = 256
    train_feature = F.normalize(train_feature, dim=-1)
    train_feature = train_feature.t()
    # valid_feature = valid_feature.split(chunk_size, dim=0)
    valid_feature = F.normalize(valid_feature, dim=-1).split(chunk_size, dim=0)
    valid_label = valid_label.split(chunk_size, dim=0)

    # 3. run evaluation
    result = list()
    t_k_list = torch.tensor([[t, k] for t in args.t for k in args.k]).tensor_split(args.world_size)[args.gpu]
    for x in t_k_list:
        t, k = round(float(x[0].item()), 3), int(x[1].item())
        x = x.to(args.device)
        metrics = run_test(valid_feature, valid_label, train_feature, train_label, k, t, args.num_classes, args)
        result.append(torch.stack(list(x) + metrics, dim=0))

    # 4. (optional) gather result
    result = torch.stack(result, dim=0)
    if args.distributed:
        result = all_gather_with_different_size(result)

    # 5. compute best top1 accuracy
    best_idx = result[:, 3].argmax()
    best_t, best_k, best_acc, best_auc = result[best_idx].cpu().tolist()[:4]

    # 6. display & save knn experiment
    if args.is_rank_zero:
        result = result.cpu().tolist()
        df = pd.DataFrame(result, columns=['t', 'k'] + args.metric_names).sort_values(by=['t', 'k'], ignore_index=True)
        df.to_csv(os.path.join(args.log_dir, 'knn_val_result.csv'))
        args.log(f'[Best KNN result] k: {int(best_k)} t: {best_t:0.2f} acc: {best_acc:.03f}% auc: {best_auc:.03f}%')
        args.log(f'validation result on {args.setting} (saved to: {args.log_dir})\n' + df.to_string())

    return best_acc, best_auc


def test_with_ensemble_knn_classifier(feature_paths, args):
    # 1. load features
    train_features = list()
    valid_features = list()
    for feature_path in feature_paths:
        args.log(f"load saved features from {feature_path}")
        train_features.append(torch.load(os.path.join(feature_path, f'train_features.pth')).to(args.device))
        valid_features.append(torch.load(os.path.join(feature_path, f'val_features.pth')).to(args.device))
        train_label = torch.load(os.path.join(feature_path, f'train_labels.pth')).to(args.device)
        valid_label = torch.load(os.path.join(feature_path, f'val_labels.pth')).to(args.device)

    chunk_size = 256
    train_feature = F.normalize(torch.cat(train_features, dim=1), dim=1).t()
    valid_feature = F.normalize(torch.cat(valid_features, dim=1), dim=1).split(chunk_size, dim=0)
    valid_label = valid_label.split(chunk_size, dim=0)

    # 3. run evaluation
    result = list()
    t_k_list = torch.tensor([[t, k] for t in args.t for k in args.k]).tensor_split(args.world_size)[args.gpu]
    for x in t_k_list:
        t, k = round(float(x[0].item()), 3), int(x[1].item())
        x = x.to(args.device)
        metrics = run_test(valid_feature, valid_label, train_feature, train_label, k, t, args.num_classes, args)
        result.append(torch.stack(list(x) + metrics, dim=0))

    # 4. (optional) gather result
    result = torch.stack(result, dim=0)
    if args.distributed:
        result = all_gather_with_different_size(result)

    # 5. compute best top1 accuracy
    best_idx = result[:, 3].argmax()
    best_t, best_k, best_acc, best_auc = result[best_idx].cpu().tolist()[:4]

    # 6. display & save knn experiment
    if args.is_rank_zero:
        result = result.cpu().tolist()
        df = pd.DataFrame(result, columns=['t', 'k'] + args.metric_names).sort_values(by=['t', 'k'], ignore_index=True)
        df.to_csv(os.path.join(args.log_dir, 'knn_val_result.csv'))
        args.log(f'[Best KNN result] k: {int(best_k)} t: {best_t:0.2f} acc: {best_acc:.03f}% auc: {best_auc:.03f}%')
        args.log(f'validation result on {args.setting} (saved to: {args.log_dir})\n' + df.to_string())

    return best_acc, best_auc