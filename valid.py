import gc
import os
import argparse
import pandas as pd

import torch
import wandb

from mecla.dataset import get_dataset
from mecla.engine.cls_base import validate
from mecla.model import get_model
from mecla.engine import test
from mecla.utils import setup, get_args_with_setting, compute_metrics, print_batch_run_settings, clear


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='pytorch-medical-classification(MECLA)',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1.setup
    setup = parser.add_argument_group('setup')
    setup.add_argument(
        '--config', type=str, default=os.path.join('config', 'valid.json'),
        help='paths for each dataset and pretrained-weight. (json)'
    )
    setup.add_argument(
        '--mode', type=str, default='valid', choices=['valid', 'test'],
        help='choose split between (valid, test)'
    )
    setup.add_argument(
        '-s', '--settings', type=str, default=['isic2018_v1'], nargs='+',
        help='settings used for default value'
    )
    setup.add_argument(
        '--entity', type=str, default='mecla',
        help='project space used for wandb logger'
    )
    setup.add_argument(
        '-proj', '--project-name', type=str, default='MECLA-valid',
        help='project name used for wandb logger'
    )
    setup.add_argument(
        '--who', type=str, default='hankyul2',
        help='enter your name'
    )
    setup.add_argument(
        '--use-wandb', action='store_true', default=False,
        help='track std out and log metric in wandb'
    )
    setup.add_argument(
        '-exp', '--exp-name', type=str, default=None,
        help='experiment name for each run'
    )
    setup.add_argument(
        '--exp-target', type=str, default=['setting', 'mode'], nargs='+',
        help='experiment name based on arguments'
    )
    setup.add_argument(
        '-out', '--output-dir', type=str, default='log_val',
        help='where log output is saved'
    )
    setup.add_argument(
        '-p', '--print-freq', type=int, default=50,
        help='how often print metric in iter'
    )
    setup.add_argument(
        '--seed', type=int, default=42,
        help='fix seed'
    )
    setup.add_argument(
        '--amp', action='store_true', default=False,
        help='enable native amp(fp16) training'
    )
    setup.add_argument(
        '--channels-last', action='store_true',
        help='change memory format to channels last'
    )
    setup.add_argument(
        '-c', '--cuda', type=str, default='0,1,2,3,4,5,6,7,8',
        help='CUDA_VISIBLE_DEVICES options'
    )
    setup.set_defaults(amp=True, channel_last=True, pin_memory=True, resume=None, use_deterministic=False,
                       dft_pool=False, drop_rate=0.0, weight_pool=False)

    # 2. augmentation & dataset & dataloader
    data = parser.add_argument_group('data')
    data.add_argument(
        '--dataset-type', type=str, default='chexpert',
        choices=[
            'chexpert', 'nihchest', # chest
            'ddsm', 'vindr', # breast
            'isic2018', 'isic2019', # skin
            'eyepacs', 'messidor2', # eye
            'pcam', # lymph
        ],
        help='dataset type'
    )
    data.add_argument(
        '--test-size', type=int, default=(224, 224), nargs='+',
        help='test image size'
    )
    data.add_argument(
        '--test-resize-mode', type=str, default='resize_shorter', choices=['resize_shorter', 'resize'],
        help='test resize mode'
    )
    data.add_argument(
        '--center-crop-ptr', type=float, default=0.875,
        help='test image crop percent'
    )
    data.add_argument(
        '--interpolation', type=str, default='bicubic',
        help='image interpolation mode'
    )
    data.add_argument(
        '--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+',
        help='image mean'
    )
    data.add_argument(
        '--std', type=float, default=(0.229, 0.224, 0.225), nargs='+',
        help='image std'
    )
    data.add_argument(
        '-b', '--batch-size', type=int, default=256,
        help='batch size'
    )
    data.add_argument(
        '-j', '--num-workers', type=int, default=8,
        help='number of workers'
    )
    data.add_argument(
        '--pin-memory', action='store_true', default=False,
        help='pin memory in dataloader'
    )
    data.add_argument(
        '--ten-crop', action='store_true',
        help='apply 10 x crop'
    )
    data.add_argument(
        '--multi-crop', type=int, default=None,
        help='apply multi crop'
    )
    data.add_argument(
        '--drop-last', action='store_true',
        help='drop last batch',
    )

    # 3.model
    model = parser.add_argument_group('model')
    model.add_argument(
        '-m', '--model-names', type=str, default=[], nargs='+',
        help='model name'
    )
    model.add_argument(
        '--model-type', type=str, default='timm',
        help='timm or torchvision'
    )
    model.add_argument(
        '--in-channels', type=int, default=3,
        help='input channel dimension'
    )
    model.add_argument(
        '--drop-path-rate', type=float, default=0.0,
        help='stochastic depth rate'
    )
    model.add_argument(
        '--sync-bn', action='store_true', default=False,
        help='apply sync batchnorm'
    )
    model.add_argument(
        '--pretrained', action='store_true', default=False,
        help='load pretrained weight'
    )

    # 4.optimizer & scheduler & criterion
    metric = parser.add_argument_group('metric')
    metric.add_argument(
        '--metric-names', type=str, nargs='+',
        default=[
            'accuracy', 'auroc', 'f1_score', 'specificity',
            'recall', 'precision', 'average_precision',
        ],
        help='metric name'
    )

    return parser


def run(args, valid_dataset, valid_dataloader):
    model = get_model(args)
    result = test(valid_dataloader=valid_dataloader, valid_dataset=valid_dataset, model=model, args=args)

    return result


if __name__ == '__main__':
    # 1. parse command
    parser = get_args_parser()
    args = parser.parse_args()

    # 2. run N(setting) x N(model_names) experiment
    prev_args = None
    for setting in args.settings:
        # 2-1. load complementary option from cmd and set logger
        new_args, model_weight_dict = get_args_with_setting(parser, args.config, setting, None, prev_args, args.mode)
        setup(new_args)
        print_batch_run_settings(new_args)

        # 2-2. load dataset & dataloader
        valid_dataset, valid_dataloader = get_dataset(new_args, new_args.mode)

        # 2-2. valid each model
        pred_list = []
        metric_list = []
        for model_name in new_args.model_names:
            new_args.model_name, new_args.checkpoint_path = model_weight_dict.get(model_name, None)
            pred, label, metric = run(new_args, valid_dataset, valid_dataloader)
            pred_list.append(pred)
            metric_list.append([model_name]+metric)

            torch.cuda.empty_cache()
            gc.collect()

        # 2-3. valid ensemble of each model
        if len(new_args.model_names) > 1:
            if new_args.mode == 'test' and new_args.dataset_type in ['isic2018', 'isic2019']:
                save_path = os.path.join(new_args.log_dir, "ensemble.csv")
                df = {"image": valid_dataset.id_list}
                df.update({c: (sum(pred_list)/len(pred_list))[:, i].tolist() for i, c in enumerate(valid_dataset.classes)})
                pd.DataFrame(df).to_csv(save_path, index=False)
                new_args.log(f'saved prediction to {save_path}')

            else:
                metric = [x.item() for x in compute_metrics(sum(pred_list)/len(pred_list), label, new_args)]
                metric_list.append(['ensemble']+metric)
                columns = ['model_name'] + args.metric_names

                table = pd.DataFrame({columns[i]: [row[i] for row in metric_list] for i in range(len(columns))})
                new_args.log(f'validation result on {setting}\n' + table.to_string())

                if new_args.use_wandb:
                    new_args.log({"valid result": wandb.Table(dataframe=table)}, metric=True)

        clear(new_args)
        prev_args = new_args