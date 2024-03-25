import os
import argparse

from mecla.dataset import get_dataset, get_dataloader
from mecla.engine import test_with_knn_classifier, test, test_with_ensemble_knn_classifier
from mecla.model import get_model
from mecla.utils import setup, get_args_with_setting, clear, load_model_list_from_config, load_weight_list_from_config


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='pytorch-medical-classification(MECLA)',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1.setup
    setup = parser.add_argument_group('setup')
    setup.add_argument(
        '--config', type=str, default=os.path.join('config', 'knn.json'),
        help='paths for each dataset and pretrained-weight. (json)'
    )
    setup.add_argument(
        '-es', '--eval-settings', type=str, default=['chexpertv1', 'chexpertv2'], nargs='+',
        help='settings used in overall evaluation'
    )
    setup.add_argument(
        '-ws', '--weight-settings', type=str, default=['chexpertv1_v1',], nargs='+',
        help='settings used for choosing weight of model'
    )
    setup.add_argument(
        '--eval-protocol', type=str, default='knn', choices=['knn'],
        help="choose between fully-connected classifier and knn classifier"
    )
    setup.add_argument(
        '--t', type=float, default=0.07, nargs='+',
        help = 'temperature used for knn classifier. this value is copied from moco cifar demo.'
    )
    setup.add_argument(
        '--k', type=int, default=20, nargs='+',
        help = 'k for knn classifier. this value is copied from moco cifar demo.'
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
        '--exp-target', type=str, default=['weight_setting', 'model_name'], nargs='+',
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
    setup.set_defaults(amp=False, channel_last=False, pin_memory=True,
                       resume=None, mode='knn', train_split='train', val_split='val',
                       aug_repeat=False, mixup=None, cutmix=None, use_arcface=False,)

    # 2. augmentation & dataset & dataloader
    data = parser.add_argument_group('data')
    data.add_argument(
        '--dataset-type', type=str, default='chexpert_with_idx',
        choices=['chexpert_with_idx'],
        help='dataset type'
    )
    data.add_argument(
        '--feature-path', type=str, default=None,
        help='feature saved path used for knn classifier'
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
        '-j', '--num-workers', type=int, default=4,
        help='number of workers'
    )
    data.add_argument(
        '--pin-memory', action='store_true', default=False,
        help='pin memory in dataloader'
    )
    data.add_argument(
        '--drop-last', action='store_true', default=False,
        help='drop last batch in train dataloader'
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
    model.add_argument(
        '--metric-names', type=str, nargs='+',
        default=[
            'accuracy', 'auroc', 'f1_score', 'specificity',
            'recall', 'precision', 'average_precision',
        ],
        help='metric name'
    )

    return parser


def run(args):
    # 1. load data
    train_dataloader, test_dataloader = get_dataset(args, args.mode)

    # 2. load model
    model = get_model(args)

    # 3. evaluate model
    top1, top5 = test_with_knn_classifier(train_dataloader, test_dataloader, model, args)

    # 4. log result
    if args.use_wandb:
        args.log({'top1': top1, 'top5': top5}, metric=True)


if __name__ == '__main__':
    # 1. parse command
    parser = get_args_parser()
    args = parser.parse_args()

    prev_args = None
    need_to_load_setting = len(args.weight_settings) == 0
    need_to_load_models = len(args.model_names) == 0
    feature_paths = list()

    for eval_setting in args.eval_settings:
        # 2. load weight_settings
        weight_settings = load_weight_list_from_config(args)
        if need_to_load_setting:
            args.weight_settings = weight_settings

        for weight_setting in args.weight_settings:
            # 3. load model_list
            args.weight_setting = weight_setting
            model_names = load_model_list_from_config(args, args.mode)
            if need_to_load_models:
                args.model_names = model_names

            # 4. valid each model (3 steps)
            for model_name in args.model_names:
                # 3-1. create new args object for new logger assignment for each mode
                new_args, _ = get_args_with_setting(parser, args.config, model_name=model_name, prev_args=prev_args,
                                                 eval_setting=eval_setting, weight_setting=weight_setting,
                                                 mode=args.mode)

                # 3-2. run model
                setup(new_args)
                run(new_args)
                clear(new_args)
                prev_args = new_args
                feature_paths.append(new_args.feature_path)

    # setup(new_args)
    # feature_paths = list(set(feature_paths))
    # test_with_ensemble_knn_classifier(feature_paths, new_args)
    clear(new_args)
