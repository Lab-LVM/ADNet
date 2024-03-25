import gc
import glob
import json
import logging
import os
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from datetime import datetime

import numpy
import torch
import wandb


def allow_print_to_master(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)

        if force or is_master:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def check_need_init():
    if os.environ.get('INITIALIZED', None):
        return False
    else:
        return True


def init_distributed_mode(args):
    os.environ['INITIALIZED'] = 'TRUE'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(f'{datetime.now().strftime("[%Y/%m/%d %H:%M]")} ', end='')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_backend = 'nccl'
        args.dist_url = 'env://'

        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
    else:
        print('| Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.gpu = 0

    args.is_rank_zero = args.gpu == 0
    allow_print_to_master(args.is_rank_zero)
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(f'cuda:{args.gpu}')


def make_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "[%Y/%m/%d %H:%M]")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(filename=log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log(msg, metric=False, logger=None, step=None):
    if logger:
        if metric:
            wandb.log(msg, step=step)
        else:
            logger.info(msg)


def init_logger(args):
    if args.resume:
        args.exp_name = Path(args.checkpoint_path).parents.name
    else:
        if args.exp_name is None:
            args.exp_name = '_'.join(str(getattr(args, target)) for target in args.exp_target)
        args.version_id = len(list(glob.glob(os.path.join(args.output_dir, f'{args.exp_name}_v*'))))
        args.exp_name = f'{args.exp_name}_v{args.version_id}'

    args.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args.log_dir = os.path.join(args.output_dir, args.exp_name)
    args.text_log_path = os.path.join(args.log_dir, 'log.txt')
    args.best_weight_path = os.path.join(args.log_dir, 'best_weight.pth')

    if args.distributed:
        torch.distributed.barrier()  # to ensure have save version id (must be same for knn classifier)

    if args.is_rank_zero:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        args.logger = make_logger(args.text_log_path)
        if args.use_wandb:
            wandb.init(project=args.project_name, name=args.exp_name, entity=args.entity,
                       config=args, reinit=True, resume=args.resume)
    else:
        args.logger = None

    args.log = partial(log, logger=args.logger)


def clear(args):
    # 1. clear gpu memory
    torch.cuda.empty_cache()
    # 2. clear cpu memory
    gc.collect()
    # 3. close logger
    args.exp_name = None
    if args.is_rank_zero:
        handlers = args.logger.handlers[:]
        for handler in handlers:
            args.logger.removeHandler(handler)
            handler.close()
        if args.use_wandb:
            wandb.finish(quiet=True)


def setup(args):
    if check_need_init():
        init_distributed_mode(args)
    init_logger(args)

    if args.seed is not None:
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True


def load_model_list_from_config(args, mode):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if mode in ['train', 'valid']:
        return config['model_list']
    else:
        return list(config['checkpoint'][args.weight_setting]['model_weight'].keys())


def load_weight_list_from_config(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    return list(config['checkpoint'].keys())


def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def get_args_with_setting(parser, config, setting=None, model_name=None, prev_args=None, mode='train',
                          eval_setting=None, weight_setting=None):
    with open(config) as f:
        config = json.load(f)

    if mode == 'train':
        parser_with_setting = deepcopy(parser)
        parser_with_setting.set_defaults(**config['settings'][setting])
        args = parser_with_setting.parse_args()

        args.setting = setting
        args.model_name = model_name
        args.data_dir = config['data_dir'][args.dataset_type]
        args.checkpoint_path = config['model_weight'].get(args.model_name, None)
        pass_required_variable_from_previous_args(args, prev_args)

        return args

    elif mode == 'valid':
        parser_with_setting = deepcopy(parser)
        parser_with_setting.set_defaults(**config[setting]['settings'])
        args = parser_with_setting.parse_args()

        args.setting = setting
        args.data_dir = config['data_dir'][args.dataset_type]
        model_weight_dict = config[setting]['model_weight']

        if len(args.model_names) == 0:
            args.model_names = list(config[args.setting]['model_weight'].keys())

        print(model_weight_dict)

        pass_required_variable_from_previous_args(args, prev_args)

        return args, model_weight_dict

    else:
        # parser_with_setting = deepcopy(parser)
        # parser_with_setting.set_defaults(**config['settings'][eval_setting])
        # args = parser_with_setting.parse_args()
        #
        # args.weight_setting = weight_setting
        # args.setting = setting
        # args.model_name = model_name
        # args.data_dir = config['data_dir'][args.dataset_type]
        # model_weight_dict = config['checkpoint'][weight_setting]
        # args.checkpoint_path = model_weight_dict['model_weight'][model_name]
        #
        # if len(args.model_names) == 0:
        #     args.model_names = list(config['checkpoint'][weight_setting]['model_weight'].keys())
        #
        # print(model_weight_dict)
        #
        # pass_required_variable_from_previous_args(args, prev_args)
        #
        # return args, model_weight_dict

        parser_with_setting = deepcopy(parser)
        parser_with_setting.set_defaults(**config['settings'][eval_setting])
        # parser_with_setting.set_defaults(**config[setting]['settings'])
        args = parser_with_setting.parse_args()

        args.setting = eval_setting
        args.model_name = model_name
        args.weight_setting = weight_setting
        args.data_dir = config['data_dir'][args.dataset_type]
        args.checkpoint_path = config['checkpoint'][weight_setting]['model_weight'].get(args.model_name, None)
        args.feature_path = config['checkpoint'][weight_setting]['feature_path'].get(args.model_name, None)
        pass_required_variable_from_previous_args(args, prev_args)

        return args, args.checkpoint_path


def print_batch_run_settings(args):
    print("Batch Run Setting")
    print(f" - model (num={len(args.model_names)}): {', '.join(args.model_names)}")
    print(f" - setting (num={len(args.settings)}): {', '.join(args.settings)}")
    print(f" - cuda: {args.cuda}")
    print(f" - output dir: {args.output_dir}")
