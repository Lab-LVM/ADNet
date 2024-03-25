import json
import os

from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from .data_utils.transforms import TrainTransform, ValTransform


_dataset_dict = {}


def register_dataset(fn):
    dataset_name = fn.__name__
    if dataset_name not in _dataset_dict:
        _dataset_dict[fn.__name__] = fn
    # else:
    #     raise ValueError(f"{dataset_name} already exists in dataset_dict")

    return fn


def idx_wrapper(ds):
    class DatasetWithIdx(ds):
        def __getitem__(self, item):
            return super().__getitem__(item), item

    return DatasetWithIdx

def get_dataset(args, mode):
    if mode == 'train':
        # 1. define transforms
        train_transform = TrainTransform(
            resize=args.train_size,
            resize_mode=args.train_resize_mode,
            gray_image=_dataset_dict[args.dataset_type].gray_images,
            pad=args.random_crop_pad,
            scale=args.random_crop_scale,
            ratio=args.random_crop_ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            auto_aug=args.auto_aug,
            random_affine=args.random_affine,
            remode=args.remode,
            recount=args.recount,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std
        )
        val_transform = ValTransform(
            size=args.test_size,
            resize_mode=args.test_resize_mode,
            gray_image=_dataset_dict[args.dataset_type].gray_images,
            crop_ptr=args.center_crop_ptr,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            ten_crop=args.ten_crop,
            multi_crop=args.multi_crop,
        )

        # 2. define datasets
        dataset_class = _dataset_dict[args.dataset_type]
        if args.dataset_type in _dataset_dict.keys():
            train_dataset = dataset_class(
                root=args.data_dir,
                mode='train',
                transform=train_transform
            )
            val_dataset = dataset_class(
                root=args.data_dir,
                mode='valid',
                transform=val_transform
            )
            args.num_classes = 1 if train_dataset.task == 'binary' else len(train_dataset.classes)
            args.task = train_dataset.task
            args.num_labels = train_dataset.num_labels
            args.weight = train_dataset.weight
            args.norm_weight = train_dataset.norm_weight
        else:
            assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

        return train_dataset, val_dataset

    elif mode == 'valid':
        val_transform = ValTransform(
            size=args.test_size,
            resize_mode=args.test_resize_mode,
            gray_image=_dataset_dict[args.dataset_type].gray_images,
            crop_ptr=args.center_crop_ptr,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            ten_crop = args.ten_crop,
            multi_crop=args.multi_crop,
        )

        # 2. define datasets
        dataset_class = _dataset_dict[args.dataset_type]
        if args.dataset_type in _dataset_dict.keys():
            val_dataset = dataset_class(
                root=args.data_dir,
                mode=mode,
                transform=val_transform
            )
            args.num_classes = 1 if val_dataset.task == 'binary' else len(val_dataset.classes)
            args.task = val_dataset.task
            args.num_labels = val_dataset.num_labels
            args.classes = val_dataset.classes

            if args.ten_crop:
                args.eval_batch_size = int(args.batch_size // 10)
            elif args.multi_crop:
                args.eval_batch_size = int(args.batch_size // (args.multi_crop**2))
            else:
                args.eval_batch_size = args.batch_size

            val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=None, pin_memory=False)
        else:
            assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

        return val_dataset, val_dataloader

    else:
        train_transform = val_transform = ValTransform(
            size=args.test_size,
            resize_mode=args.test_resize_mode,
            gray_image=_dataset_dict[args.dataset_type].gray_images,
            crop_ptr=args.center_crop_ptr,
            interpolation=args.interpolation,
            mean=args.mean,
            std=args.std,
            ten_crop = False,
            multi_crop = False,
        )

        dataset_class = idx_wrapper(_dataset_dict[args.dataset_type])
        # dataset_class = _dataset_dict[args.dataset_type]
        if args.dataset_type in _dataset_dict.keys():
            train_dataset = dataset_class(
                root=args.data_dir,
                mode='train',
                transform=val_transform
            )
            val_dataset = dataset_class(
                root=args.data_dir,
                mode='valid',
                # mode='test',
                transform=val_transform
            )
            args.num_classes = 1 if val_dataset.task == 'binary' else len(val_dataset.classes)
            args.task = val_dataset.task
            args.num_labels = val_dataset.num_labels

        else:
            assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=False, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = SequentialSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                      num_workers=args.num_workers, collate_fn=None, pin_memory=args.pin_memory,
                                      drop_last=args.drop_last)

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                                    num_workers=args.num_workers, collate_fn=None, pin_memory=False)

        return train_dataloader, val_dataloader
        # return val_dataset, val_dataloader
