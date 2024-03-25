from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import RandomChoice

from mecla.dataset import MixUP, CutMix, RepeatAugSampler


def get_dataloader(train_dataset, val_dataset, args):
    # 1. create sampler
    if args.distributed:
        if args.aug_repeat:
            train_sampler = RepeatAugSampler(train_dataset, num_repeats=args.aug_repeat)
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        if args.aug_repeat:
            train_sampler = RepeatAugSampler(train_dataset, num_repeats=args.aug_repeat)
        else:
            train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # 2. create collate_fn
    mix_collate = []
    if args.mixup:
        mix_collate.append(MixUP(alpha=args.mixup, nclass=args.num_classes))
    if args.cutmix:
        mix_collate.append(CutMix(alpha=args.mixup, nclass=args.num_classes))

    if mix_collate:
        mix_collate = RandomChoice(mix_collate)
        collate_fn = lambda batch: mix_collate(*default_collate(batch))
    else:
        collate_fn = None

    # 3. create dataloader
    if args.ten_crop:
        args.eval_batch_size = int(args.batch_size // 10)
    else:
        args.eval_batch_size = args.batch_size

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory,
                                  drop_last=args.drop_last)

    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, sampler=val_sampler,
                                  num_workers=args.num_workers, collate_fn=None, pin_memory=False)

    args.iter_per_epoch = len(train_dataloader)

    return train_dataloader, val_dataloader