def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_tabular(title, table, args):
    title_space = int((81 - len(title)) / 2)
    args.log("-" * 81)
    args.log(" " * title_space + title)
    args.log("-" * 81)
    for (key, value) in table:
        args.log(f"{key:<25} | {value}")


def print_meta_data(model, train_dataset, test_dataset, args):
    title = 'INFORMATION'
    table = [('Project Name', args.project_name), ('Project Administrator', args.who),
             ('Experiment Name', args.exp_name), ('Experiment Start Time', args.start_time),
             ('Experiment Model Name', args.model_name), ('Experiment Log Directory', args.log_dir)]
    print_tabular(title, table, args)

    title = 'EXPERIMENT SETUP'
    table = [(target, str(getattr(args, target))) for target in [
        'train_size', 'test_size', 'center_crop_ptr',
        'interpolation', 'mean', 'std', 'hflip', 'auto_aug', 'cutmix', 'mixup', 'remode',
        'model_name', 'lr', 'epoch', 'criterion', 'optimizer', 'weight_decay', 'scheduler', 'warmup_epoch', 'batch_size'
    ]]
    print_tabular(title, table, args)

    title = 'DATA & MODEL'
    table = [('Model Parameters(M)', count_parameters(model)),
             ('Number of Train Examples', len(train_dataset)),
             ('Number of Valid Examples', len(test_dataset)),
             ('Number of Class', args.num_classes),
             ('Task', args.task)]
    print_tabular(title, table, args)
    args.log("-" * 81)
