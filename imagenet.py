import torch
from homura import optim, lr_scheduler, callbacks, reporters
from homura.trainers import SupervisedTrainer, DistributedSupervisedTrainer
from homura.vision.data import imagenet_loaders
from torch.nn import functional as F

from senet.se_resnet import se_resnet50


def main():
    model = se_resnet50(num_classes=1000)

    optimizer = optim.SGD(lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([50, 70])

    c = [callbacks.AccuracyCallback(), callbacks.LossCallback()]
    r = reporters.TQDMReporter(range(args.epochs), callbacks=c)
    tb = reporters.TensorboardReporter(c)
    rep = callbacks.CallbackList(r, tb, callbacks.WeightSave("checkpoints"))

    if args.distributed:
        # DistributedSupervisedTrainer sets up torch.distributed
        if args.local_rank == 0:
            print("\nuse DistributedDataParallel")
        trainer = DistributedSupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=rep, scheduler=scheduler,
                                               init_method=args.init_method, backend=args.backend)
    else:
        multi_gpus = torch.cuda.device_count() > 1
        if multi_gpus:
            print("\nuse DataParallel")
        trainer = SupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=rep,
                                    scheduler=scheduler, data_parallel=multi_gpus)
    # if distributed, need to setup loaders after DistributedSupervisedTrainer
    train_loader, test_loader = imagenet_loaders(args.root, args.batch_size, distributed=args.distributed,
                                                 num_train_samples=args.batch_size * 10 if args.debug else None,
                                                 num_test_samples=args.batch_size * 10 if args.debug else None)
    for _ in r:
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import miniargs
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    p = miniargs.ArgumentParser()
    p.add_str("root")
    p.add_int("--epochs", default=90)
    p.add_int("--batch_size", default=128)
    p.add_true("--distributed")
    p.add_int("--local_rank", default=-1)
    p.add_str("--init_method", default="env://")
    p.add_str("--backend", default="nccl")
    p.add_true("--debug", help="Use less images and less epochs")
    args, _else = p.parse(return_unknown=True)
    num_device = torch.cuda.device_count()

    print(args)
    if args.distributed and args.local_rank == -1:
        raise RuntimeError(
            f"For distributed training, use python -m torch.distributed.launch "
            f"--nproc_per_node={num_device} {__file__} {args.root} ...")
    main()
