import torch
from torch.nn import functional as F

from homura import callbacks, init_distributed, lr_scheduler, optim, reporters, is_distributed
from homura.trainers import SupervisedTrainer
from homura.vision import DATASET_REGISTRY
from senet.se_resnet import se_resnet50


def main():
    if is_distributed():
        init_distributed()

    model = se_resnet50(num_classes=1000)

    optimizer = optim.SGD(lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.MultiStepLR([50, 70])
    train_loader, test_loader = DATASET_REGISTRY("imagenet")(args.batch_size)

    c = [
        callbacks.AccuracyCallback(),
        callbacks.AccuracyCallback(k=5),
        callbacks.LossCallback(),
        callbacks.WeightSave("."),
        reporters.TensorboardReporter("."),
        reporters.TQDMReporter(range(args.epochs)),
    ]

    with SupervisedTrainer(model, optimizer, F.cross_entropy, callbacks=c, scheduler=scheduler,) as trainer:
        for _ in c[-1]:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == "__main__":
    import argparse
    import warnings

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    main()
