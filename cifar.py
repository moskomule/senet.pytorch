import torch.nn.functional as F

from homura import callbacks, lr_scheduler, optim, reporters
from homura.trainers import SupervisedTrainer as Trainer
from homura.vision import DATASET_REGISTRY
from senet.baseline import resnet20
from senet.se_resnet import se_resnet20


def main():
    train_loader, test_loader = DATASET_REGISTRY("cifar10")(args.batch_size, num_workers=args.num_workers)

    if args.baseline:
        model = resnet20()
    else:
        model = se_resnet20(num_classes=10, reduction=args.reduction)

    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(80, 0.1)
    tqdm_rep = reporters.TQDMReporter(range(args.epochs))
    _callbacks = [tqdm_rep, callbacks.AccuracyCallback()]
    with Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler, callbacks=_callbacks) as trainer:
        for _ in tqdm_rep:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()
