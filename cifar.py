import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, reporters
from homura.trainers import SupervisedTrainer as Trainer
from homura.vision.data.loaders import cifar10_loaders

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20


def main():
    train_loader, test_loader = cifar10_loaders(args.batch_size)

    if args.baseline:
        model = resnet20()
    else:
        model = se_resnet20(num_classes=10, reduction=args.reduction)
    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(80, 0.1)
    tqdm_rep = reporters.TQDMReporter(range(args.epochs), callbacks=[callbacks.AccuracyCallback()])
    trainer = Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler, callbacks=[tqdm_rep])
    for _ in tqdm_rep:
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()
