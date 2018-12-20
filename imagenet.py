from pathlib import Path

import torch
import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, reporter
from homura.utils.trainer import SupervisedTrainer as Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from senet.se_resnet import se_resnet50


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
    data_augmentation = [transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(), ]

    traindir = str(Path(root) / "train")
    valdir = str(Path(root) / "val")
    train = datasets.ImageFolder(traindir, transforms.Compose(data_augmentation + to_normalized_tensor))
    val = datasets.ImageFolder(valdir, transforms.Compose(to_normalized_tensor))
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        val, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, test_loader


def main():
    train_loader, test_loader = get_dataloader(args.batch_size, args.root)
    gpus = list(range(torch.cuda.device_count()))
    se_resnet = nn.DataParallel(se_resnet50(num_classes=1000),
                                device_ids=gpus)
    optimizer = optim.SGD(lr=0.6 / 1024 * args.batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(30, gamma=0.1)
    weight_saver = callbacks.WeightSave("checkpoints")
    tqdm_rep = reporter.TQDMReporter(range(args.epochs), callbacks=[callbacks.AccuracyCallback()])

    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, scheduler=scheduler,
                      callbacks=callbacks.CallbackList(weight_saver, tqdm_rep))
    for _ in tqdm_rep:
        trainer.train(train_loader)
        trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--epochs", default=90, type=int)
    args = p.parse_args()
    main()
