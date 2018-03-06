import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from se_resnet import se_resnet20
from baseline import resnet20
from utils import Trainer, StepLR


def main(batch_size, baseline, reduction):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True)
    if baseline:
        model = resnet20()
    else:
        model = se_resnet20(num_classes=10, reduction=reduction)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = StepLR(optimizer, 80, 0.1)
    trainer = Trainer(model, optimizer, F.cross_entropy)
    trainer.loop(200, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main(args.batchsize, args.baseline, args.reduction)
