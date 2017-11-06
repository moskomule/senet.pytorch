import os
from se_resnet import se_resnet50
from utils import Trainer, StepLR

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def main(batch_size, data_root):
    transform_train = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    train = datasets.ImageFolder(traindir, transform_train)
    val = datasets.ImageFolder(valdir, transform_test)
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=True, num_workers=8)
    se_resnet = se_resnet50(num_classes=1000)
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, 20, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(50, train_loader, test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=64, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root)
