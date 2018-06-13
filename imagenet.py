from pathlib import Path
from se_resnet import se_resnet50
from utils import Trainer

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]
    data_augmentation = [transforms.RandomSizedCrop(224),
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


def main(batch_size, root):
    train_loader, test_loader = get_dataloader(batch_size, root)
    se_resnet = nn.DataParallel(se_resnet50(num_classes=1000),
                                device_ids=list(range(torch.cuda.device_count())))
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", help="imagenet data root")
    p.add_argument("--batch_size", default=128, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root)
