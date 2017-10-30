from se_resnet import se_resnet50
from utils import Trainer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models


def main(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True,
                             transform=transform),
            batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform),
            batch_size=batch_size, shuffle=True)
    print("<<<<<< SERESNET >>>>>>")
    se_resnet = se_resnet50(num_classes=10)
    optimizer = optim.Adam(params=se_resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy)
    trainer.loop(200, train_loader, test_loader)


if __name__ == '__main__':
    main()
