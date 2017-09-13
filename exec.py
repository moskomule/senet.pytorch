from se_resnet import se_resnet50
from se_inception import se_inception_v3
from utils import Trainer

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models


def main(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    optimizer = optim.Adam(params=se_resnet.parameters(), lr=1e-3)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy)
    trainer.loop(10, train_loader, test_loader)


if __name__ == '__main__':
    main()
