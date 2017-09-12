from senet import se_resnet50

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models

from tqdm import tqdm


class Trainer(object):
    cuda = torch.cuda.is_available()

    def __init__(self, model, optimizer, loss_f):
        self.model = model
        if self.cuda:
            model.cuda()
        self.optimizer = optimizer
        self.loss_f = loss_f

    def _loop(self, data_loader, is_train=True):
        loop_loss = []
        correct = []
        for idx, (data, target) in tqdm(enumerate(data_loader)):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            if is_train:
                self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_f(output, target)
            loop_loss.append(loss.data[0] / len(data_loader))
            correct.append((output.data.max(1)[1] == target.data).sum() / len(data_loader.dataset))
            if is_train:
                loss.backward()
                self.optimizer.step()
        print(f"loss: {sum(loop_loss):.2f}")
        print(f"accuracy: {sum(correct):.2f}")
        return loop_loss, correct

    def train(self, data_loader):
        print(">>>train")
        loss, correct = self._loop(data_loader)

    def test(self, data_loader):
        print(">>>test")
        loss, correct = self._loop(data_loader, is_train=False)

    def loop(self, epochs, train_data, test_data):
        for _ in range(epochs):
            self.train(train_data)
            self.test(test_data)


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
    print("<<<<<< SENET >>>>>>")
    senet = se_resnet50()
    optimizer = optim.Adam(params=senet.parameters(), lr=1e-3)
    trainer = Trainer(senet, optimizer, F.cross_entropy)
    trainer.loop(10, train_loader, test_loader)

    print("<<<<<< RESNET >>>>>>")
    resnet = models.resnet50()
    resnet.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    optimizer = optim.Adam(params=resnet.parameters(), lr=1e-3)
    trainer = Trainer(resnet, optimizer, F.cross_entropy)
    trainer.loop(10, train_loader, test_loader)


if __name__ == '__main__':
    main()
