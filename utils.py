import torch
from torch.autograd import Variable

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
        for data, target in tqdm(data_loader):
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
        mode = "train" if is_train else "test"
        print(f">>>[{mode}] loss: {sum(loop_loss):.2f}/accuracy: {sum(correct):.2f}")
        return loop_loss, correct

    def train(self, data_loader):
        self.model.train()
        loss, correct = self._loop(data_loader)

    def test(self, data_loader):
        self.model.eval()
        loss, correct = self._loop(data_loader, is_train=False)

    def loop(self, epochs, train_data, test_data):
        for _ in range(epochs):
            self.train(train_data)
            self.test(test_data)
