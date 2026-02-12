import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss().to(self.args.device)
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, idxs),
            batch_size=self.args.local_bs,
            shuffle=True
        )

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum
        )

        epoch_loss = []
        for ep in range(self.args.local_ep):
            batch_loss = []
            for images, labels in self.ldr_train:
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                net.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                if self.args.optimizer  == 'fedavg':
                    optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
