import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import copy
import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision
import os

class Dataset:
    def __init__(self, root_dir, train, test, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # TODO: rewrite dataloader
        self.root_dir = root_dir
        self.train = train
        self.test = test
        self.transform = transform

    def __len__(self):
        return len(os.listdir(dir)


    def __getitem__(self, index):
        img_name = self.root_dir + + str(index) + '.jpg'
        label = index
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, label


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(784,256) #4096
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

train_loader = torch.utils.data.DataLoader(
    dataset = Dataset(
        root_dir=r'.\train',
        train=True,
        test=False,
        transform=transforms.Compose([transforms.Resize((64,64)),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomVerticalFlip(p=0.5),
                                      transforms.ToTensor()
                                      ])),
    batch_size=10,
    shuffle=False,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    dataset = Dataset(
        root_dir=r'.\test',
        train=False,
        test=True,
        transform=transforms.Compose([transforms.Resize((64,64)),
                                      transforms.ToTensor()
                                      ])),
    batch_size=10,
    shuffle=False,
    num_workers=4
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01)
dataloaders = {'train': train_loader,
                 'test': test_loader}

def train(epoch):
    net.train()
    batch_correct = 0
    batch_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloaders['train']):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = net(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            output = F.softmax(output, dim=1)
            _, pred = torch.max(output, 1)

            # pred = output.max(1, keepdim=True)[1]
            loss.backward()
            optimizer.step()
        batch_correct += pred.eq(target.view_as(pred)).sum().item()
        batch_loss += loss.item()
    print('Train Epoch: {} \nLoss: {:.6f} \tAccuracy: {:.4f}'.format(
        epoch, batch_loss/len(dataloaders['train'].dataset),
        batch_correct / len(dataloaders['train'].dataset)))

def test():
    with torch.no_grad():
        net.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloaders['test']:
            data, target = data.to(device), target.to(device)
            output = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloaders['test'].dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(dataloaders['test'].dataset),
                      100. * correct / len(dataloaders['test'].dataset)))

if __name__ == '__main__':
    # for multiprocessing
    torch.multiprocessing.freeze_support()

    print("Params to learn:")
    for name, param in net.named_parameters():
        print("\t", name)

    for i in range(10):
        train(i)

    test()

