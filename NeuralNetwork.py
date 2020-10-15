import numpy as np
import matplotlib.pyplot as plt
from Layers import *
from Data import DataLoaders
from Loss import Loss
from Optimizers import Optimizers
from Activations import Func
import torchvision
from torchvision import transforms
import torch
import torch.nn.functional as F
import copy


class NeuralNetwork:
    def __init__(self, categories, batch_size=10):
        self.batch_size = batch_size
        self.categories = categories

        self.input_tensor = None
        self.label_tensor = None

        self.layers = []
        self.loss = []
        self.loss_layer = None
        self.optimizer = None


    def forward(self, inputs, labels):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        outputs, loss = self.loss_layer.forward(inputs, labels)
        preds = np.argmax(outputs, axis=1)
        # loss = self.loss_layer.forward(labels, outputs)
        return outputs, loss, preds

    def backward(self, output_tensor):
        error_tensor = self.loss_layer.backward(output_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def train(self,iteration, data_generator):
        max_count = 100
        idx = 0
        for i in range(iteration):
            batch_loss = 0.0
            batch_acc = 0.0
            for inputs, labels in data_generator: #.batch_generator():
                if idx > max_count:
                    break
                idx += 1
                labels = labels.numpy()
                onehot_labels = one_hot_label(self.batch_size, labels)
                outputs,loss, preds = self.forward(inputs, onehot_labels)
                accuracy = preds[np.where(preds==labels)].shape[0]
                batch_loss += loss
                batch_acc += accuracy
                self.backward(outputs)


            # batch_loss /= len(data_generator.dataset)
            # batch_acc /= len(data_generator.dataset)
            batch_loss /= 1000
            batch_acc /= 1000
            self.loss.append(batch_loss)
            print("Epoch:{}: loss: {:.4f} acc:{:.4f}".format(i,batch_loss,batch_acc))


    def test(self, dataloaders):
        batch_loss = 0.0
        batch_acc = 0
        max_count = 100
        idx = 0
        for inputs, labels in dataloaders.batch_generator():
            if idx > max_count:
                break
            idx =+1
            labels = labels.numpy()
            inputs = inputs.numpy()
            onehot_labels = one_hot_label(self.batch_size, labels)
            outputs, loss, preds = self.forward(inputs, onehot_labels)
            accuracy = preds[np.where(preds == labels)].shape[0]
            batch_loss += loss
            batch_acc += accuracy
            self.backward(outputs)

        # batch_loss /= len(dataloaders.dataset)
        # batch_acc /= len(dataloaders.dataset)
        batch_loss /= 1000
        batch_acc /= 1000
        return batch_acc

def calculate_accuracy(preds, labels):
    idx_max = np.argmax(preds, axis=1)

    # one_hot = np.zeros_like(preds)
    # for i in range(one_hot.shape[0]):
    #     one_hot[i, idx_max[i]] = 1

    correct = idx_max[np.where(idx_max==labels)].shape[0]
    return correct/labels.shape[0]

def one_hot_label(batch_size, labels):
    onehot = np.zeros((batch_size,labels.shape[0]))
    for i in range(onehot.shape[0]):
        onehot[i][labels[i]] = 1
    return onehot

def load_mnist(batch_size):
    data_dir = r"D:\Subjects\PycharmProjects\Img\data\FashionMNIST"

    # Number of classes in the dataset
    num_classes = 10
    # Use standard FashionMNIST dataset
    train_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False, #transforms.Resize(64),
        transform=transforms.Compose([
                                      transforms.ToTensor()])
    )
    valid_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False, #
        transform=transforms.Compose([transforms.ToTensor()])
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False, #
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                              shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False)
    # Create training and validation dataloaders
    dataloaders_dict = {'train': train_loader,
                        'val': valid_loader,
                        'test': test_loader}

    return dataloaders_dict


if __name__=='__main__':
    batch_size = 10
    categories = 10
    input_size = 28
    iteration = 30
    learning_rate = 0.01


    dataset = DataLoaders.Dataset(
        root_dir=r'D:\Subjects\PycharmProjects\Img\data\dogs_small\train',
        train=True,
        test=False,
        transform=True)
    dataloaders = load_mnist(batch_size)

    net = NeuralNetwork(categories,batch_size)

    net.loss_layer = Softmax.Softmax()  # Loss.MSE()
    net.optimizer = Optimizers.Sgd(learning_rate)
    conv1 = Conv.Conv(stride_shape=(1, 1), conv_shape=(1,3,3),num_kernels=4)
    conv1.set_optimizer(copy.deepcopy(net.optimizer))
    pool = Pooling.Pooling((2, 2), (2, 2))
    # pool_out_shape = (4, 4, 4)
    fc1_input_size = 784 #4096  # np.prod(pool_out_shape)
    fc1 = FullyConnected.FullyConnected(fc1_input_size, 256)
    fc2 = FullyConnected.FullyConnected(256, categories)

    net.layers.append(conv1)
    # net.layers.append(BatchNormalization.BatchNormalization())
    net.layers.append(Func.ReLU())
    net.layers.append(pool)
    net.layers.append(Flatten.Flatten())
    net.layers.append(fc1)
    # net.layers.append(BatchNormalization.BatchNormalization())
    net.layers.append(Func.ReLU())
    net.layers.append(fc2)
    # net.layers.append(BatchNormalization.BatchNormalization())

    # data_generator = DataLoaders.DataGenerator(batch_size, dataset, shuffle=False)
    net.train(iteration, dataloaders['train'])
    plt.figure('Loss function for a Neural Net on the Cat-Dog dataset')
    plt.plot(net.loss, '-x')
    plt.show()

    dataset_test = DataLoaders.Dataset(
        root_dir=r'D:\Subjects\PycharmProjects\Img\data\dogs_small\test',
        train=False,
        test=True,
        transform=True)
    # test_data = DataLoaders.DataGenerator(10, dataset_test, shuffle=False)
    accuracy = net.test(dataloaders['test'])
    print('Test Accuracy: {:.4f}'.format(accuracy))




