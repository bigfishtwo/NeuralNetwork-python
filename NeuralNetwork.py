import numpy as np
import matplotlib.pyplot as plt
from Layers import *
from Data import DataLoaders
from Loss import Loss
from Optimizers import Optimizers
from Activations import Func
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


    def forward(self, inputs, labels):
        for layer in self.layers:
            inputs = layer.forward(inputs)

        outputs, loss = self.loss_layer.forward(inputs, labels)
        preds = np.argmax(outputs, axis=1)
        return outputs, loss, preds

    def backward(self, output_tensor):
        error_tensor = self.loss_layer.backward(output_tensor)
        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def train(self,iteration, data_generator):
        for i in range(iteration):
            batch_loss = 0.0
            batch_acc = 0.0
            for inputs, labels in data_generator.batch_generator():
                onehot_labels = one_hot_label(self.batch_size, labels)
                outputs,loss, preds = self.forward(inputs, onehot_labels)
                accuracy = preds[np.where(preds==labels)].shape[0]
                batch_loss += loss
                batch_acc += accuracy
                self.backward(outputs)


            batch_loss /= len(data_generator.dataset)
            batch_acc /= len(data_generator.dataset)

            self.loss.append(batch_loss)
            print("Epoch:{}: loss: {:.4f} acc:{:.4f}".format(i,batch_loss,batch_acc))


    def test(self, dataloaders):
        batch_loss = 0.0
        batch_acc = 0
        for inputs, labels in dataloaders.batch_generator():
            onehot_labels = one_hot_label(self.batch_size, labels)
            outputs, loss, preds = self.forward(inputs, onehot_labels)
            accuracy = preds[np.where(preds == labels)].shape[0]
            batch_loss += loss
            batch_acc += accuracy
            self.backward(outputs)

        batch_loss /= len(dataloaders.dataset)
        batch_acc /= len(dataloaders.dataset)
        return batch_acc

def calculate_accuracy(preds, labels):
    idx_max = np.argmax(preds, axis=1)
    correct = idx_max[np.where(idx_max==labels)].shape[0]
    return correct/labels.shape[0]

def one_hot_label(batch_size, labels):
    onehot = np.zeros((batch_size,labels.shape[0]))
    for i in range(onehot.shape[0]):
        onehot[i][labels[i]] = 1
    return onehot

