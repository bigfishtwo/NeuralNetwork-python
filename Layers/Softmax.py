import numpy as np

class Softmax:
    def __init__(self):
        self.input_tensor = []
        self.label_tensor = []

    def predict(self, input):
        '''
        predict labels with given input tensor
        :param input: input tensor x
        :return: predicted labels y_h, represent class probability
        '''
        input_tensor = input - np.tile(np.max(input, axis=1), (input.shape[1], 1)).T
        exp = np.exp(input_tensor)
        sum = np.sum(exp, axis=1)
        label = exp/ np.tile(sum, (input_tensor.shape[1], 1)).T
        return label

    def forward(self, input, labels):
        '''
        softmax loss = sum over batch(- log y_h)
        :param input: input tensor x
        :param labels: true labels y, one-hot label
        :return: prediction, softmax loss
        '''
        self.label_tensor = self.predict(input)
        loss = np.sum(-np.log(self.label_tensor[np.where(labels == 1)]))
        return self.label_tensor, loss

    def backward(self, label):
        '''
        start point of backward, loss = y_true - y_pred
        :param label: true label
        :return: loss/ error
        '''
        self.label_tensor[np.where(label == 1)] -= 1
        return self.label_tensor