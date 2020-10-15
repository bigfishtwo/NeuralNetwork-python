import numpy as np

class MSE:
    def __init__(self):
        self.pred = None
        self.labels = None

    # loss function and its derivative
    def forward(self, labels, pred):
        self.pred = pred
        self.labels = labels
        return np.mean(np.power(labels - pred, 2))

    def backward(self, output_tensor):
        # to one-hot label
        label = np.zeros_like(output_tensor)
        for i in range(label.shape[0]):
            label[i][self.labels[i]] = 1

        error_tensor = 2 * (output_tensor - label) / output_tensor.shape[0]
        return error_tensor

class L2Loss:
    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)