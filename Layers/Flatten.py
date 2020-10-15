import numpy as np
class Flatten:
    def __init__(self):
        self.input_shape = []

    def forward(self,input_tensor):
        # reshape and return input_tensor
        # input_tensor = input_tensor.reshape(-1)
        self.input_shape = input_tensor.shape[1:]
        input_tensor = input_tensor.reshape(input_tensor.shape[0], np.prod(self.input_shape))
        return input_tensor

    def backward(self,error_tensor):
        # reshape and return error_tensor
        error_tensor = error_tensor.reshape(error_tensor.shape[0], *self.input_shape)
        return error_tensor