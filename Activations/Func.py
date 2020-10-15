import numpy as np

# ReLU
class ReLU:
    def __init__(self):
        self.input_tensor = []

    def forward(self, input):
        '''
        ReLU activation function: f(x) = max(0,x)
        '''
        self.input_tensor = input
        input[np.where(input <= 0)] = 0
        return input

    def backward(self, error):
        '''
        derivative: f'(x) = 1 ,if x > 0
                            0 ,else
        :param error: gradient of loss with respect to y = f(x)
        :return: gradient of loss with respect to x
        '''
        error[np.where(self.input_tensor <= 0)] = 0
        return error

