import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.delta = 0.01
        self.input_tensor = None
        self.output_tensor = None
        self.weights = np.random.rand(self.input_size+1, self.output_size) - 0.5
        # self.bias = np.random.rand(1,output_size) - 0.5#input_size plus one for the bias
        self.error = []
        self.optimizer = None

    def forward(self, input_tensor):
        # returns the input tensor for the next layer
        # extend input matrix with bias
        self.input_tensor = np.concatenate((input_tensor, np.ones([input_tensor.shape[0], 1])), axis=1)
        # self.input_tensor = input_tensor
        input_tensor = np.dot(self.input_tensor, self.weights)  #+ self.bias
        return input_tensor

    def backward(self,error_tensor):
        # updates the parameters and returns the error tensor for the next layer
        self.error = error_tensor
        error_tensor = np.dot(self.error,self.weights.T)
        gradient_w = self.get_gradient_weights()
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights,gradient_w)
        else:
            self.weights -= self.delta * gradient_w
        # self.bias -= self.delta * self.error
        error_tensor = np.delete(error_tensor,-1,1)
        return error_tensor

    def get_gradient_weights(self):
        # returns the gradient with respect to the weights, after they have been calculated in the backward-pass.
        gradient_w = np.dot(self.input_tensor.T, self.error)
        return gradient_w


    def initialize(self,weights_initializer,bias_initializer):
        weight = self.weights[0:-1,:]
        bias = self.weights[-1,:].reshape(1,self.weights.shape[1])
        weight = weights_initializer.initialize(weight.shape,weight.ndim,weight.ndim)
        bias = bias_initializer.initialize(bias.shape,bias.ndim,bias.ndim)
        self.weights = np.concatenate((weight,bias),axis=0)

    def set_optimizer(self,optimizer):
        self.optimizer = optimizer