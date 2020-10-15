import numpy as np
import copy
import Helper

class BatchNormalization:
    def __init__(self,channels=0):
        self.bias = np.array([0]) # beta
        self.weights = np.array([1])  # gamma
        self.mean = 0
        self.var = 0
        self.channels = channels
        self.input_tensor = []
        self.input_shape = []
        self.input_tensor_norm = []

        self.gradient_weights = 0
        self.gradient_bias = 0
        self.optimizer_weights = None
        self.optimizer_bias = None
        self.initial = True

        # self.interation_b = 0
        # self.interation_f = 0


    def forward(self, input_tensor):
        if self.channels > 0:
            input_tensor=self.reformat(input_tensor)
        # Normalization as a new layer with 2 parameters, γ and β

        self.input_tensor = input_tensor
        mean_b = np.mean(input_tensor,axis=0,keepdims=True)
        var_b = np.var(input_tensor,axis=0,keepdims=True)

        # var_b = var_b.reshape(input_tensor.shape)
        if self.initial:
            self.weights = np.ones((1,input_tensor.shape[1]))
            self.bias = np.zeros((1,input_tensor.shape[1]))
            self.mean = mean_b
            self.var = var_b
            self.initial=False

        # if self.phase == Base.Phase.train:
        alpha = 0.8
        self.mean = alpha * self.mean + (1- alpha) * mean_b
        self.var = alpha * self.var + (1- alpha) * var_b
        # # test time
        # if self.phase == Base.Phase.test:
        #     mean_b= self.mean
        #     var_b= self.var

        self.input_tensor_norm = (input_tensor - mean_b) / np.sqrt(var_b+1e-18)  # normalized input_tensor
        output_tensor = self.weights*self.input_tensor_norm+ self.bias

        # self.interation_f += 1
        # if self.interation_f == 8:
        #     print("input_tensor shape:" , self.input_tensor.shape)
        #     print("mean shape:",self.mean.shape)

        if self.channels > 0:
            output_tensor=self.reformat(output_tensor)
        return output_tensor

    def backward(self,error_tensor):
        # compute_bn_gradients
        if self.channels > 0:
            error_tensor=self.reformat(error_tensor)
        # self.interation_b +=1
        # if self.input_tensor.shape[0] == 148:
        #     print(self.interation_b,"shape of mean:",self.mean.shape, "shape input:",self.input_tensor.shape)
        gradient_input = Helper.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var, eps=1e-18)
        self.gradient_weights = np.sum((error_tensor*self.input_tensor_norm),axis=0,keepdims=True)
        self.gradient_bias = np.sum(error_tensor,axis=0,keepdims=True)
        if self.channels > 0:
            gradient_input=self.reformat(gradient_input)
        if self.optimizer_weights is not None and self.optimizer_bias is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.gradient_bias)
        return gradient_input

    def reformat(self,tensor):
        if tensor.ndim == 4:
            # vectorise
            self.input_shape = tensor.shape
            tensor = tensor.reshape((tensor.shape[0],tensor.shape[1],np.prod(tensor.shape[2:])))
            tensor = np.transpose(tensor,(0,2,1))
            tensor = tensor.reshape((np.prod((tensor.shape[0],tensor.shape[1])), tensor.shape[2]))
            return tensor
        else:
            # invers
            b, c, w, h = self.input_shape
            tensor = tensor.reshape((b, w * h, c))
            tensor = np.transpose(tensor, (0,2,1))
            tensor = tensor.reshape(self.input_shape)
            return tensor

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias

    def initialize(self,weight_initializer,bias_initializer):
        self.weights = weight_initializer.initialize(self.weights.shape,self.weights.ndim,self.weights.ndim)
        self.bias = bias_initializer.initialize(self.bias.shape,self.bias.ndim,self.bias.ndim)

    def set_optimizer(self,optimizer):
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_bias = copy.deepcopy(optimizer)
