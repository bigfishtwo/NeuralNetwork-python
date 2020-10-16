import numpy as np
from scipy import signal
import copy

class Conv:
    def __init__(self, stride_shape, conv_shape, num_kernels):
        # b = batch, c = channel, y, x = spatial dimension
        # n = number of kernel, f = kernel shape
        self.stride_shape = stride_shape
        self.convolution_shape = conv_shape
        self.num_kernels = num_kernels

        # initializte weights and bias
        self.weights = np.random.random_sample(np.concatenate(([self.num_kernels], self.convolution_shape)))
        self.bias = np.random.rand(self.num_kernels)
        self.input_shape = []
        self.input_pad = []
        self.gradient_weight = np.zeros(np.concatenate(([self.num_kernels], self.convolution_shape)))
        self.gradient_bias = []

    def forward(self, input_tensor):
        # returns the input tensor for the next layer.


        self.input_shape = input_tensor.shape  # (b, c, y, x)
        # zero-padding, p = (f-1)/2
        padding_size = [int((self.convolution_shape[1] - 1) // 2), int((self.convolution_shape[2] - 1) // 2)]
        # padding residual
        padding_r = [int((self.convolution_shape[1] - 1) % 2), int((self.convolution_shape[2] - 1) % 2)]

        self.input_pad = np.pad(input_tensor, ((0, 0), (0, 0), (padding_size[0] + padding_r[0], padding_size[0]),
                                (padding_size[1] + padding_r[1], padding_size[1])), 'constant',
                                constant_values=0)  #(b, c, y+f-1, x+f-1)
        output_tensor = np.zeros(np.concatenate(([input_tensor.shape[0], self.num_kernels], input_tensor.shape[2:])))  #(b, n, y, x)

        # convolution
        for b in range(input_tensor.shape[0]):
            for n in range(self.num_kernels):
                # for c in range(input_tensor.shape[1]):
                # for each batch, covolve input with every kernel, (c, y+f-1, x+f-1) * (f, f, f) = (1, y, x)
                output_tensor[b, n, :, :] = signal.correlate(self.input_pad[b, :, :, :], self.weights[n,:, :, :],
                                                                   mode='valid')[0, :, :]  # shape
                # add bias on spatial dimension
                bias = self.bias[n] * np.ones_like(output_tensor[b, n, :, :])
                output_tensor[b, n, :, :] += bias

        # stride
        output_tensor = output_tensor[:, :, 0:input_tensor.shape[2]:self.stride_shape[0],
                        0:input_tensor.shape[3]:self.stride_shape[1]]
        return output_tensor

    def backward(self, error_tensor):
        # updates the parameters using the optimizer and returns the error tensor for the next layer
        # gradient with respect to layers
        # resize kernels
        num_kernels_b = self.convolution_shape[0]
        kernels_b = np.zeros(
            (num_kernels_b, error_tensor.shape[1], self.convolution_shape[-2], self.convolution_shape[-1]))

        # restride
        error_restride = np.zeros(
            (error_tensor.shape[0], error_tensor.shape[1], self.input_shape[2], self.input_shape[3]))
        error_restride[:, :, 0:self.input_shape[2]:self.stride_shape[0],0:self.input_shape[3]:self.stride_shape[-1]] = error_tensor
        output = np.zeros((error_tensor.shape[0], num_kernels_b, self.input_shape[2], self.input_shape[3]))
        padding_size = [int((self.convolution_shape[1] - 1) // 2), int((self.convolution_shape[2] - 1) // 2)]  #(b,c,y,x)
        padding_r = [int((self.convolution_shape[1] - 1) % 2), int((self.convolution_shape[2] - 1) % 2)]
        error_pad = np.pad(error_restride, ((0, 0), (0, 0), (padding_size[0] + padding_r[0], padding_size[0]),
                                            (padding_size[1] + padding_r[1], padding_size[1])), 'constant',
                           constant_values=0)
        for i in range(num_kernels_b):
            for j in range(error_tensor.shape[1]):
                # j_r = error_tensor.shape[1] - j - 1
                kernels_b[i, j, :, :] = self.weights[j, i, :, :]
        for i in range(error_tensor.shape[0]):
            for j in range(num_kernels_b):
                output[i, j, :, :] = -signal.convolve(error_pad[i, :, :, :], kernels_b[j, :, :, :],
                                                           mode='valid')  # shape

        # gradient with respect to weights
        self.gradient_weight = np.zeros(np.concatenate(([self.num_kernels], self.convolution_shape)))
        # for b in range(self.input_shape[0]):
        for n in range(self.num_kernels):
            for c in range(self.input_shape[1]):
                self.gradient_weight[n, c, :, :] = signal.correlate(self.input_pad[:, c, :, :],
                                                                          error_restride[:, n, :, :], mode='valid')

        # gradient with respect to bias
        self.gradient_bias = np.sum(np.sum(np.sum(error_tensor, axis=3), axis=2), axis=0)

        self.weights -= 0.01 * self.gradient_weight
        self.bias -= 0.01 * self.gradient_bias
  
        return output

    def get_gradient_weights(self):
        # return the gradient with respect to the weights
        return self.gradient_weight

    def get_gradient_bias(self):
        # return the gradient with respect to the bias
        return self.gradient_bias

