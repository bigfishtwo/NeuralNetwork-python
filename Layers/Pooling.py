import numpy as np

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride = stride_shape
        self.pooling_shape = pooling_shape
        self.max_index = []
        self.input_shape = []
        self.output_shape = []

    def forward(self, input_tensor):
        # return input_tensor for next layer

        self.input_shape = input_tensor.shape
        self.output_shape = (input_tensor.shape[0], input_tensor.shape[1],
                             (input_tensor.shape[2] - self.pooling_shape[0]) // self.stride[0] + 1,
                             (input_tensor.shape[3] - self.pooling_shape[1]) // self.stride[1] + 1)

        output = np.empty(self.output_shape)
        self.max_index = np.zeros(self.output_shape, dtype=int)

        for b in range(self.output_shape[0]):
            for c in range(self.output_shape[1]):
                for i in range(self.output_shape[2]):
                    for j in range(self.output_shape[3]):
                        pooling_x = i * self.stride[0]
                        pooling_y = j * self.stride[1]
                        pooling_field = input_tensor[b, c, pooling_x:pooling_x + self.pooling_shape[0],
                                        pooling_y:pooling_y + self.pooling_shape[1]]
                        output[b, c, i, j] = np.max(pooling_field)
                        self.max_index[b, c, i, j] = np.argmax(pooling_field)

        return output

    def backward(self, error_tensor):
        # return error_tensor for next layer

        error_extend = np.zeros(self.input_shape)

        for b in range(self.input_shape[0]):
            for c in range(self.input_shape[1]):
                for i in range(self.output_shape[2]):
                    for j in range(self.output_shape[3]):
                        back_x = i * self.stride[0]
                        back_y = j * self.stride[1]
                        pooling_field = error_extend[b, c, back_x:back_x + self.pooling_shape[0],
                                        back_y:back_y + self.pooling_shape[1]]
                        index0 = self.max_index[b, c, i, j] // self.pooling_shape[0]
                        index1 = self.max_index[b, c, i, j] % self.pooling_shape[1]
                        pooling_field[index0, index1] += error_tensor[b, c, i, j]
                        error_extend[b, c, back_x:back_x + self.pooling_shape[0],
                        back_y:back_y + self.pooling_shape[1]] = pooling_field

        error_tensor = error_extend
        return error_tensor
