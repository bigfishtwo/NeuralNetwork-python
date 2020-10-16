if __name__=='__main__':
    batch_size = 10
    categories = 10
    input_size = 28
    iteration = 30
    learning_rate = 0.01


    dataset = DataLoaders.Dataset(
        root_dir=r'.\train',
        train=True,
        test=False,
        transform=True)

    net = NeuralNetwork(categories,batch_size)

    net.loss_layer = Softmax.Softmax()  
    conv1 = Conv.Conv(stride_shape=(1, 1), conv_shape=(1,3,3),num_kernels=4)
    pool = Pooling.Pooling((2, 2), (2, 2))
    # pool_out_shape = (4, 4, 4)
    fc1_input_size = 784   # np.prod(pool_out_shape)
    fc1 = FullyConnected.FullyConnected(fc1_input_size, 256)
    fc2 = FullyConnected.FullyConnected(256, categories)

    net.layers.append(conv1)
    net.layers.append(Func.ReLU())
    net.layers.append(pool)
    net.layers.append(Flatten.Flatten())
    net.layers.append(fc1)
    net.layers.append(Func.ReLU())
    net.layers.append(fc2)

    data_generator = DataLoaders.DataGenerator(batch_size, dataset, shuffle=False)
    net.train(iteration, data_generator)
    plt.figure('Loss function')
    plt.plot(net.loss, '-x')
    plt.show()

    dataset_test = DataLoaders.Dataset(
        root_dir=r'.\test',
        train=False,
        test=True,
        transform=True)
    test_data = DataLoaders.DataGenerator(10, dataset_test, shuffle=False)
    accuracy = net.test(test_data)
    print('Test Accuracy: {:.4f}'.format(accuracy))

