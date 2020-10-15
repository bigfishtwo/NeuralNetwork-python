import numpy as np
from PIL import Image
import os

class Dataset:
    def __init__(self, root_dir, train, test, transform=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            train (bool): if True, apply datset in training procedure
            test (bool): if True, apply datset in test procedure
            transform (bool, optional): Optional transform to be applied
                on a sample.
        """
        # TODO: rewrite dataloader
        self.root_dir = root_dir
        self.train = train
        self.test = test
        self.transform = transform

    def __len__(self):
        dir_dogs = self.root_dir + '/dogs'
        dir_cats = self.root_dir + '/cats'
        # TODO: change number of images
        # return len(os.listdir(dir_dogs)) + len(os.listdir(dir_cats))
        return 100

    def __getitem__(self, index):
        start = 0
        if not self.train and not self.test:
            start += 2000
        if self.test:
            start += 2500
        i = np.random.randint(0, 2)
        if i == 0:
            img_name = self.root_dir + '/dogs/dog.' + str(index // 2 + start) + '.jpg'
            label = 0
        else:
            img_name = self.root_dir + '/cats/cat.' + str(index // 2 + start) + '.jpg'
            label = 1
        image = Image.open(img_name)
        if self.transform:
            image = image.resize((64,64))
            image = self.normalize(image)
        return image, label

    def normalize(self, arr):
        """
        Linear normalization
        http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
        """
        arr = np.array(arr).astype('float')
        # Do not touch the alpha channel
        for i in range(3):
            minval = arr[..., i].min()
            maxval = arr[..., i].max()
            if minval != maxval:
                # arr[..., i] -= minval
                # arr[..., i] *= (255.0 / (maxval - minval))
                arr[..., i] /= 255.0
        # arr = Image.fromarray(arr.astype('uint8'))
        return arr

class DataGenerator:
    def __init__(self,batch_size, dataset, shuffle):
        """
        generate batch-wise data
        :param batch_size: int, number of images in a batch
        :param dataset: class Dateset
        :param shuffle: bool, if True, shuffle the sequence of data
        """
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
    def batch_generator(self):
        start = 0
        sequence = np.arange(0,len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(sequence)
        while start+ self.batch_size <len(self.dataset):
            images = []
            labels = []
            for i in range(self.batch_size):
                img, lab = self.dataset[sequence[i+start]]
                images.append(np.asarray(img).transpose(2,0,1))
                labels.append(lab)
            start += self.batch_size
            images = np.array(images)
            labels = np.array(labels)
            yield images,labels

    def forward(self):
        return next(self.batch_generator())
