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
        return len(os.listdir(dir))
    
    def __getitem__(self, index):
        img_name = self.root_dir + '' + '.jpg'
        label = index
        
        image = Image.open(img_name)
        if self.transform:
            image = image.resize((64,64))
            image = self.normalize(image)
        return image, label

    def normalize(self, arr):
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
