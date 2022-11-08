import pickle, gzip
import numpy as np
import sys, random

class Dataset:

    def __init__(self):
        train_set, valid_set, test_set = self.get_model_data()

        self.train_set = self.get_samples(train_set)
        self.valid_set = self.get_samples(valid_set)
        self.test_set = self.get_samples(test_set)

    def get_model_data(self):
        with gzip.open('./Data/mnist.pkl.gz', 'rb') as fd:
            train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
            
            np.set_printoptions(threshold=sys.maxsize)
            # print(train_set)
            # print(valid_set)
            # print(test_set)

        return (train_set, valid_set, test_set)

    def get_samples(self, dataset):
        labeled_set = []

        for i in range(len(dataset[0])):
            input = dataset[0][i]

            # label as hot-spot vector 
            label = np.zeros((10, 1))
            label[dataset[1][i]] = 1

            labeled_set.append([np.reshape(input, (28 * 28, 1)), label])

        return labeled_set

    def shuffle(self):
        self.train_set = random.sample(self.train_set, len(self.train_set))

    # retrieve n batches from the dataset
    def get_batches(self, n):
        batch_len = len(self.train_set) // n

        batches = [self.train_set[chunk:chunk+batch_len] \
            for chunk in range(0, len(self.train_set), batch_len)]

        return batches

