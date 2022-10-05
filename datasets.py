# some code taken from https://github.com/e-hulten/made
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
import gzip
import pickle


class MNIST:
    """ 
    This is a version of: https://github.com/gpapamak/maf/blob/master/datasets/mnist.py, 
    adapted to work with Python 3.x and PyTorch. 
    """

    class Dataset:
        def __init__(self, data, logit, dequantize, rng):
            self.alpha = 1e-6
            x = (
                self._dequantize(data[0], rng) if dequantize else data[0]
            )  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x  # logit
            self.N = self.x.shape[0]  # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        def _logit_transform(self, x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            x = self.alpha + (1 - 2 * self.alpha) * x
            return np.log(x / (1.0 - x))

    def __init__(self, path, logit=True, dequantize=True):
        # load dataset
        with gzip.open(path, "rb") as f:
            train, val, test = pickle.load(f, encoding="latin1")

        rng = np.random.RandomState(42)
        self.train = self.Dataset(train, logit, dequantize, rng)
        self.val = self.Dataset(val, logit, dequantize, rng)
        self.test = self.Dataset(test, logit, dequantize, rng)

        self.n_dims = self.train.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims))] * 2

    def get_data_splits(self):
        return (
            torch.from_numpy(self.train.x),
            torch.from_numpy(self.val.x),
            torch.from_numpy(self.test.x),
        )

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if not data_split:
            raise ValueError("Invalid data split")
        if not pixel:
            data = data_split.x.flatten()
        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, density=True, color="lightblue")
        ax.set_yticklabels("")
        ax.set_yticks([])
        plt.show()

class CIFAR10:
    """
    The CIFAR-10 dataset.
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    alpha = 0.05

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, x, l, logit, flip, dequantize, rng):

            D = int(x.shape[1] / 3)                                 # number of pixels
            x = self._dequantize(x, rng) if dequantize else x  # dequantize
            x = self._logit_transform(x) if logit else x       # logit
            x = self._flip_augmentation(x) if flip else x      # flip
            self.x = x                                         # pixel values
            self.r = self.x[:, :D]                             # red component
            self.g = self.x[:, D:2*D]                          # green component
            self.b = self.x[:, 2*D:]                           # blue component
            self.N = self.x.shape[0]                           # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return (x + rng.rand(*x.shape).astype(np.float32)) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            temp = CIFAR10.alpha + (1 - 2*CIFAR10.alpha) * x
            return np.log(temp / (1.0 - temp))

        @staticmethod
        def _flip_augmentation(x):
            """
            Augments dataset x with horizontal flips.
            """
            D = x.shape[1] / 3
            I = int(np.sqrt(D))
            r = x[:,    :D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            g = x[:, D:2*D].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            b = x[:,  2*D:].reshape([-1, I, I])[:, :, ::-1].reshape([-1, D])
            x_flip = np.hstack([r, g, b])
            return np.vstack([x, x_flip])

    def __init__(self, path = 'cifar10/', logit=False, flip=False, dequantize=True):

        rng = np.random.RandomState(42)

        # load train batches
        x = []
        l = []
        for i in range(1, 6):
            f = open(path + 'data_batch_' + str(i), 'rb')
            dict = pickle.load(f, encoding='latin1')
            x.append(dict['data'])
            l.append(dict['labels'])
            f.close()
        x = np.concatenate(x, axis=0)
        l = np.concatenate(l, axis=0)

        # use part of the train batches for validation
        split = int(0.9 * x.shape[0])
        self.train = self.Data(x[:split], l[:split], logit, flip, dequantize, rng)
        self.val = self.Data(x[split:], l[split:], logit, False, dequantize, rng)

        # load test batch
        f = open(path + 'test_batch', 'rb')
        dict = pickle.load(f, encoding='latin1')
        x = dict['data']
        l = np.array(dict['labels'])
        f.close()
        self.test = self.Data(x, l, logit, False, dequantize, rng)

        self.n_dims = self.train.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims / 3))] * 2 + [3]

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        # get split
        data = getattr(self, split, None)
        if data is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data_r = data.r.flatten()
            data_g = data.g.flatten()
            data_b = data.b.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data_r = data.r[:, idx]
            data_g = data.g[:, idx]
            data_b = data.b[:, idx]

        n_bins = int(np.sqrt(data.N))
        fig, axs = plt.subplots(3, 1)
        for ax, d, t in zip(axs, [data_r, data_g, data_b], ['red', 'green', 'blue']):
            ax.hist(d, n_bins, normed=True)
            ax.set_title(t)
        plt.show()