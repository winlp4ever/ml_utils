import numpy as np
from PIL import Image
from sklearn.utils import shuffle as shuffle_
import matplotlib.pyplot as plt
import glob
import os
import random


def one_hot_encoding(labels, num_classes=None):
    """Return one_hot_encoding of an array of class indices
    Parameters
    ----------
    labels : np.ndarray
        array of class indices.
    num_classes : None or int
        total number of classes. if None, then `num_classes` will be
        `np.max(labels) + 1`

    Returns
    -------
    np.ndarray
        Matrix of one hot encodings of labels

    """
    if num_classes is None:
        num_clases = np.max(labels) + 1
    return np.eye(num_classes)[labels]


def to_nparray(dir: str, size, channels_first=False, labels=None, shuffle=True, verbose=False):
    """Read an image dir and output all images into a numpy array with labels (optional)

    Parameters
    ----------
    dir : str
        Image dir path. There must be no subfolder within, only images
    size : int
        2-element tuple (height, width)
    channels_first : bool
        Decide whether the output array if of dims batch x channels x height x width (like pytorch)
        or batch x height x width x channels (like np)
    labels : None or dict
        if None, then no labels array is output, otherwise must be a dict of form
        {class_name: index, ...}
    shuffle : bool
        Decide whether shuffle data or not. Default True
    verbose : bool
        Decide whether logging or not

    Returns
    -------
    np.array or tuple
        if labels not None, return a tuple of images array and labels list,
        otherwise return only the images array
    """
    imgs = []
    if labels is not None:
        assert isinstance(labels, dict), 'if not None, labels must be a dict'
        y = []

    for fn in glob.glob(os.path.join(dir, '*')):
        if verbose:
            print('loading... {}'.format(os.path.basename(fn)), end='\r', flush=True)
        if labels is not None:
            for name in labels:
                if os.path.basename(fn).startswith(name):
                    y.append(labels[name])
                    break
        im = Image.open(fn)
        im = im.resize(size)
        im = np.array(im)
        if channels_first:
            im = np.transpose(im, (2, 0, 1))
        imgs.append(im)
    if verbose:
        print('\nDone.')
    X = np.stack(imgs, axis=0)
    if labels is not None:
        if shuffle:
            X, y = shuffle_(X, y)
        return X, y
    if shuffle:
        X = shuffle_(X)
    return X


if __name__ == '__main__':
    dir_name = 'dogs-vs-cats/train'
    labels = {'dog': 0, 'cat': 1}
    revers = {i: w for w, i in labels.items()}
    X, y = to_nparray(dir_name, size=(28, 28), labels=labels, verbose=True)
    print(X.shape)
    print(y[:5])
    im = X[12]
    print(im.shape)
    plt.imshow(im)
    plt.show()
    print("It's a : {}".format(revers[y[12]]))
