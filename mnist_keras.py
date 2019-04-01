from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import regularizers, optimizers
import numpy as np
from sklearn.utils import shuffle as shuffle_
import pandas as pd
import os

# self-written file
# keras101 with mnist (format .csv)

# define network
conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'init': 'normal',
            'activation': 'relu'}
kwargs = {'kernel_regularizer': regularizers.l2(1e-5)} # add l2 regularizers
im_size = (28, 28)
num_classes = 10

model = Sequential([
        # feature extractor
        Conv2D(filters=32, input_shape=(*im_size, 1), **conv_kwargs, **kwargs),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, **conv_kwargs, **kwargs),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=64, **conv_kwargs, **kwargs),
        MaxPooling2D(pool_size=2),
        # classifier
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])

optim = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# load data
data_path = os.path.join('data', 'mnist')
data_train = pd.read_csv(os.path.join(data_path, 'mnist_train.csv'))
data_test = pd.read_csv(os.path.join(data_path, 'mnist_test.csv'))

def load_and_shuffle(data, im_size, shuffle=True):
    """Read csv into numpy and reshape to match image size.
    Works only for mnist, change image size for each use case
    """
    y, X = np.split(data.values, (1,), axis=1)
    Y = np.eye(num_classes)[y[:, 0]]
    X = X.reshape(-1, *im_size, 1)
    if shuffle:
        return shuffle_(X, Y)
    return X, Y

X_train, Y_train = load_and_shuffle(data_train, im_size)
X_test, Y_test = load_and_shuffle(data_test, im_size)

# train and evaluate model
nb_epochs = 10
batch_size = 128

model.fit(X_train, Y_train, epochs=nb_epochs, batch_size=batch_size, validation_data=(X_test, Y_test))
model.save_weights('mnist.h5')
