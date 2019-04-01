from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import regularizers, optimizers
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
        Conv2D(filters=32, input_shape=(*im_size, 3), **conv_kwargs, **kwargs),
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

def load_and_shuffle(data):
    print(data.info())

load_and_shuffle(data_train)
load_and_shuffle(data_test)
