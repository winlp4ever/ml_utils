import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.utils.np_utils import to_categorical
import imageio
import glob
from PIL import Image
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator


class VerrePlastiqueClassifier(object):
    def __init__(self, im_size, **transforms):
        self.im_size = im_size
        if data_augmentation:
            self.data_augment =
        self.net = Sequential([
            # feature extractor
            Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(*im_size, 3)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            # classifier
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax'),
        ])
        self.optim = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    def train(self, X_train, y_train, transforms=ImageDataGenerator(rescale=1./255), nb_epochs=10, batch_size=32):
        Y_train = to_categorical(y_train, num_classes=2)
        self.net.compile(optimizer=self.optim, loss='categorical_crossentropy', metrics=['accuracy'])
        self.net.fit(transforms.flow(X_train), Y_train, epochs=nb_epochs, batch_size=batch_size)

    def classify(self, image):
        im = Image.open(image)
        im = im.resize(self.im_size)
        im_arr = np.array(im) / 255.
        im_arr = im_arr[np.newaxis]
        return np.argmax(self.net.predict(im_arr)[0])


if __name__ == '__main__':
    im_dir = 'dogs-vs-cats/train'
    labels = {'dog': 0, 'cat': 1}
    from utils import to_nparray
    X_train, y_train = to_nparray(im_dir, size=(32, 32), verbose=True, labels=labels)
    print(np.sum(y_train))
    print(X_train.shape)
    dogcat = VerrePlastiqueClassifier((32, 32))
    print(dogcat.net.summary())

    transforms = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        channel_shift_range=9,
        fill_mode='nearest'
    )
    dogcat.train(X_train, y_train, transforms)
    print(dogcat.classify('dog.jpg'))
