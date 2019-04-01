from keras.layers import Conv2D, Dense
from keras.models import Sequential
import numpy as np

kwargs = {'use_bias': False}

model = Sequential([
        Dense(2, input_shape=(3,), weights=[np.array([[0, 0], [0, 1], [1, 0]])], **kwargs)
    ])

print(model.layers[0].get_weights())
