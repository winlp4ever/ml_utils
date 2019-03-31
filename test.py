from keras.layers import Conv2D, Dense
from keras.models import Sequential
import numpy as np

model = Sequential([
        Dense(2, input_shape=(3,), weights=[np.array([[0, 0], [0, 1], [1, 0]])], use_bias=False)
    ])

print(model.layers[0].get_weights())
