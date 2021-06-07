from keras.activations import softmax
import numpy as np
import tensorflow.keras.backend as K

x = np.array([[1.1, 4.2, 3.1],[1.3, 0.5, 3.2]])
x = K.constant(x)

output = softmax(x, axis=1)

print(output)
