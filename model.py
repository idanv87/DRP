import pickle

import keras.backend as K
import matplotlib.pyplot
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from constants import Constants




import utils

# Open the file in binary mode
with open('files/ex.pkl', 'rb') as file:
    ex = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('files/ey.pkl', 'rb') as file:
    ey = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('files/hx_x.pkl', 'rb') as file:
    hx_x = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('files/hy_x.pkl', 'rb') as file:
    hy_x = tf.cast(pickle.load(file),tf.dtypes.float32)
with open('files/hx_y.pkl', 'rb') as file:
    hx_y =tf.cast(pickle.load(file),tf.dtypes.float32)
with open('files/hy_y.pkl', 'rb') as file:
    hy_y = tf.cast(pickle.load(file),tf.dtypes.float32)
w=tf.Variable([3.,2.],trainable=True, dtype=tf.dtypes.float32, name='w')
E_input = keras.Input(shape=(Constants.N,Constants.N,1), name="E")
Hx_input = keras.Input(shape=(Constants.N-2,Constants.N-1,1), name="Hx")
Hy_input = keras.Input(shape=(Constants.N-1,Constants.N-2,1), name="Hy")
E_output,Hx_output, Hy_output =utils.MAIN_LAYER(w)(E_input, Hx_input, Hy_input)
model = keras.Model(
    inputs=[E_input, Hx_input, Hy_input],
    outputs=[E_output,Hx_output, Hy_output]
)
model.compile(
    optimizer=keras.optimizers.SGD(1e-3),
    loss= [keras.losses.MeanSquaredError(),keras.losses.MeanSquaredError(), keras.losses.MeanSquaredError()]
)
history=model.fit(
    [ex,hx_x, hy_x],[ey,hx_y,hy_y],
    epochs=2,
    batch_size=32, validation_split=0.2
)
print(model.trainable_weights)

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
#
print(history.history)
