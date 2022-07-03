import pickle

import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow import keras

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
model.fit(
    [ex,hx_x, hy_x],[ey,hx_y,hy_y],
    epochs=2,
    batch_size=32
)
print(model.trainable_weights)

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
# class Network(keras.Model):
#
#     def __init__(self,w):
#         super(Network, self).__init__()
#         self.block1=utils.MAIN_LAYER(w)
#     def __call__(self, E,Hx,Hy):
#         return self.block1(E,Hx,Hy)
#
# model=Network(1.)

#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
#block1=Network(1.)
#block1(ex,hx_x,hy_x)

from  utils import *




# def build_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Flatten(input_shape=(40,40)),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(5)
#     ])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(0.001),
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=tf.keras.metrics.MeanSquaredError(),
#     )
#     return model
#
#
# def train_model(model, x_train,y_train):
#     model.fit(
#         x_train,
#         y_train,
#         batch_size=10,
#         epochs=3
        # validation_data=(x_val, y_val),
    # )


#if __name__ == "__main__":
   # model = build_model()
    #train_model(model, np.random.rand(100,40,40),np.random.rand(100,5))




