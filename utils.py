import numpy as np
import tensorflow as tf
from tensorflow import keras


from constants import Constants


class MAIN_LAYER(keras.layers.Layer):
#d
    def __init__(self,w):
        super(MAIN_LAYER,self).__init__()
        # self.a=tf.Variable([1.])
        # self.b=tf.constant([2.])
        # self.c=tf.reshape(tf.concat([self.a, self.b],0),[1,2,1,1])
        # print(self.c)
        #self.w=w
        self.filter = tf.Variable(
           tf.constant([[1., -1., 1., 0], [0, -1., 1., 0], [1., -1., 1., 0]], shape=[3, 4, 1, 1]),trainable=True)
        self.pad1=tf.constant([[0, 0], [2, 2], [2, 2], [0,0]], shape=[4, 2])
        self.pad2=tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]], shape=[4, 2])
        self.pad3=tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]], shape=[4, 2])



    def amper(self, E, Hx, Hy):
        S1 = tf.pad(( Constants.DT / Constants.DX) * self.Dx(Hy,tf.transpose(self.filter, perm=[1, 0, 2, 3])), self.pad1)+ \
        tf.pad(self.Dx(Hy,tf.transpose(Constants.KERNEL_FORWARD, perm=[1, 0, 2, 3])),Constants.PADY_FORWARD)+ \
        tf.pad(self.Dx(Hy, tf.transpose(Constants.KERNEL_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADY_BACWARD)

        S2 = tf.pad((Constants.Z * Constants.DT / Constants.DY) * self.Dy(Hx,self.filter), self.pad1)+ \
        tf.pad(self.Dy(Hx, Constants.KERNEL_FORWARD), Constants.PADX_FORWARD)+ \
        tf.pad(self.Dy(Hx, Constants.KERNEL_BACKWARD), Constants.PADX_BACWARD)
        return (E + S1 - S2)

    def faraday(self, E, Hx, Hy):


        S3 = (Constants.DT / (Constants.Z * Constants.DY))*tf.pad(self.Dy(E,self.filter), self.pad2)+ \
        tf.pad(self.Dy(E, Constants.KERNEL_E_FORWARD), Constants.PADEX_FORWARD)[:,1:-1,:,:] + \
        tf.pad(self.Dy(E, Constants.KERNEL_E_BACKWARD), Constants.PADEX_BACKWARD)[:,1:-1,:,:]

        S4 = (Constants.DT / (Constants.Z * Constants.DX))*tf.pad(self.Dx(E,tf.transpose(self.filter, perm=[1, 0, 2, 3])), self.pad3)+ \
        tf.pad(self.Dx(E, tf.transpose(Constants.KERNEL_E_FORWARD, perm=[1, 0, 2, 3])), Constants.PADEY_FORWARD)[:,:,1:-1,:]  + \
        tf.pad(self.Dx(E, tf.transpose(Constants.KERNEL_E_BACKWARD, perm=[1, 0, 2, 3])), Constants.PADEY_BACKWARD)[:,:,1:-1,:]

        Ax = (Hx - S3)
        Ay = (Hy + S4)

        return Ax, Ay


    def Dy(self, B, kernel ):
        return tf.nn.conv2d(B, kernel, strides=1, padding='VALID')


    def Dx(self, B, kernel):
        return  tf.nn.conv2d(B, kernel, strides=1, padding='VALID')

    # def call(self, E,H):
    #     print(self.w)
    #     return H+E+self.w[0]**2, E

    def call(self, E, Hx, Hy ):
        E_n = self.amper(E, Hx, Hy)
        Hx_n, Hy_n = self.faraday(E_n, Hx, Hy)
        E_m = self.amper(E_n, Hx_n, Hy_n)
        Hx_m, Hy_m = self.faraday(E_m, Hx_n, Hy_n)
        print(Hx_m.shape)
        print(tf.concat([Hx_n,Hx_m],1).shape)
        return tf.concat([E_n,E_m],1), tf.concat([Hx_n,Hx_m],1), tf.concat([Hy_n,Hy_m],1)




def f_a(c, n, k1, k2):
    e = c * np.cos(c * n * Constants.DT) * (
            np.sin(Constants.PI * k1 * Constants.X) * np.sin(Constants.PI * k2 * Constants.Y) +
            np.sin(Constants.PI * k2 * Constants.X) * np.sin(
        Constants.PI * k1 * Constants.Y))

    hx = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            -Constants.PI * k2 * np.sin(Constants.PI * k1 * Constants.X) * np.cos(
        Constants.PI * k2 * (Constants.Y + Constants.DX / 2)) - Constants.PI * k1 * np.sin(
        Constants.PI * k2 * Constants.X) * np.cos(Constants.PI * k1 * (Constants.Y + Constants.DX / 2)))

    hy = np.sin(c * (Constants.DT / 2) * (2 * n + 1)) * (
            Constants.PI * k1 * np.cos(Constants.PI * k1 * (Constants.X + Constants.DX / 2)) * np.sin(
        Constants.PI * k2 * Constants.Y) + Constants.PI * k2 * np.cos(
        Constants.PI * k2 * (Constants.X + Constants.DX / 2)) * np.sin(Constants.PI * k1 * Constants.Y))

    return e, hx[1:-1, :-1], hy[:-1, 1:-1]
