import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D

class Conv2D_CK7x7(tf.keras.layers.Layer):
    def __init__(self, ch, name):
        super().__init__()
        self.c_path1 = Conv2D(ch, (3,3), padding='same')
        self.c_path2 = Conv2D(ch, (3,3), padding='same')
        self.c_path3 = Conv2D(ch, (3,3), padding='same')
        self.conv = Conv2D(ch, 7, padding='same', name=name)

    def call(self, inputs):
        x_path = self.c_path1(inputs)
        x_path = self.c_path2(x_path)
        x_path = self.c_path3(x_path)
        x = self.conv(inputs)
        return x+x_path
    
class Conv2D_CL(tf.keras.layers.Layer):
    def __init__(self, ch, name, ratio=2):
        super().__init__()
        self.c_1 = Conv2D(ch*ratio, (1,1), padding='same')
        self.c_2 = Conv2D(ch*ratio, (1,1), padding='same')
        self.c_3 = Conv2D(ch, (1,1), padding='same', name=name)

    def call(self, inputs):
        x = self.c_1(inputs)
        x = self.c_2(x)
        x = self.c_3(x)
        return x
    
def ck_reparam(ker1, ker2, bias1, bias2):
    new_w = np.pad(np.transpose(ker1, (2,0,1,3)),((0,0),(1,1),(1,1),(0,0)))
    new_w = tf.nn.conv2d(new_w, ker2[::-1,::-1], strides=[1, 1, 1, 1], padding='SAME')
    new_w = np.transpose(new_w, (1,2,0,3))
    b_sum = np.sum(ker2, axis=(0,1))
    new_b = np.zeros_like(bias2)
    for i in range(b_sum.shape[-1]):
        new_b[i] = np.sum(b_sum[:,i]*bias1)

    new_b += bias2
    return new_w, new_b

def cl_reparam(ker1, ker2, bias1, bias2):
    new_k = np.matmul(ker1[0,0], ker2[0,0])
    new_k = new_k[np.newaxis,np.newaxis]
    new_b = np.matmul(bias1[np.newaxis], ker2[0,0])[0]
    new_b += bias2
    return new_k, new_b

def convertWeightsCK(old_weights):
    w_12, b_12 = ck_reparam(old_weights[0], old_weights[2], old_weights[1], old_weights[3])
    w_123, b_123 = ck_reparam(w_12, old_weights[4], b_12, old_weights[5])
    w_1234, b_1234 = w_123+old_weights[6], b_123+old_weights[7]
    return [w_1234, b_1234]

def convertWeightsCL(old_weights):
    new_ker, new_bias = cl_reparam(old_weights[0], old_weights[2], old_weights[1], old_weights[3])
    new_ker, new_bias = cl_reparam(new_ker, old_weights[4], new_bias, old_weights[5])
    return [new_ker, new_bias]