from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from keras_radam import RAdam
from keras.layers import *
import math

def swish(inputs):
    return (K.sigmoid(inputs) * inputs)

def h_swish(inputs):
    return inputs * tf.nn.relu6(inputs + 3) / 6

def Mish(inputs):
    return inputs*(K.tanh(K.softplus(inputs)))
get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'h_swish': Activation(h_swish)})
get_custom_objects().update({'Mish': Activation(Mish)})


def wdcnn(input_x, filters, kernel_size, strides, conv_padding, pool_padding, pool_size, pool_stride,name,BatchNormal=False):
    """wdcnn层神经元
    input_x: 输入
    :param filters: 卷积核的数目，整数
    :param kernerl_size: 卷积核的尺寸，整数
    :param strides: 步长，整数
    :param conv_padding: 'same','valid'
    :param pool_padding: 'same','valid'
    :param pool_size: 池化层核尺寸，整数
    :param pool_stride:池化步幅度，整数
    :param BatchNormal: 是否Batchnormal，布尔值
    :return: model
    """
    x = Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=conv_padding, kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input_x)  # , kernel_regularizer=l2(1e-4)
    if BatchNormal:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=pool_size, strides=pool_stride,padding=pool_padding,name=name)(x)
    return x

def build_model(input_shape,num_classes):
  
    """input_sig = Input(input_shape)
    x = Conv1D(filters=32, kernel_size=64, strides=2, padding='same', kernel_initializer='he_normal')(input_sig)  # , kernel_regularizer=l2(1e-4)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = wdcnn(x, filters=32, kernel_size=5, strides=2, conv_padding='same',
              pool_padding='same', pool_size=2)
    
    x = wdcnn(x, filters=64, kernel_size=3, strides=1, conv_padding='same',
              pool_padding='same', pool_size=2)
              """
    input_sig = Input(input_shape)
    #x=Dropout(0.1)(input_sig)
    x = Conv1D(filters=16, kernel_size=64, strides=4, padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(
        input_sig)  # , kernel_regularizer=l2(1e-4) #strides=1
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=2,padding='valid',name='Pool1')(x)

    x = wdcnn(x, filters=32, kernel_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2,pool_stride=2,BatchNormal=False,name='Pool2')
  
    x = wdcnn(x, filters=64, kernel_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2,pool_stride=2,name='Pool3')

    x = wdcnn(x, filters=64, kernel_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2,pool_stride=2,BatchNormal=False,name='Pool4')

    
    x = wdcnn(x, filters=128, kernel_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2,pool_stride=2,BatchNormal=False,name='Pool5')

  
    x = Flatten()(x)

    #x = Dense(units=256, activation='relu', name="FC1")(x)
    x = Dense(units=100, activation='relu', name="FC1")(x)  # , kernel_regularizer=l2(1e-4)
    #x = GlobalAveragePooling1D(name="FC1")(x)

    output_Q = Dense(units=num_classes, activation='linear',name="Output_Q")(x)  # , kernel_regularizer=l2(1e-4)
    model = Model(input=input_sig, output=output_Q)
    return model







