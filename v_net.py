
import keras
from keras.layers import Add, Concatenate, Conv3D, UpSampling3D, Dense, Activation, Input
from keras import backend as K
from keras import Model

def Residual_block_3D(layer_in,depth=3,kernel_size=5,filters=None,activation='relu',name=None):
    l = Conv3D(filters,kernel_size,padding='same',activation=activation,name='{}_c0'.format(name))(layer_in)
    for i in range(1,depth):
        l = Conv3D(filters, kernel_size, padding='same',activation=activation,name='{}_c{}'.format(name,i))(l)
    o = Add()([layer_in, l])
    return o

def DownConv3D(layer_in, kernel_size=2, stride=2, filters=None,  activation='relu', name=None):
    stride = (stride, stride, stride)
    dc = Conv3D(filters, kernel_size, strides=stride, padding='valid', activation=activation, name='{}_dc0'.format(name))(layer_in)
    return dc

def UpConv3D(layer_in, size=(2,2,2), filters=None, activation='relu', name=None, data_format=None):
    u = UpSampling3D(size=size, data_format=data_format, name='{}_us0'.format(name))(layer_in)
    c = Conv3D(filters, 1, padding='same', activation=activation, name='{}_usc0'.format(name))(u)
    return c

def v_net(input_shape):
    x,y,z = input_shape

    in_layer = Input(batch_shape=(None, x, y, z, 1), dtype='float32', name='Input')
    
    # Step 0 - down
    conv_0 = Residual_block_3D(in_layer, depth=1, kernel_size=5, filters=16, name='resblock_0')
    downconv_0 = DownConv3D(conv_0, filters=32, name='downconv0')

    # Step 1 - down 
    conv_1 = Residual_block_3D(downconv_0, depth=2, kernel_size=5, filters=32, name='resblock_1')
    downconv_1 = DownConv3D(conv_1, filters=64, name='downconv1')

    # Step 2 - down
    conv_2 = Residual_block_3D(downconv_1, depth=3, kernel_size=5, filters=64, name='resblock_2')
    downconv_2 = DownConv3D(conv_2, filters=128, name='downconv2')

    # Step 3 - down
    conv_3 = Residual_block_3D(downconv_2, depth=3, kernel_size=5, filters=128, name='resblock_3')
    downconv_3 = DownConv3D(conv_3, filters=256, name='downconv3')

    # Step 4 - bottom
    conv_4 = Residual_block_3D(downconv_3, depth=3, kernel_size=5, filters=256, name='resblock_4')
    upconv_0 = UpConv3D(conv_4, filters=128, name='upconv0')

    # Step 5 - up
    concat0 = Concatenate()([conv_3,upconv_0])
    conv_5 = Residual_block_3D(concat0, depth=3, kernel_size=5, filters=256, name='resblock_5')
    upconv_1 = UpConv3D(conv_5, filters=64, name='upconv1')

    # Step 6 - up
    concat1 = Concatenate()([conv_2,upconv_1])
    conv_6 = Residual_block_3D(concat1, depth=3, kernel_size=5, filters=128, name='resblock_6')
    upconv_2 = UpConv3D(conv_6, filters=32, name='upconv2')

    # Step 7 - up
    concat2 = Concatenate()([conv_1,upconv_2])
    conv_7 = Residual_block_3D(concat2, depth=2, kernel_size=5, filters=64, name='resblock_7')
    upconv_3 = UpConv3D(conv_7, filters=16, name='upconv3')

    # Step 8 - up
    concat3 = Concatenate()([conv_0,upconv_3])
    conv_8 = Residual_block_3D(concat3, depth=1, kernel_size=5, filters=32, name='resblock_8')

    one_by_one = Conv3D(2, 1, strides=(1,1,1), padding='same',name='one_by_one', activation='softmax')(conv_8)
    
    return Model(inputs=in_layer, outputs=one_by_one)

def get_model_memory_usage(batch_size, model):
    import numpy as np

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

