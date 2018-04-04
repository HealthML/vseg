'''
classes to work with nifty images + contours
'''

import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt

import os 
import h5py
from keras.models import *
from keras.layers import Activation, Input, Dense, Flatten, merge, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import *
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


class mri_scan:
    '''
    class that stores t1, t2 and segmentation mask together
    to be expanded as needed....
    '''
    def __init__(self, t1=None, t2=None, c=None):
        self.loaded=False
        self.dims=[None, None]
        if not t1 is None:
            self.t1 = nib.load(t1)
            self.dims[0] = self.t1.header.get_data_shape()
        else:
            self.t1 = None
        if not t2 is None:
            self.t2 = nib.load(t2)
            self.dims[1] = self.t1.header.get_data_shape()
        else:
            self.t2 = None
        if not c is None:
            self.load_contours(c)

    def load_contours(self, contour):
        self.contour = nib.load(contour)
        cdims = self.contour.header.get_data_shape()
        for d in self.dims:
            if not d is None:
                assert d == cdims

    def get_data(self):
        self.t1_img = None
        self.t2_img = None
        self.contour_mask = None
        if not self.t1 is None:
            self.t1_img=self.t1.get_data()
        if not self.t2 is None:
            self.t2_img=self.t2.get_data()
        if not self.contour is None:
            self.contour_mask=self.contour.get_data()
        self.loaded=True

    def export(self):
        if self.loaded == False:
            self.get_data()
            self.loaded = True
        return np.array([self.t1_img, self.t2_img, self.contour_mask])

    def plot(self, t, s, contour=False):
        if t == 1:
            img = self.t1_img
        elif t == 2:
            img = self.t2_img
        else:
            return None
        img = img[:,:,s]
        plt.imshow(img, cmap="gray", origin="lower")
        if contour:
            c = self.contour_mask[:,:,s]
            c = np.ma.masked_where(c < 1, c)
            plt.imshow(c, cmap="hsv", alpha=0.6, origin="lower")
        plt.show()

def make_convnet(nrow,ncol,learning_rate=1e-5,load_weights=None,dropout_rate=0.2):
    inputs = Input((nrow, ncol,1))

    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)    
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    # conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    # conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    flat = Flatten()(conv5)
    flat = Dropout(dropout_rate)(flat)
    out = Dense(1,activation='sigmoid')(flat)

    model = Model(input = inputs, output = out)

    if not load_weights is None:
        model.load_weights(load_weights)

    model.compile(optimizer = Adam(lr = learning_rate), loss = "binary_crossentropy", metrics = ["binary_accuracy"])
    return model

def make_unet(nrow, ncol, learning_rate=1e-4, load_weights=None):
    '''
    UNet as implemented by https://github.com/zhixuhao/unet.git
    '''
    inputs = Input((nrow, ncol,1))

    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)    
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)
    # conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    # conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    # conv4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    # conv5 = Dropout(0.2)(conv5)

    up6 = Conv2D(512, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = LeakyReLU()(up6)
    merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU()(conv6)

    up7 = Conv2D(256, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = LeakyReLU()(up7)
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU()(conv7)

    up8 = Conv2D(128, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = LeakyReLU()(up8)
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU()(conv8)

    up9 = Conv2D(64, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = LeakyReLU()(up9)
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(2, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    if not load_weights is None:
        model.load_weights(load_weights)
    model.compile(optimizer = Adam(lr = learning_rate), loss = generalised_dice_loss, metrics = ['accuracy',dice_coef])

    return model

def plot_prediction(x,y,y_pred,odir,slice):
    import matplotlib.pyplot as plt
    outfile=odir+'/s'+str(slice)+'.pdf'
    fig, (a1, a2, a3) = plt.subplots(1,3)
    a1.imshow(x[slice][:,:,0],cmap='gray')
    a2.imshow(y[slice][:,:,0],cmap='gray')
    a3.imshow(y_pred[slice][:,:,0],cmap='gray')
    fig.savefig(outfile, dpi=300)
    plt.close()

def train_unet(img_train,mask_train,img_test=None,mask_test=None,batch_size=5,validation_split=0.05,learning_rate=1e-4,epochs=10,shuffle=True,odir="./",plot=False):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    model = make_unet(512,512,learning_rate=learning_rate)
    model_checkpoint = ModelCheckpoint(odir+'/unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    csv_logger = CSVLogger(odir+'/training_unet.log')
    model.fit(img_train, mask_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=validation_split, shuffle=shuffle, callbacks=[model_checkpoint,csv_logger])
    with open(odir+'/params.txt','w') as outfile:
        outfile.write("n_train={0}\nn_test={1}\nbatch_size={2}\nvalidation_split={3}\nlearning_rate={4}\nepochs={5}\nshuffle={6}\n".format(len(img_train),len(img_test),batch_size,validation_split, learning_rate, epochs, shuffle))
    if not img_test is None:
        y_pred = model.predict(img_test, batch_size=1, verbose=1)
        np.save(odir+'/x_test.npy',img_test)
        np.save(odir+'/y_test.npy',mask_test)
        np.save(odir+'/y_pred_test.npy', y_pred)
        if plot:
            if not os.isdir('plots'):
                os.mkdir('plots')
            for i in list(range(0,len(img_test))):
                plot_prediction(img_test,mask_test,y_pred,'plots',i)
    return model

def train_convnet(img_train,mask_train,img_test=None,mask_test=None,batch_size=5,validation_split=0.05,learning_rate=1e-4,epochs=10,shuffle=True,odir="./",plot=False,dropout_rate=0.5):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    model = make_convnet(512,512,learning_rate=learning_rate,dropout_rate=dropout_rate)
    model_checkpoint = ModelCheckpoint(odir+'/convnet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    csv_logger = CSVLogger(odir+'/training_convnet.log')
    mask_train = np.array([np.sum(x) > 0 for x in mask_train]).astype('float32')
    model.fit(img_train, mask_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=validation_split, shuffle=shuffle, callbacks=[model_checkpoint,csv_logger])
    with open(odir+'/params.txt','w') as outfile:
        outfile.write("n_train={0}\nn_test={1}\nbatch_size={2}\nvalidation_split={3}\nlearning_rate={4}\nepochs={5}\nshuffle={6}\ndropout_rate={7}".format(len(img_train),len(img_test),batch_size,validation_split, learning_rate, epochs, shuffle,dropout_rate))
    if not img_test is None:
        y_pred = model.predict(img_test, batch_size=1, verbose=1)
        np.save(odir+'/x_test.npy',img_test)
        np.save(odir+'/y_test.npy',mask_test)
        np.save(odir+'/y_pred_test.npy', y_pred)
        if plot:
            None
    return model

def dice_coef(y_true, y_pred):
    # similar to intersection over union
    '''
    props to https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    smooth = 1.
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def generalised_dice_loss(y_true, y_pred):
	# written for the two-class case
    y_true_f = keras.cast(keras.flatten(y_true), 'float32')
    y_pred_f = keras.cast(keras.flatten(y_pred), 'float32')
    y_true_bg_f = keras.cast(keras.equal(keras.flatten(y_true),0), 'float32')
    y_pred_bg_f = 1 - y_pred_f
    w_fg = 1 / (keras.square(keras.sum(y_true_f))+1)
    w_bg = 1 / (keras.square(keras.sum(y_true_bg_f))+1)
    numerator = w_fg * keras.sum( y_true_f * y_pred_f ) + w_bg * keras.sum( y_true_bg_f * y_pred_bg_f )
    denominator = w_fg * keras.sum( y_pred_f + y_true_f ) + w_bg * keras.sum( y_pred_bg_f + y_true_bg_f )
    return (1. - 2. * numerator/denominator)

def load_data_h5(i,path="./scans.h5",t=1):
    f = h5py.File(path,'r')
    def load_transform(ind,dataset):
        img = f[dataset][ind]
        img = np.concatenate(img,2)
        img = img[np.newaxis,:,:,:]
        img = img.transpose((3,1,2,0))
        return img
    def liam(ind):
        img = load_transform(ind,"imgdata/t"+str(t))
        img = img.astype('float32')
        img /= 2048
        m = load_transform(ind, "contours/mask")
        return np.array([img,m])
    r = list()
    for s in i:
        r.append(liam(s))
    return r


