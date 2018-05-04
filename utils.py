'''
classes to work with nifty images + contours
'''

import numpy as np
import nibabel as nib
# import matplotlib.pyplot as plt

import os 
import h5py
import keras.utils
from keras.models import *
from keras import regularizers
from keras.layers import Activation, Input, Dense, Flatten, merge, BatchNormalization, Conv2DTranspose,  Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
import keras.initializers
from keras.optimizers import *
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import skimage.transform
import skimage.exposure

import matplotlib as mpl
# mpl.use('Agg')

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

def make_Ridge(nrow,ncol,learning_rate=1e-5,load_weights=None,lambd=0.01):
    inputs = Input((nrow, ncol, 1))
    flat = Flatten()(inputs)
    flat = BatchNormalization()(flat)
    node1 = Dense(1, activation="linear",kernel_regularizer=regularizers.l2(lambd),bias_regularizer=regularizers.l2(lambd))(flat)
    out = Activation('sigmoid')(node1)
    model = Model(input = inputs, output = out)
    if not load_weights is None:
        model.load_weights(load_weights)
    model.compile(optimizer = Adam(lr = learning_rate), loss = "binary_crossentropy", metrics = ["binary_accuracy"])
    return model

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

def make_unet_2(nrow, ncol, learning_rate=1e-4, load_weights=None):
    '''
    UNet as Using Conv2DTranspose instead of Upsampling
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
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = LeakyReLU()(conv5)
    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)

    # up6 = Conv2DTranspose(512,3,activation=None,kernel_initializer='he_normal')(conv5)
    up6 = Conv2D(512, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = LeakyReLU()(up6)
    merge6 = merge([conv4,up6], mode = 'concat', concat_axis = 3)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = LeakyReLU()(conv6)

    # up7 = Conv2DTranspose(256,3,activation=None,kernel_initializer='he_normal')(conv6)
    up7 = Conv2D(256, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    up7 = LeakyReLU()(up7)
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = LeakyReLU()(conv7)

    # up8 = Conv2DTranspose(128,3,activation=None,kernel_initializer='he_normal')(conv7)
    up8 = Conv2D(128, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    up8 = LeakyReLU()(up8)
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = LeakyReLU()(conv8)

    #  up9 = Conv2DTranspose(64,3,activation=None,kernel_initializer='he_normal')(conv8)
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
    
def plot_prediction_new(x,y_pred,odir,slice):
    import matplotlib.pyplot as plt
    outfile=odir+'/s'+str(slice)+'.pdf'
    fig, (a1, a2) = plt.subplots(1,2)
    a1.imshow(x[slice][:,:,0],cmap='gray')
    a2.imshow(y_pred[slice][:,:,0],cmap='gray')
    fig.savefig(outfile, dpi=300)
    plt.close()

def plot_prediction_new_withprob(x,y_pred,y_prob,problabels,odir,slice):
    import matplotlib.pyplot as plt
    outfile=odir+'/s'+str(slice)+'.pdf'
    fig, (a1, a2, a3) = plt.subplots(1,3)
    a1.imshow(x[slice][:,:,0],cmap='gray')
    a2.imshow(y_pred[slice][:,:,0],cmap='gray')
    p = y_prob[slice,:]
    x = np.arange(len(p))
    a3.bar(x = x, height=p)
    a3.set_xticks(x)
    a3.set_xticklabels(problabels)
    a3.set_ylim([0,1])
    fig.savefig(outfile, dpi=300)
    plt.close()

def train_Ridge(img_train,mask_train,img_test=None,mask_test=None,batch_size=32,validation_split=0.05,learning_rate=1e-4,epochs=10,shuffle=True,odir="./",plot=False,lambd=0.5):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    model = make_Ridge(512,512,learning_rate=learning_rate,lambd=lambd)
    model_checkpoint = ModelCheckpoint(odir+'/Ridge.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    csv_logger = CSVLogger(odir+'/training_Ridge.log')
    mask_train = np.array([np.sum(x) > 20 for x in mask_train]).astype('float32')
    model.fit(img_train, mask_train, batch_size=batch_size, epochs=epochs, verbose=1,validation_split=validation_split, shuffle=shuffle, callbacks=[model_checkpoint,csv_logger])
    with open(odir+'/params.txt','w') as outfile:
        outfile.write("n_train={0}\nn_test={1}\nbatch_size={2}\nvalidation_split={3}\nlearning_rate={4}\nepochs={5}\nshuffle={6}\nlambda={7}".format(len(img_train),len(img_test),batch_size,validation_split, learning_rate, epochs, shuffle,lambd))
    if not img_test is None:
        y_pred = model.predict(img_test, batch_size=1, verbose=1)
        np.save(odir+'/x_test.npy',img_test)
        np.save(odir+'/y_test.npy',mask_test)
        np.save(odir+'/y_pred_test.npy', y_pred)
        if plot:
            None
    return model

def train_convnet(img_train,mask_train,img_test=None,mask_test=None,batch_size=5,validation_split=0.05,learning_rate=1e-4,epochs=10,shuffle=True,odir="./",plot=False,dropout_rate=0.5):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    model = make_convnet(512,512,learning_rate=learning_rate,dropout_rate=dropout_rate)
    model_checkpoint = ModelCheckpoint(odir+'/convnet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    csv_logger = CSVLogger(odir+'/training_convnet.log')
    mask_train = np.array([np.sum(x) > 20 for x in mask_train]).astype('float32')
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
            if not os.path.isdir(odir+'/plots'):
                os.mkdir(odir+'/plots')
            for i in list(range(0,len(img_test))):
                plot_prediction(img_test,mask_test,y_pred,odir+'/plots',i)
    return model


def dice_coef(y_true, y_pred):
    # similar to intersection over union
    '''
    props to https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def generalised_dice_loss(y_true, y_pred):
    # written for the two-class case
    y_true_f = K.cast(K.flatten(y_true), 'float32')
    y_pred_f = K.cast(K.flatten(y_pred), 'float32')
    y_true_bg_f = K.cast(K.equal(K.flatten(y_true),0), 'float32')
    y_pred_bg_f = 1 - y_pred_f
    w_fg = 1 / (K.square(K.sum(y_true_f))+1)
    w_bg = 1 / (K.square(K.sum(y_true_bg_f))+1)
    numerator = w_fg * K.sum( y_true_f * y_pred_f ) + w_bg * K.sum( y_true_bg_f * y_pred_bg_f )
    denominator = w_fg * K.sum( y_pred_f + y_true_f ) + w_bg * K.sum( y_pred_bg_f + y_true_bg_f )
    return (1. - 2. * numerator/denominator)

def generalised_dice_loss_2(y_true, y_pred):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    shapes = K.shape(y_pred)
    vol = K.cast(shapes[1]*shapes[2]*shapes[3], 'float32')
    # flatten everything except channels (i.e. class probabilities)
    y_true = K.reshape( y_true, (shapes[0], shapes[1]*shapes[2]*shapes[3], shapes[4]))
    y_pred = K.reshape( y_pred, (shapes[0], shapes[1]*shapes[2]*shapes[3], shapes[4]))

    # equal weights for each class:
    # weights = 1. - ((K.sum(y_true, 2)+1.) / vol)
    
    # weights like in paper:
    weights = 1. / K.square((K.sum(y_true, 1)+1.))
    
    overlaps = K.sum( y_pred * y_true, axis = 1)
    total = K.sum(y_pred + y_true, axis = 1)
    
    numerator = -2. * (weights * overlaps)
    denominator = total * weights
    return 1. + K.sum(numerator, -1) / K.sum(denominator, -1)
    

def dice_coef_foreground(y_true, y_pred):
    smooth = 1.
    shapes = K.shape(y_pred)
    # flatten everything except channels (i.e. class probabilities)
    y_true = K.reshape( y_true, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    y_pred = K.reshape( y_pred, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    _, y_true = tf.split(y_true,2,axis=-1)
    _, y_pred = tf.split(y_pred,2,axis=-1)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_background(y_true, y_pred):
    smooth = 1.
    shapes = K.shape(y_pred)
    # flatten everything except channels (i.e. class probabilities)
    y_true = K.reshape( y_true, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    y_pred = K.reshape( y_pred, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    y_true, _ = tf.split(y_true,2,axis=-1)
    y_pred, _ = tf.split(y_pred,2,axis=-1)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + smooth)

def dice_coef_foreground_loss(y_true, y_pred):
    smooth = 1.
    shapes = K.shape(y_pred)
    # flatten everything except channels (i.e. class probabilities)
    y_true = K.reshape( y_true, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    y_pred = K.reshape( y_pred, (shapes[0]*shapes[1]*shapes[2]*shapes[3], shapes[4]))
    _, y_true = tf.split(y_true,2,axis=-1)
    _, y_pred = tf.split(y_pred,2,axis=-1)
    intersection = K.sum(y_true * y_pred)
    return (-2. * intersection) / (K.sum(y_true) + K.sum(y_pred) + smooth)

class Scan_DataAugmenter:

    def __init__(self, contrast_prob=0.5, zoom_prob=0.5,rotation_prob=0.5, flip_prob=0.5, shift_prob=0.5, affine_prob=0.5):
        self.rotp = rotation_prob
        self.zp = zoom_prob
        self.fp = flip_prob
        self.sp = shift_prob
        self.affp = affine_prob
        self.conp = contrast_prob

    def _flip(self,img,horizontal=False, vertical=False):
        if horizontal:
            if vertical:
                return img[::-1,::-1]
            else:
                return img[:,::-1]
        elif vertical:
            return img[::-1,:]
        else:
            return img

    def _rotate(self, img, angle):
        return skimage.transform.rotate(img, angle, mode='edge')

    def _shift(self, img, l=0, r=0, t=0, b=0):
        x, y = img.shape
        img = img[b:(x-t),r:(y-l)]
        img = np.pad(img, [(t,b),(l,r)],'reflect')
        return img

    def _zoom(self, img, scale=1.1):
        x, y = img.shape
        img_zoom = skimage.transform.rescale(img, scale)
        newx, newy = img_zoom.shape
        diffx = (newx - x) // 2 
        diffy = (newy - y) // 2
        return img_zoom[diffx:(newx-diffx),diffy:(newy-diffy)]

    def _contrast(self, img, nbins=2048, clip_limit=None, kernel_size=(128,128)):
        img=skimage.exposure.equalize_adapthist(img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
        return img

    def _affine(self, img):
        rows, cols = img.shape[0], img.shape[1]
        img = skimage.transform.warp(img, self.atform, output_shape=(rows, cols))
        return img

    def _set_affine_transform(self, rows, cols, n, slopes, intercepts):
        src_cols = np.linspace(0, cols, n)
        src_rows = np.linspace(0, rows, n)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        dst_rows = src[:, 1]
        dst_cols = src[:, 0] - np.linspace(0, n, src.shape[0])*np.repeat(slopes,n)+np.repeat(intercepts,n)
        dst = np.vstack([dst_cols, dst_rows]).T
        atform = skimage.transform.PiecewiseAffineTransform()
        atform.estimate(src, dst)
        self.atform = atform

    def update_augmenterfun(self, nrow=512, ncol=512):
        # local contrast enhancement parameters:
        clip_limit = np.random.rand()*0.002+0.001
        # rotation angle
        rot_angle = 5 * np.random.randn()
        # flips 
        flips = np.random.choice([0,1], 2, replace=True)
        scale_zoom = np.random.rand()*0.3+1.
        # shifts y/n
        s_hor, s_vert = np.random.choice([0,1], 2, replace=True)
        s_hor = np.random.rand()*25*s_hor
        s_vert = np.random.rand()*25*s_vert
        # shift direction
        d_hor, d_vert = np.random.choice([0,1], 2, replace=True)

        # slopes for affine transform:
        slopes = np.random.randn(5)*3
        # intercepts for affine transform:
        intercepts = np.random.randn(5)*5

        # should we perform these operations?
        rot = self.rotp >= np.random.rand()
        zoom = self.zp >= np.random.rand()
        flip = self.fp >= np.random.rand()
        shift = self.sp >= np.random.rand()
        affine = self.affp >= np.random.rand()
        contr = self.conp >= np.random.rand()

        def transformerfun(img, mask=False, nrow=nrow, ncol=ncol, clip_limit=clip_limit, scale_zoom = scale_zoom, rot_angle=rot_angle, flips=flips, s_hor=s_hor, s_vert=s_vert, d_hor=d_hor, d_vert=d_vert, slopes=slopes, intercepts=intercepts):
            if np.sum(img) <= 2:
                return img

            if contr and not mask:
                img = self._contrast(img, clip_limit=clip_limit)
            if zoom:
                img = self._zoom(img, scale_zoom)
            if rot:
                img = self._rotate(img, rot_angle)
            if flip:
                self._flip(img, False, vertical=flips[1]==1)
                # img = self._flip(img, horizontal=flips[0]==1, vertical=flips[1]==1)
            if shift:
                lrtb = (s_hor*(d_hor==0),s_hor*(d_hor==1),s_vert*(d_vert==0),s_vert*(d_vert==1))
                l, r, t, b = map(int, lrtb)
                img = self._shift(img, l=l, r=r, t=t, b=b)
            if affine:
                self._set_affine_transform(nrow, ncol, 5, slopes, intercepts)
                img = self._affine(img)
            return img
        
        self.augment = transformerfun
            
class Scan_DataGenerator(keras.utils.Sequence):
    def __init__(self, h5file, i, batch_size=1, data_x='imgdata/t1', data_y='contours/mask', shuffle=False, data_augmentor=None, crop=(0,0), target_dim=(256,256,32), yc=2):
        self.i = i
        self.h5file = h5file
        self.batch_size = batch_size
        self.data_x = data_x
        self.data_y = data_y
        self.shuffle=False
        self.data_augmenter = data_augmentor
        self.crop = crop
        self.target_dim = target_dim
        self.yc = yc
        self.on_epoch_end()

    def __len__(self):
        return len(self.index) // self.batch_size

    def on_epoch_end(self):
        index = np.arange(len(self.i))
        if self.shuffle:
            np.random.shuffle(index)
        self.index = index
        if not self.data_augmenter is None:
            self.data_augmenter.update_augmenterfun(nrow=self.target_dim[0], ncol=self.target_dim[1])

    def __getitem__(self, index):
        indexes = self.index[ index*self.batch_size : (index+1)*self.batch_size ]
        indexes = self.i[indexes]
        indexes = list(np.sort(indexes))
        X, Y = self.__data_generation(indexes)
        return X, Y

    def __data_generation(self, indexes):
        X = None
        Y = None
        with h5py.File(self.h5file,'r') as f:
            X = f[self.data_x][indexes]
            Y = f[self.data_y][indexes]
        X = X.astype('float32')
        Y = Y.astype('float32')
        X /= np.max(X)
        if self.crop != (0,0):
            X = X[:,self.crop[0]:(X.shape[1]-self.crop[0]),self.crop[1]:(X.shape[2]-self.crop[1]),:]
            Y = Y[:,self.crop[0]:(Y.shape[1]-self.crop[0]),self.crop[1]:(Y.shape[2]-self.crop[1]),:]
        if not self.target_dim is None:
            target_dim = self.target_dim
            X_new = np.empty([X.shape[0],target_dim[0],target_dim[1],X.shape[3]])
            Y_new = np.empty([X.shape[0],target_dim[0],target_dim[1],X.shape[3]])
            for i in range(X.shape[0]):
                for j in range(X.shape[3]):
                    X_new[i,:,:,j] = skimage.transform.resize(X[i,:,:,j],(target_dim[0],target_dim[1]))
                    Y_new[i,:,:,j] = skimage.transform.resize(Y[i,:,:,j],(target_dim[0],target_dim[1]))
            if target_dim[2] != X.shape[3]:
                diff = (target_dim[2] - X.shape[3]) / 2
                if diff > 0:
                    pad_top = np.floor(diff).astype('int')
                    pad_bottom = np.ceil(diff).astype('int')
                    X_new = np.pad(X_new, [(0,0),(0,0),(0,0),(pad_top,pad_bottom)], mode='constant')
                    Y_new = np.pad(Y_new, [(0,0),(0,0),(0,0),(pad_top,pad_bottom)], mode='constant', constant_values=0.)
                if diff < 0:
                    crop_top = abs(np.floor(diff).astype('int'))+2
                    crop_bottom = abs(np.floor(diff).astype('int'))-2
                    X_new = X_new[:,:,:,crop_bottom:(X.shape[3]-crop_top)]
                    Y_new = Y_new[:,:,:,crop_bottom:(X.shape[3]-crop_top)]
            X = X_new
            Y = Y_new

        if not self.data_augmenter is None:
            for i in range(X.shape[0]):
                self.data_augmenter.update_augmenterfun(X.shape[1],X.shape[2])
                for j in range(X.shape[3]):
                    X[i,:,:,j] = self.data_augmenter.augment(X[i,:,:,j])
                    Y[i,:,:,j] = self.data_augmenter.augment(Y[i,:,:,j], mask = True)
        else:
            None
        # change format to batch, x, y, z, channel
        X = X[np.newaxis, :, :, :, :]
        X = X.transpose((1,2,3,4,0))
        Y = Y[np.newaxis, :, :, :, :]
        Y = Y.transpose((1,2,3,4,0))
        if self.yc == 2:
            Y=np.pad(Y, ((0,0),(0,0),(0,0),(0,0),(1,0)), mode='constant', constant_values=0.)
            Y[:,:,:,:,0] = 1.-Y[:,:,:,:,1]
        return X, Y
            
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
        # returns tuple image, mask
        return img, m
    r = list()
    for s in i:
        r.append(liam(s))
    return r
