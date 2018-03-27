'''
classes to work with nifty images + contours
'''

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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
