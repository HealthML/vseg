#!/usr/bin/env python

'''
Loads all the images (t1/t2) and contours, dumps them to an hdf5 file
one chunk = one complete scan, i.e. all slices

this script loads all the scans into memory before writing them to disk
in the future this might be undesireable.
'''

from utils import mri_scan
import numpy as np
import h5py

t1_txt='scans_t1.txt'
t2_txt='scans_t2.txt'
c_txt='contours_corrected.txt'

with open(t1_txt, 'r') as f:
    t1_paths = [ l.rstrip() for l in f.readlines() ]

with open(t2_txt, 'r') as f:
    t2_paths = [ l.rstrip() for l in f.readlines() ]

with open(c_txt, 'r') as f:
    c_paths = [ l.rstrip() for l in f.readlines() ]

scans = [ mri_scan(t1_paths[i], t2_paths[i], c_paths[i]) for i in range(0,len(t1_paths)) ]

for i in range(0,len(scans)):
    scans[i].get_data()

f = h5py.File('scans.h5','w')

f.create_group('imgdata')
f.create_group('contours')

d = scans[0].dims[0]

imgdata_t1=f.create_dataset('imgdata/t1',shape=(len(scans),d[0],d[1],d[2]),dtype='i',chunks=(1,512,512,44))
imgdata_t2=f.create_dataset('imgdata/t2',shape=(len(scans),d[0],d[1],d[2]),dtype='i',chunks=(1,512,512,44))
imgdata_cm=f.create_dataset('contours/mask',shape=(len(scans),d[0],d[1],d[2]),dtype='i',chunks=(1,512,512,44))

scans = np.array( [ s.export() for s in scans ] )

imgdata_t1[:,:,:,:]=scans[:,0,:,:,:]
imgdata_t2[:,:,:,:]=scans[:,1,:,:,:]
imgdata_cm[:,:,:,:]=scans[:,2,:,:,:]

f.flush()
f.close()
