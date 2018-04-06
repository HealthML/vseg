#!/usr/bin/env python

from utils import *
import numpy as np
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type",choices=['unet', 'convnet', 'linear'])
    parser.add_argument("prefix")
    args = parser.parse_args()
    return args.model_type, args.prefix

def main():
    data = load_data_h5([range(0,209),range(209,221)], path="./scans.h5", t=1)
    sums = np.array([np.sum(x) for x in data[0][1]])

    x = data[0][0][sums > 0]
    y = data[0][1][sums > 0]

    val_x = data[1][0]
    val_y = data[1][1]
    new_x = np.load("new_data_imgarray.npy") / 2048.
    
    model, prefix = get_args()

    if model == "unet":
        outdir='./{0}_unet/'.format(prefix)
        m_u = train_unet(x,y,val_x,val_y,learning_rate=2.5*1e-5,odir=outdir,plot=True,epochs=20)
        new_y_pred = m_u.predict(new_x,batch_size=1,verbose=1)
        np.save(outdir+'new_y_pred.npy',new_y_pred)
        os.mkdir(outdir+'new_y_pred_plot')
        for i in range(len(new_x)):
            plot_prediction_new(new_x, new_y_pred, outdir+'new_y_pred_plot', i)
    elif model == "convnet":
        m_c = train_convnet(data[0][0],data[0][1],data[1][0],data[1][1],learning_rate=1e-5, odir='./{0}_cnet/'.format(prefix), dropout_rate=0.4,plot=False)
    else:
        m_linear = train_Ridge(x,y,val_x,val_y,learning_rate=1e-5,lambd=0.1,odir='./{0}_ridge/'.format(prefix),plot=False)

if __name__ == "__main__":
    main()

