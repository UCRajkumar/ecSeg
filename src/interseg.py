#!/usr/bin/env python3
################## IN PROGRESS #################
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors

MODEL_NAME = 'metaseg.h5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
    inpath = var['inpath']

    #create folders  
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)

    if(os.path.exists(os.path.join(inpath, 'dapi'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'dapi'))

    if(os.path.exists(os.path.join(inpath, 'labels'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'labels'))

    model = load_model(MODEL_NAME)
    image_paths = get_imgs(inpath)
    print("Reading from: ", inpath)
    for i in image_paths:
        print("Processing image: ", i)
        
        I = segment(model, i, 'inter')
        cmap = colors.ListedColormap(['#386cb0', '#ffff99', '#7fc97f', '#f0027f'])
        path_split = os.path.split(i)
        outpath = os.path.join(path_split[0], 'labels', path_split[1].split('.')[0])
        
        print("Saving image: ", i, " to ", outpath)

        plt.imsave(outpath + '.png', I.astype('uint8'), cmap=cmap, vmin=0, vmax=4)
#         np.save(outpath, I)

if __name__ == "__main__":
   main(sys.argv[1:])
