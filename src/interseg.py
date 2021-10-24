#!/usr/bin/env python3
################## IN PROGRESS #################
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors

META_MODEL = 'metaseg.h5'
INTER_MODEL = 'interseg.h5'
CELL_THRESHOLD = 0.5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
    inpath = var['inpath']
    fish_index = var['FISH_color']

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

    #meta_model = load_model(META_MODEL)
    inter_model = load_model(INTER_MODEL)
    image_paths = get_imgs(inpath)
    print("Reading from: ", inpath)
    print("Identifying cells...")
    '''
    for i in image_paths:
        I = meta_segment(meta_model, i)
        path_split = os.path.split(i)
        outpath = os.path.join(path_split[0], 'labels', path_split[1].split('.')[0])
        print("Saving segmentation of cells: ", i, " to ", outpath)
        np.save(outpath, I)
    '''
    print("Classifying images...")
    for i in image_paths: 
        clasification = inter_classify(inter_model, i, fish_index, CELL_THRESHOLD)
        print(i, classification)

if __name__ == "__main__":
   main(sys.argv[1:])
