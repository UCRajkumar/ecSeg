#!/usr/bin/env python3
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage as ndi
from skimage import *

import warnings
warnings.filterwarnings('ignore')

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['nuclei_fish']
    inpath = var['inpath']
    fish_color = var['FISH_color'].lower()
    sensitivity = var['color_sensitivity']
    bbox_min_score = var['min_score'] 
    nms_thresh = var['nms_threshold']
    resize_scale = var['scale_ratio']

    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
    else:
        if((fish_color != 'green') & (fish_color != 'red')):
            print("FISH_color can only be \"green\" or \"red\". Please update the config.yaml file accordingly.")
            sys.exit(2)
     

    if(os.path.exists(os.path.join(inpath, 'red'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'red'))

    if(os.path.exists(os.path.join(inpath, 'green'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'green'))

    if(os.path.exists(os.path.join(inpath, 'nuclei'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'nuclei'))

    image_paths = get_imgs(inpath)

    dfs = []
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        
        I = imread(i)
        red, green = split_FISH_channels(I, i, sensitivity)
        blue = I[:,:,2]

        if(fish_color == 'green'):
            fish = green
        else:
            fish = red

        if(isinstance(fish, np.ndarray) == False):
            continue

        with tf.Graph().as_default():
            segmented_cells = nuclei_segment(blue, path_split, bbox_min_score, nms_thresh, resize_scale)

        segmented_cells = measure.label(segmented_cells)
        regions = measure.regionprops(segmented_cells)

        fish_sizes = []; cell_sizes = []; centroids = []
        df = pd.DataFrame()
        for region in regions:
            cell = (segmented_cells == region.label)
            fish_sizes.append(count_cc(fish*cell)[1])
            cell_sizes.append(np.sum(cell==1))
            center = region.centroid
            centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
        df['nucleus center'] = centroids
        df['# of fish pixels'] = fish_sizes
        df['# of nuclei pixels'] = cell_sizes
        df.to_csv(os.path.join(path_split[0],  'nuclei', path_split[1][:-4]+'.csv'), index=False)


if __name__ == "__main__":
   main(sys.argv[1:])