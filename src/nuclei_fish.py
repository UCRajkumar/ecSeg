#!/usr/bin/env python3
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage as ndi
from skimage import measure
from skimage.io import *
from skimage.color import *
from skimage.morphology import *
from skimage.filters.rank import *
from skimage.segmentation import *
from skimage import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['nuclei_fish']
    inpath = var['inpath']
    fish_color = var['FISH_color'].lower()
    sensitivity = var['color_sensitivity']
    
    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
    else:
        if(os.path.isdir(os.path.join(inpath,'labels')) == False):
            print("`labels` folder is missing in the input folder.")
            print("Please make sure metaseg was run on the input folder first. This will generate the labels folder.")
            sys.exit(2)
        if(os.path.isdir(os.path.join(inpath,'dapi')) == False):
            print("`dapi` folder is missing in the input folder.")
            print("Please make sure metaseg was run on the input folder first. This will generate the labels folder.")
            sys.exit(2)
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

    image_paths = get_imgs(inpath)

    dfs = []
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)

        red, green = split_FISH_channels(i, sensitivity)
        if(fish_color == 'green'):
            fish = green
        else:
            fish = red

        if(isinstance(fish, np.ndarray) == False):
            continue

        _, nuclei, chrom, ec = read_seg(i)

        distance = ndi.distance_transform_edt(nuclei)

        local_max_coords = feature.peak_local_max(distance, min_distance=7)
        local_max_mask = np.zeros(distance.shape, dtype=bool)
        local_max_mask[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max_mask)
        
        segmented_cells = segmentation.watershed(-distance, markers, mask=nuclei)
        fish_sizes = []
        cell_sizes = []
        df = pd.DataFrame()
        
        for j in np.unique(segmented_cells)[1:]:
            cell = segmented_cells==j
            fish_sizes.append(count_cc(fish*cell)[1])
            cell_sizes.append(np.sum(cell==1))
            # fig, ax = plt.subplots(figsize=(5, 5))
            # print(fish_sizes[-1], cell_sizes[-1])
            # ax.imshow(cell)
            # plt.show()
        df['# of fish pixels'] = fish_sizes
        df['# of pixels in nucleus'] = cell_sizes
        df['image'] = path_split[1]
        dfs.append(df)

        # fig, ax = plt.subplots(ncols=3, figsize=(15, 7))
        # ax[0].imshow(imread(i)[:,:,-1], cmap='gray')
        # ax[0].set_title('DAPI')
        # ax[0].axis('off')
        # ax[1].imshow(nuclei, cmap='gray')
        # ax[1].set_title('ecSeg prediction')
        # ax[1].axis('off')
        # ax[2].imshow(color.label2rgb(segmented_cells, bg_label=0))
        # ax[2].set_title('Prediction post-process')
        # ax[2].axis('off')
        # plt.show()

    df = pd.concat(dfs)
    df.to_csv(os.path.join(path_split[0],  'nuclei_fish.csv'), index=False)



if __name__ == "__main__":
   main(sys.argv[1:])