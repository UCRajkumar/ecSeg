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
    sensitivity = var['color_sensitivity']
    bbox_min_score = var['min_score'] 
    nms_thresh = var['nms_threshold']
    resize_scale = var['scale_ratio']
    nuclei_size_t = var['nuclei_size_T']

    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
     

    if(os.path.exists(os.path.join(inpath, 'red'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'red'))

    if(os.path.exists(os.path.join(inpath, 'green'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'green'))

    if(os.path.exists(os.path.join(inpath, 'dapi'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'dapi'))

    if(os.path.exists(os.path.join(inpath, 'nuclei'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'nuclei'))

    image_paths = get_imgs(inpath)
    first_fish = 'green'
    second_fish = 'red'

    with tf.Graph().as_default():
        sess1, sess2, pred_masks, train_initial, pred_masks_watershed, resize_scale = load_nuset(bbox_min_score, nms_thresh, resize_scale)

        dfs = []
        for i in image_paths:
            path_split = os.path.split(i)
            print("Processing image: ", i)
            
            I = u16_to_u8(imread(i))
            red, green = split_FISH_channels(I, i, sensitivity)
            blue = I[:,:,2]
            cv2.imwrite(os.path.join(path_split[0], 'dapi', path_split[1]), cv2.bitwise_not(blue))

            fish = green
            fish2 = red

            if(isinstance(fish, np.ndarray) == False):
                continue

            segmented_cells = nuclei_segment(blue, path_split, 
            resize_scale, sess1, sess2, pred_masks, train_initial, pred_masks_watershed, nuclei_size_t)
            
            imheight, imwidth = segmented_cells.shape
            fish = fish[:imheight, :imwidth]
            fish2 = fish2[:imheight, :imwidth]

            segmented_cells = measure.label(segmented_cells)
            regions = measure.regionprops(segmented_cells)

            names = []
            fish_sizes = []; cell_sizes = []; centroids = []; fish_blobs = []
            fish_sizes2 = []; fish_blobs2 = []

            df = pd.DataFrame()
            print('Number of regions: ', len(regions))

            for region in regions:
                cell = (segmented_cells == region.label)
                num_fish_blobs, num_fish_pixels = count_cc(fish*cell)
                num_fish_blobs2, num_fish_pixels2 = count_cc(fish2*cell)

                fish_sizes.append(num_fish_pixels)
                fish_sizes2.append(num_fish_pixels2)

                fish_blobs.append(num_fish_blobs)
                fish_blobs2.append(num_fish_blobs2)
                
                cell_sizes.append(np.sum(cell==1))
                center = region.centroid
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                names.append(path_split[-1][:-4])

            df['image_name'] = names
            df['nucleus_center'] = centroids
            df['#_FISH_pixels (' + first_fish + ')'] = fish_sizes
            df['#_FISH_blobs (' + first_fish + ')'] = fish_blobs

            df['#_FISH_pixels (' + second_fish + ')'] = fish_sizes2
            df['#_FISH_blobs (' + second_fish + ')'] = fish_blobs2

            df['#_DAPI_pixels'] = cell_sizes
            dfs.append(df)
        
        dfs = pd.concat(dfs)
        dfs.to_csv(os.path.join(path_split[0],  'nuclei', 'nuclei_fish.csv'), index=False)
        sess1.close()
        sess2.close()


if __name__ == "__main__":
   main(sys.argv[1:])