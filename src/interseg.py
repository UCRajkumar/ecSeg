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

INTER_MODEL = 'interseg'
CELL_THRESHOLD = 0.5
IMG_THRESHOLD = 0.5

def split_nuclei_segments(img, overlap = 75, scw = 256): 
    h, w = img.shape[:2]
    patches = []
    for i in range(0, math.ceil(h/scw)):
        min_row = i*scw
        max_row = min_row + scw
        if(max_row > h):
            if(max_row - h < overlap):
                continue
            else:
                max_row = h
                min_row = max_row - scw
        for j in range(0, math.ceil(w/scw)):
            min_col = j*scw
            max_col = min_col + scw
            if(max_col > w):
                if(max_row - h < overlap):
                    continue
                else:
                    max_col = w
                    min_col = max_col - scw
                    patches.append(img[min_row:max_row, min_col:max_col, :])
    return patches

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
    inpath = var['inpath']
    fish_color = var['FISH_color'].lower()

    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
    else:
        if((fish_color != 'green') & (fish_color != 'red')):
            print("FISH_color can only be \"green\" or \"red\". Please update the config.yaml file accordingly.")
            sys.exit(2)
     
    if(fish_color == 'green'):
        fish_index = 1

    if(fish_color == 'red'):
        fish_index = 0

    if(os.path.exists(os.path.join(inpath, 'annotated'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'annotated'))

    image_paths = get_imgs(inpath)

    model = load_model(INTER_MODEL)
    img_dict = {}
    dfs = []
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        
        I = u16_to_u8(imread(i))
        seg_cells_path = os.path.join(path_split[0], 'annotated', path_split[1][:-4], f"{path_split[1][:-4]}_segmentation.tif")
        segmented_cells = io.imread(seg_cells_path)
        
        imheight, imwidth = segmented_cells.shape
        I = I[:imheight, :imwidth, fish_index]

        segmented_cells = measure.label(segmented_cells)
        regions = measure.regionprops(segmented_cells)

        centroids = []; predictions = []; names = []
        df = pd.DataFrame()

        # cell_dict = {}
        for region in regions:
            center = region.centroid
            mask = (segmented_cells == region.label)
            temp = I * mask
            bb = region.bbox
            h = bb[2] - bb[0]; w = bb[3] - bb[1]
            if((h <= 256) & (w <= 256)):
                nuclei = temp[bb[0]:(bb[0] + min(256, h)), bb[1]:(bb[1]+ min(256, w))]
                cell_prediction = model.predict(np.expand_dims(resize(nuclei, (256, 256), preserve_range=True), 0))
                # cell_dict[str(int(center[0])) + '_' + str(int(center[1]))] = list(cell_prediction[0])
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                predictions.append(list(cell_prediction[0]))
                names.append(path_split[-1][:-4])
            else:
                nuclei = temp[bb[0]:(bb[0] + h), bb[1]:(bb[1]+ w)]
                patches = split_nuclei_segments(nuclei)
                for p in patches: 
                    cell_prediction = model.predict(np.expand_dims(p, 0))
                    # cell_dict[str(int(center[0])) + '_' + str(int(center[1]))] = list(cell_prediction[0])
                    centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                    predictions.append(list(cell_prediction[0]))
                    names.append(path_split[-1][:-4])
        
        df['image_name'] = np.array(names)
        df['nucleus_center'] = np.array(centroids)
        df['predictions'] = predictions
        dfs.append(df)
        # img_dict[i] = (cell_dict)

    dfs = pd.concat(dfs)
    dfs.to_csv(os.path.join(path_split[0],  'interphase_prediction.csv'), index=False)
    # df = pd.DataFrame(img_dict).reset_index()
    # df.to_csv(os.path.join(path_split[0],  'interphase_prediction.csv'), index=False)


if __name__ == "__main__":
   main(sys.argv[1:])
