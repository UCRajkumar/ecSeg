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
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

INTER_MODEL = 'interseg'
CELL_THRESHOLD = 0.5
IMG_THRESHOLD = 0.5

def im2patches_overlap(img, overlap = 75, scw = 256): 
    h, w = img.shape[:2]
    patches = []
    for i in range(0, math.ceil(h/scw)):
        min_row = i*scw
        if(h < 256):
            max_row = h
        else:
            max_row = min_row + scw
            if(max_row > h):
                continue
        for j in range(0, math.ceil(w/scw)):
            min_col = j*scw
            if(w < 256):
                max_col = w
            else:
                max_col = min_col + scw
                if(max_col > w):
                    continue
            patches.append(resize(img[min_row:max_row, min_col:max_col], (256, 256), preserve_range=True).astype('uint8'))
    return patches

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
    Path("README.md").touch()
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
    
    label_map = {
        0: 'No-amp',
        1: 'EC-amp',
        2: 'HSR-amp',
    }
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

        segmented_cells = measure.label(segmented_cells, connectivity=None)
        regions = measure.regionprops(segmented_cells)

        centroids = []; pred_no_amp = []; pred_ec = []; pred_hsr = []; majority_label = []; names = []
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
                
                
                pred_no_amp_, pred_ec_, pred_hsr_  = cell_prediction[0]
                pred_no_amp.append(pred_no_amp_)
                pred_ec.append(pred_ec_)
                pred_hsr.append(pred_hsr_)
                majority_label_ = label_map[np.argmax(cell_prediction[0])]
                majority_label.append(majority_label_)
                
                names.append(path_split[-1][:-4])
            else:
                nuclei = temp[bb[0]:(bb[0] + h), bb[1]:(bb[1]+ w)]
                patches = im2patches_overlap(nuclei)
                for p in patches: 
                    cell_prediction = model.predict(np.expand_dims(p, 0))
                    # cell_dict[str(int(center[0])) + '_' + str(int(center[1]))] = list(cell_prediction[0])
                    centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                    pred_no_amp_, pred_ec_, pred_hsr_  = cell_prediction[0]
                    pred_no_amp.append(pred_no_amp_)
                    pred_ec.append(pred_ec_)
                    pred_hsr.append(pred_hsr_)
                    names.append(path_split[-1][:-4])
                    
                    majority_label_ = label_map[np.argmax(cell_prediction[0])]
                    majority_label.append(majority_label_)
        
        df['image_name'] = np.array(names)
        df['nucleus_center'] = np.array(centroids)
        
        df['Majority_label'] = majority_label
        df['pred_No-amp'] = pred_no_amp
        df['pred_EC-amp'] = pred_ec
        df['pred_HSR-amp'] = pred_hsr
        
        dfs.append(df)
        # img_dict[i] = (cell_dict)

    dfs = pd.concat(dfs)
    dfs.to_csv(os.path.join(path_split[0],  f'interphase_prediction_{fish_color}.csv'), index=False)
    # df = pd.DataFrame(img_dict).reset_index()
    # df.to_csv(os.path.join(path_split[0],  'interphase_prediction.csv'), index=False)


if __name__ == "__main__":
   main(sys.argv[1:])