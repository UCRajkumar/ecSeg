#!/usr/bin/env python3
import yaml, glob, os, sys, pickle
from pathlib import Path
config = open("config.yaml")
var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
Path("README.md").touch()

import subprocess as sp
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage as ndi
from scipy.stats import kurtosis
from skimage import *
import warnings
warnings.filterwarnings('ignore')
import tensorflow.python.util.deprecation as deprecation
from tqdm import tqdm
deprecation._PRINT_DEPRECATION_WARNINGS = False

ECSEG_I_MODEL = 'interseg'
ECSEG_C_MODEL = 'ecseg_c'

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
    inpath = var['inpath']
    fish_color = var['FISH_color'].lower()
    has_centromeric_probe = var['has_centromeric_probe']

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
    
    ecseg_i_label_map = {
        0: 'No-amp',
        1: 'EC-amp',
        2: 'HSR-amp',
    }
    
    ecseg_c_label_map = {
        0: 'No-amp',
        1: 'Focal-amp',
    }
    
    interseg_label_map = {
        ('No-amp', 'No-amp'): 'No-amp',
        ('No-amp', 'EC-amp'): 'No-amp',
        ('No-amp', 'HSR-amp'): 'No-amp',
        ('Focal-amp', 'No-amp'): 'No-amp',
        ('Focal-amp', 'EC-amp'): 'EC-amp',
        ('Focal-amp', 'HSR-amp'): 'HSR-amp',
    }
    
    image_paths = get_imgs(inpath)

    ecseg_i_model = load_model(ECSEG_I_MODEL)
    if has_centromeric_probe:
        ecseg_c_model = load_model(ECSEG_C_MODEL)
        
        
    stat_fish_results = pd.read_csv(os.path.join(inpath, 'annotated/stat_fish_lsq.csv'), keep_default_na=False, na_values=['_'])
                                    
    img_dict = {}
    dfs = []
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        
        stat_fish_results_img = stat_fish_results[stat_fish_results['image_name'] == path_split[1][:-4]]
        centromeric_quality_score = kurtosis(stat_fish_results_img[f"Avg fish intensity ({['red', 'green'][1-fish_index]})"]) if len(stat_fish_results) else float('inf')
        centromeric_quality_score_pass = centromeric_quality_score <= 3

        I = u16_to_u8(imread(i))
        seg_cells_path = os.path.join(path_split[0], 'annotated', path_split[1][:-4], f"{path_split[1][:-4]}_segmentation.tif")
        segmented_cells = io.imread(seg_cells_path)
        
        imheight, imwidth = segmented_cells.shape
        I = I[:imheight, :imwidth,:] #the labels is slightly truncated
        I = np.dstack([I[...,fish_index], I[...,1-fish_index], I[...,2]])
        
        segmented_cells = measure.label(segmented_cells, connectivity=None)
        regions = measure.regionprops(segmented_cells)

        centroids = []; pred_no_amp = []; pred_ec = []; pred_hsr = []; ecseg_i_label = []; names = []
        pred_no_focal_amp = []; pred_focal_amp = []; ecseg_c_label = []; interseg_label = []
        df = pd.DataFrame()

        # cell_dict = {}
        for region in tqdm(regions):
            center = region.centroid
            mask = (segmented_cells == region.label)
            temp = I * np.expand_dims(mask, -1)
            
            if (np.sum(temp[...,0])/np.sum(mask) < 12.75):
                interseg_label.append('No_Prediction (Low_TRGT_brightness)')
                ecseg_i_label.append('No_Prediction (Low_TRGT_brightness)')
                pred_no_amp.append('No_Prediction (Low_TRGT_brightness)')
                pred_ec.append('No_Prediction (Low_TRGT_brightness)')
                pred_hsr.append('No_Prediction (Low_TRGT_brightness)')
                
                if has_centromeric_probe:
                    ecseg_c_label.append('No_Prediction (Low_TRGT_brightness)')
                    pred_no_focal_amp.append('No_Prediction (Low_TRGT_brightness)')
                    pred_focal_amp.append('No_Prediction (Low_TRGT_brightness)')
                
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                names.append(path_split[-1][:-4])
                continue
            
            bb = region.bbox
            h = bb[2] - bb[0]; w = bb[3] - bb[1]
            if((h <= 256) & (w <= 256)):
                nuclei = temp[bb[0]:(bb[0] + min(256, h)), bb[1]:(bb[1]+ min(256, w))]
                p = np.expand_dims(resize(nuclei, (256, 256), preserve_range=True), 0).astype('uint8')
                ecseg_i_prediction = ecseg_i_model.predict(p[...,0])
                
                pred_no_amp_, pred_ec_, pred_hsr_  = ecseg_i_prediction[0]
                pred_no_amp.append(pred_no_amp_)
                pred_ec.append(pred_ec_)
                pred_hsr.append(pred_hsr_)
                
                ecseg_i_label_ = ecseg_i_label_map[np.argmax(ecseg_i_prediction[0])]
                ecseg_i_label.append(ecseg_i_label_)
                
                if has_centromeric_probe and p[...,1].max() > 10 and centromeric_quality_score_pass:
                    p = np.expand_dims(preprocess_ecseg_c(p[0]), 0)
                    ecseg_c_prediction = ecseg_c_model.predict(p)
                
                    pred_no_focal_amp_, pred_focal_amp_ = 1-ecseg_c_prediction[0,0],ecseg_c_prediction[0,0]
                    pred_no_focal_amp.append(pred_no_focal_amp_)
                    pred_focal_amp.append(pred_focal_amp_)

                    ecseg_c_label_ = ecseg_c_label_map[int(ecseg_c_prediction[0,0] > 0.5)]
                    ecseg_c_label.append(ecseg_c_label_)

                    interseg_label.append(interseg_label_map[(ecseg_c_label_, ecseg_i_label_)])
                else:
                    if has_centromeric_probe and not centromeric_quality_score_pass:
                        ecseg_c_label.append('No_Prediction (Failed Centromeric Quality Score)')
                        pred_no_focal_amp.append('No_Prediction (Failed Centromeric Quality Score)')
                        pred_focal_amp.append('No_Prediction (Failed Centromeric Quality Score)')
                    elif has_centromeric_probe and p[...,1].max() <= 10:
                        ecseg_c_label.append('No_Prediction (Low_CENT_Brightness)')
                        pred_no_focal_amp.append('No_Prediction (Low_CENT_Brightness)')
                        pred_focal_amp.append('No_Prediction (Low_CENT_Brightness)')
                    interseg_label.append(ecseg_i_label_)

                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                names.append(path_split[-1][:-4])
            else:
                nuclei = temp[bb[0]:(bb[0] + h), bb[1]:(bb[1]+ w)]
                patches = im2patches_overlap(nuclei)
                for p in patches:
                    names.append(path_split[-1][:-4])
                    centroids.append(str(int(center[0])) + '_' + str(int(center[1])))

                    if not p.any():
                        interseg_label.append('No_Prediction (Segmentation_Empty)')
                        ecseg_i_label.append('No_Prediction (Segmentation_Empty)')
                        pred_no_amp.append('No_Prediction (Segmentation_Empty)')
                        pred_ec.append('No_Prediction (Segmentation_Empty)')
                        pred_hsr.append('No_Prediction (Segmentation_Empty)')

                        if has_centromeric_probe:
                            ecseg_c_label.append('No_Prediction (Segmentation_Empty)')
                            pred_no_focal_amp.append('No_Prediction (Segmentation_Empty)')
                            pred_focal_amp.append('No_Prediction (Segmentation_Empty)')
                        continue
                    
                    p = np.expand_dims(p, 0)
                    ecseg_i_prediction = ecseg_i_model.predict(p[...,0])
                
                    pred_no_amp_, pred_ec_, pred_hsr_  = ecseg_i_prediction[0]
                    pred_no_amp.append(pred_no_amp_)
                    pred_ec.append(pred_ec_)
                    pred_hsr.append(pred_hsr_)

                    ecseg_i_label_ = ecseg_i_label_map[np.argmax(ecseg_i_prediction[0])]
                    ecseg_i_label.append(ecseg_i_label_)

                    if has_centromeric_probe and p[...,1].max() > 10 and centromeric_quality_score_pass:
                        p = np.expand_dims(preprocess_ecseg_c(p[0]), 0)
                        ecseg_c_prediction = ecseg_c_model.predict(p)
                        
                        pred_no_focal_amp_, pred_focal_amp_ = 1-ecseg_c_prediction[0,0],ecseg_c_prediction[0,0]
                        pred_no_focal_amp.append(pred_no_focal_amp_)
                        pred_focal_amp.append(pred_focal_amp_)

                        ecseg_c_label_ = ecseg_c_label_map[int(ecseg_c_prediction[0,0] > 0.5)]
                        ecseg_c_label.append(ecseg_c_label_)

                        interseg_label.append(interseg_label_map[(ecseg_c_label_, ecseg_i_label_)])
                    else:
                        if has_centromeric_probe and not centromeric_quality_score_pass:
                            ecseg_c_label.append('No_Prediction (Failed Centromeric Quality Score)')
                            pred_no_focal_amp.append('No_Prediction (Failed Centromeric Quality Score)')
                            pred_focal_amp.append('No_Prediction (Failed Centromeric Quality Score)')
                        elif has_centromeric_probe and p[...,1].max() <= 10:
                            ecseg_c_label.append('No_Prediction (Low_CENT_Brightness)')
                            pred_no_focal_amp.append('No_Prediction (Low_CENT_Brightness)')
                            pred_focal_amp.append('No_Prediction (Low_CENT_Brightness)')
                        interseg_label.append(ecseg_i_label_)

        
        df['image_name'] = np.array(names)
        df['nucleus_center'] = np.array(centroids)
        
        df['interSeg_label'] = interseg_label
        if has_centromeric_probe:
            df['ecSeg-c_label'] = ecseg_c_label
        df['ecSeg-i_label'] = ecseg_i_label
        
        dfs.append(df)
        # img_dict[i] = (cell_dict)

    current_git_commit = sp.run('git log -1 | head -1', shell=True, capture_output=True).stdout.decode().strip().split(' ')[-1]
    dfs = pd.concat(dfs)
    dfs.to_csv(os.path.join(path_split[0],  f'interphase_prediction_{fish_color}.csv'), index=False)
    
                                                  
    # df = pd.DataFrame(img_dict).reset_index()
    # df.to_csv(os.path.join(path_split[0],  'interphase_prediction.csv'), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])