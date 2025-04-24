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

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
    Path("README.md").touch()
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

    img_dict = {}
    dfs = []
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        
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
            
            if (np.sum(temp[...,0])/np.sum(mask) < 0.05):
                interseg_label.append('No_Prediction: Low_brightness')
                ecseg_i_label.append('No_Prediction: Low_brightness')
                pred_no_amp.append('No_Prediction: Low_brightness')
                pred_ec.append('No_Prediction: Low_brightness')
                pred_hsr.append('No_Prediction: Low_brightness')
                
                if has_centromeric_probe:
                    ecseg_c_label.append('No_Prediction: Low_brightness')
                    pred_no_focal_amp.append('No_Prediction: Low_brightness')
                    pred_focal_amp.append('No_Prediction: Low_brightness')
                
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                names.append(path_split[-1][:-4])
                continue
            
            bb = region.bbox
            h = bb[2] - bb[0]; w = bb[3] - bb[1]
            if((h <= 256) & (w <= 256)):
                nuclei = temp[bb[0]:(bb[0] + min(256, h)), bb[1]:(bb[1]+ min(256, w))]
                p = resize(nuclei, (256, 256), preserve_range=True)
                ecseg_i_prediction = ecseg_i_model.predict(np.expand_dims(p[...,0], 0))
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                
                pred_no_amp_, pred_ec_, pred_hsr_  = ecseg_i_prediction[0]
                pred_no_amp.append(pred_no_amp_)
                pred_ec.append(pred_ec_)
                pred_hsr.append(pred_hsr_)
                
                ecseg_i_label_ = ecseg_i_label_map[np.argmax(ecseg_i_prediction[0])]
                ecseg_i_label.append(ecseg_i_label_)
                
                if has_centromeric_probe:
                    p = tf.convert_to_tensor(np.expand_dims(ecseg_c_preprocess(p.astype('uint8')), 0), dtype=tf.float32)
                    ecseg_c_prediction = tf.nn.softmax(list(ecseg_c_model(**{'input.1': p}).values())[0])
                
                    pred_no_focal_amp_, pred_focal_amp_ = ecseg_c_prediction[0]
                    pred_no_focal_amp.append(pred_no_focal_amp_)
                    pred_focal_amp.append(pred_focal_amp_)

                    ecseg_c_label_ = ecseg_c_label_map[np.argmax(ecseg_c_prediction[0])]
                    ecseg_c_label.append(ecseg_c_label_)

                    interseg_label.append(interseg_label_map[(ecseg_c_label_, ecseg_i_label_)])
                else:
                    interseg_label.append(ecseg_i_label_)
                
                names.append(path_split[-1][:-4])
            else:
                nuclei = temp[bb[0]:(bb[0] + h), bb[1]:(bb[1]+ w)]
                patches = im2patches_overlap(nuclei)
                for p in patches: 
                    ecseg_i_prediction = ecseg_i_model.predict(np.expand_dims(p[...,0], 0))
                
                    centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                    pred_no_amp_, pred_ec_, pred_hsr_  = ecseg_i_prediction[0]
                    pred_no_amp.append(pred_no_amp_)
                    pred_ec.append(pred_ec_)
                    pred_hsr.append(pred_hsr_)

                    ecseg_i_label_ = ecseg_i_label_map[np.argmax(ecseg_i_prediction[0])]
                    ecseg_i_label.append(ecseg_i_label_)

                    if has_centromeric_probe:
                        p = tf.convert_to_tensor(np.expand_dims(ecseg_c_preprocess(p), 0), dtype=tf.float32)
                        ecseg_c_prediction = tf.nn.softmax(list(ecseg_c_model(**{'input.1': p}).values())[0])
                        
                        pred_no_focal_amp_, pred_focal_amp_ = ecseg_c_prediction[0]
                        pred_no_focal_amp.append(pred_no_focal_amp_)
                        pred_focal_amp.append(pred_focal_amp_)

                        ecseg_c_label_ = ecseg_c_label_map[np.argmax(ecseg_c_prediction[0])]
                        ecseg_c_label.append(ecseg_c_label_)

                        interseg_label.append(interseg_label_map[(ecseg_c_label_, ecseg_i_label_)])
                    else:
                        interseg_label.append(ecseg_i_label_)
                    names.append(path_split[-1][:-4])
        
        df['image_name'] = np.array(names)
        df['nucleus_center'] = np.array(centroids)
        
        df['interSeg_label'] = interseg_label
        if has_centromeric_probe:
            df['ecSeg-c_label'] = ecseg_c_label
        df['ecSeg-i_label'] = ecseg_i_label
        
        dfs.append(df)
        # img_dict[i] = (cell_dict)

    dfs = pd.concat(dfs)
    dfs.to_csv(os.path.join(path_split[0],  f'interphase_prediction_{fish_color}.csv'), index=False)
    # df = pd.DataFrame(img_dict).reset_index()
    # df.to_csv(os.path.join(path_split[0],  'interphase_prediction.csv'), index=False)


if __name__ == "__main__":
   main(sys.argv[1:])