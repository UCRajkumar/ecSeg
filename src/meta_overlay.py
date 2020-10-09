#!/usr/bin/env python3

import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['meta_overlay']
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
        if((sensitivity < 0) | (sensitivity > 255)):
            print("color_sensitivity can only be between 0 and 255. Please update the config.yaml file accordingly.")
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

    df = pd.DataFrame(columns=['image_name', 
    '# of ecDNA(DAPI)', 
    '# of ecDNA(FISH)', 
    '# of ecDNA(DAPI) + FISH', 
    '# of HSR'])

    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        
        fish = split_FISH_channels(i, fish_color, sensitivity)
        if(isinstance(fish, np.ndarray) == False):
            continue

        _, nuclei, chrom, ec = read_seg(i)
        fish = fish * ~nuclei #discard fish pixels in nucleic regions     

        num_ecDNA = count_cc(ec) #compute # of ecDNA based on DAPI
        num_FISH = count_cc(fish * ~chrom) # Compute # of ec based on only fish
        num_ecDNA_FISH = count_EC_FISH(ec, fish) # compute # of ecDNA (DAPI) colocated with FISH
        num_HSR = count_HSR(chrom, fish) # compute # of FISH on a chromosome
        
        df = df.append({'image_name': path_split[1], 
        '# of ecDNA(DAPI)' : num_ecDNA, 
        '# of ecDNA(DAPI) + FISH' : num_ecDNA_FISH, 
        '# of ecDNA(FISH)': num_FISH,
        '# of HSR': num_HSR}, 
        ignore_index=True)

    df.to_csv(os.path.join(path_split[0], 'ecfish_quantification.csv'), index=False)

if __name__ == "__main__":
   main(sys.argv[1:])