#!/usr/bin/env python3

import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
HSR_SIZE_THRESHOLD = 20

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['meta_overlay']
    inpath = var['inpath']
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
     

    if(os.path.exists(os.path.join(inpath, 'red'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'red'))

    if(os.path.exists(os.path.join(inpath, 'green'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'green'))

    image_paths = get_imgs(inpath)

    df = pd.DataFrame()

    first_fish = 'green'
    second_fish = 'red'
    
    for i in image_paths:
        path_split = os.path.split(i)
        print("Processing image: ", i)
        I = imread(i)
        red, green = split_FISH_channels(I, i, sensitivity)
        fish = green
        fish2 = red

        if(isinstance(fish, np.ndarray) == False):
            continue

        _, nuclei, chrom, ec = read_seg(i)
        fish = fish * ~nuclei #discard fish pixels in nucleic regions     

        num_ecDNA = count_cc(ec) #compute # of ecDNA based on DAPI
        num_FISH = count_cc(fish * ~chrom) # Compute # of ec based on only fish
        num_ecDNA_FISH = count_colocalization(ec, fish) # compute # of ecDNA (DAPI) colocated with FISH
        num_HSR = count_HSR(chrom, fish, HSR_SIZE_THRESHOLD) # compute # of FISH on a chromosome
        
        if(isinstance(fish2, np.ndarray) == False):
            continue
        
        fish2 = fish2 * ~nuclei #discard fish pixels in nucleic regions 
        num_FISH2 = count_cc(fish2 * ~chrom)
        num_FISH_FISH2 = count_colocalization(fish*~chrom, fish2 * ~chrom)
        num_ecDNA_FISH2 = count_colocalization(ec, fish2)
        num_ecDNA_FISH_FISH2 = count_colocalization(ec, fish2*fish) 
        num_HSR2 = count_HSR(chrom, fish2, HSR_SIZE_THRESHOLD) 

        df = df.append({'image_name': path_split[1], 
        '# of ecDNA (DAPI)' : num_ecDNA, 
        '# of ecDNA (DAPI and ' + first_fish +  ')' : num_ecDNA_FISH, 
        '# of ecDNA ('+ first_fish +  ')': num_FISH,
        '# of HSR (' + first_fish + ')': num_HSR,
        '# of ecDNA (DAPI and ' + second_fish +  ')' : num_ecDNA_FISH2,
        '# of ecDNA (DAPI and ' + second_fish +  ' and ' + first_fish + ')' : num_ecDNA_FISH_FISH2, 
        '# of ecDNA (' + second_fish +  ' and ' + first_fish + ')' : num_FISH_FISH2, 
        '# of ecDNA ('+ second_fish +  ')': num_FISH2,
        '# of HSR (' + second_fish + ')': num_HSR2}, 
        ignore_index=True)

    #rearrange columns
    df = df[['image_name', '# of ecDNA (DAPI)', '# of ecDNA ('+ first_fish +  ')', '# of ecDNA ('+ second_fish +  ')',
    '# of ecDNA (DAPI and ' + first_fish +  ')', '# of ecDNA (DAPI and ' + second_fish +  ')', '# of ecDNA (' + second_fish +  ' and ' + first_fish + ')', 
    '# of ecDNA (DAPI and ' + second_fish +  ' and ' + first_fish + ')', '# of HSR (' + second_fish + ')', '# of HSR (' + first_fish + ')']]

    df.to_csv(os.path.join(path_split[0], 'fish_quantification.csv'), index=False)

if __name__ == "__main__":
   main(sys.argv[1:])