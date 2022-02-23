#!/usr/bin/env python3
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors

MODEL_NAME = 'metaseg.h5'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['metaseg']
    inpath = var['inpath']

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

    model = load_model(MODEL_NAME)
    image_paths = get_imgs(inpath)

    df = pd.DataFrame(columns = ['image name', '# of ec'])
    print("Reading from: ", inpath)
    for i in image_paths:
        print("Processing image: ", i)
        
        I = meta_segment(model, i)
        num_ecDNA = count_cc(I==3)[0]
        cmap = colors.ListedColormap(['#386cb0', '#ffff99', '#7fc97f', '#f0027f'])
        path_split = os.path.split(i)
        outpath = os.path.join(path_split[0], 'labels', path_split[1].split('.')[0])
        
        print("Saving image: ", i, " to ", outpath)

        plt.imsave(outpath + '.png', I.astype('uint8'), cmap=cmap, vmin=0, vmax=4)
        np.save(outpath, I)
        df = df.append({'image name' : path_split[1], '# of ec' : num_ecDNA}, ignore_index = True)

    print("Saving ec quantification to", os.path.join(path_split[0], 'ec_quantification.csv'))
    df.to_csv(os.path.join(path_split[0], 'ec_quantification.csv'), index=False)

if __name__ == "__main__":
   main(sys.argv[1:])
