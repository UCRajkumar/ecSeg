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
    var = yaml.load(config, Loader=yaml.FullLoader)['interseg']
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

if __name__ == "__main__":
   main(sys.argv[1:])