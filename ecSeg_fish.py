#!/usr/bin/env python3
#Author: Utkrisht Rajkumar
#Email: urajkuma@eng.ucsd.edu
#Loads in trained model and produces segmentation maps of images in folder
#Outputs FISH analysis in csv file

import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from PIL import Image
import os, sys, getopt
import pandas as pd
import cv2
from keras.models import load_model
from predict import predict

if sys.version_info[0] < 3:
    raise Exception("Must run with Python version 3 or higher")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):
    inputfile = './'
    FISH_COLOR = 'green'
    THRESHOLD = 120
    PRED_BOOL = 'True'
    try:
        opts, args = getopt.getopt(argv,"h:i:c:t:p:")
    except getopt.GetoptError:
        print('ecSeg.py arguments: \n',
            '-h Displays argument options'  
            '-i <input path> (File path must end in \'\') \n',
            '-c <color of FISH> (\'green\' or \'red\') \n',
            '-t <threshold> threshold values must be [0, 255]. Indicates sensitivity of FISH interaction.', 
            '0 and 255 are the least and highest sensitivity, respectively.\n',
            '-p <predict> (\'True\' or \'false\') Indicates whether to re-segment images')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h"):
            print('ecSeg.py arguments: \n', 
                '-i <input path> (File path must end in \'\') \n',
                '-c <color of FISH> (\'green\' or \'red\') \n',
                '-t <threshold> threshold values must be [0, 255]. Indicates sensitivity of FISH interaction.', 
                '0 and 255 are the least and highest sensitivity, respectively.\n',
                '-p <predict> (\'True\' or \'false\') Indicates whether to re-segment images')
            sys.exit(2)
        if opt in ("-i"):
            inputfile = arg
            if(not os.path.exists(inputfile)):
                print("Invalid file path. Exiting...")
                sys.exit(2)
        elif opt in ('-c'):
            FISH_COLOR = arg
            if('red' not in FISH_COLOR and 'green' not in FISH_COLOR):
                print('-c <color of FISH> Must be \'green\' or \'red\'. Your input was \'', FISH_COLOR,
                    '\'. Exiting...')
                sys.exit(2)
        elif opt in ('-t'):
            THRESHOLD = int(arg)
            if(THRESHOLD >255):
                print('threshold (-t) value must be [0, 255]! Exiting...')
                sys.exit(2)
        elif opt in ('-p'):
            PRED_BOOL = arg
            if('True' not in PRED_BOOL and 'False' not in PRED_BOOL):
                print('-p <predict> Must be \'True\' or \'False\'. Indicates whether to re-segment images. Exiting...')
                sys.exit(2)
    #create folders
    if(os.path.exists((inputfile+'/coordinates'))):
        pass
    else:
        os.mkdir((inputfile+'/coordinates'))

    if(os.path.exists((inputfile+'/labels'))):
        pass
    else:
        os.mkdir((inputfile+'/labels'))

    if(os.path.exists((inputfile+'/dapi'))):
        pass
    else:
        os.mkdir((inputfile+'/dapi'))

    if(os.path.exists((inputfile+'/red'))):
        pass
    else:
        os.mkdir((inputfile+'/red'))

    if(os.path.exists((inputfile+'/green'))):
        pass
    else:
        os.mkdir((inputfile+'/green'))
    if(PRED_BOOL=='True'):
        print("Loading in trained model...")
        model = load_model('ecDNA_ResUnet.h5') #load model
        for f in os.listdir(inputfile): #get all images in path
            ext = os.path.splitext(f)[1]
            if ext.lower() == '.tif':
                print('Segmenting',f)
                predict(model, inputfile, (f))
    IMG_NAME = []
    TOT_EC = []
    TOT_CHROM = []
    FISH_EC_ratio = []
    FISH_CHROM_RATIO = []
    TOT_FISH = []
    FISH_EC = []
    FISH_CHROM = []
    for f in os.listdir(inputfile):
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.tif':
            IMG_NAME.append(f)
            A = np.load((inputfile+'/labels/'+f+'.npy'))
            B = Image.open((inputfile+'/' +f))
            red, green, blue = B.split()
            channels = B.split()
            cv2.imwrite((inputfile+'/dapi/'+f),cv2.bitwise_not(np.uint8(channels[2])))
            cv2.imwrite((inputfile+'/red/'+f),cv2.bitwise_not(np.uint8(channels[0])))
            cv2.imwrite((inputfile+'/green/'+f),cv2.bitwise_not(np.uint8(channels[1])))
            nuc = ~(A==1)
            if('green' in FISH_COLOR):
                fish = (np.array(channels[1]) > THRESHOLD)[:1024,:1280]
            else:
                fish = (np.array(channels[0]) > THRESHOLD)[:1024,:1280]
            fish = fish * nuc
            ec = (A==3)
            chrom = (A==2)
            TOT_FISH.append(len(np.where(fish)[0]))
            TOT_EC.append(len(np.where(ec)[0]))
            TOT_CHROM.append(len(np.where(chrom)[0]))
            FISH_EC.append(len(np.where((fish*ec))[0]))
            FISH_CHROM.append(len(np.where((fish*chrom))[0]))
            if(TOT_FISH[-1]==0):
                FISH_EC_ratio.append(0)
            else:
                FISH_EC_ratio.append(len(np.where((fish*ec))[0])/TOT_FISH[-1])
            if(TOT_FISH[-1]==0):
                FISH_CHROM_RATIO.append(0)
            else:
                FISH_CHROM_RATIO.append(len(np.where((fish*chrom))[0])/TOT_FISH[-1])

            df = pd.DataFrame({'image_name':IMG_NAME, 'ec_pixels':TOT_EC,
                'chrom_pixels':TOT_CHROM, 'fish_pixels({})'.format(FISH_COLOR):TOT_FISH, 'ec+fish':FISH_EC, 'chrom+fish':FISH_CHROM,
                '(ec+fish)/fish':FISH_EC_ratio, '(chrom+fish)/fish': FISH_CHROM_RATIO})
            df.to_csv((inputfile + '/ec_fish.csv'))
    print("FISH analysis complete, successfully exited...")
    K.clear_session()
if __name__ == "__main__":
    main(sys.argv[1:])
