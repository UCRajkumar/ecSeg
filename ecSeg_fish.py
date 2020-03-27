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
from skimage import measure
from skimage.io import imread

if sys.version_info[0] < 3:
    raise Exception("Must run with Python version 3 or higher")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):
    inputfile = './'
    model_name = 'ecDNA_ResUnet.h5'
    FISH_COLOR = 'green'
    THRESHOLD = 120
    PRED_BOOL = 'True'
    try:
        opts, args = getopt.getopt(argv,"h:i:c:t:p:m:")
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
        elif opt in ('-m'):
            model_name = arg

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
        model = load_model(model_name) #load model
        for f in os.listdir(inputfile): #get all images in path
            ext = os.path.splitext(f)[1]
            if ext.lower() == '.tif':
                print('Segmenting',f)
                predict(model, inputfile, (f))
    df = pd.DataFrame(columns=['image_name', 'ec_pixels', 'chrom_pixels', 'fish_pixels({})'.format(FISH_COLOR), 'ec+fish pixels', 
    'chrom+fish pxiels', '(ec+fish pixels)/fish', '(chrom+fish pixels)/fish', '# of ecDNA + fish'])
    for f in os.listdir(inputfile):
        name = os.path.splitext(f)[0]
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.tif':
            print("Processing ", name)
            img_name = f
            I = imread((inputfile+'/' +f))
            if(len(I.shape)<3):
                print(name, " isn't an RGB image and cannot be processed for FISH analysis")
                continue
                
            if(I.dtype == 'uint16'):
                I = cv2.convertScaleAbs(I, alpha=(255.0/65535.0))
                
            labels = np.load((inputfile+'/labels/'+name+'.npy'))
            height = labels.shape[0]; width = labels.shape[1]
            
            cv2.imwrite((inputfile+'/red/'+f),cv2.bitwise_not(np.uint8(I[...,0])))
            cv2.imwrite((inputfile+'/green/'+f),cv2.bitwise_not(np.uint8(I[...,1])))
            
            nuc = ~(labels==1)
            
            if('green' in FISH_COLOR):
                fish = (np.array(I[...,1]) > THRESHOLD)[:height,:width]
            else:
                fish = (np.array(I[...,0]) > THRESHOLD)[:height,:width]
                
            fish = fish * nuc
            ec = (labels==3)
            chrom = (labels==2)
            
            fish_ec_overlay = fish*ec
            fish_chrom = fish*chrom
            tot_fish = len(np.where(fish)[0])
            tot_ec = len(np.where(ec)[0])
            tot_chrom = len(np.where(chrom)[0])
            fish_ec = len(np.where((fish_ec_overlay))[0])
            fish_chrom = len(np.where((fish_chrom))[0])
            
            numecDNA = measure.label(fish_ec_overlay, return_num = True) #compute number of ecDNA
            
            if(tot_fish==0):
                fish_ec_ratio = 0
            else:
                fish_ec_ratio = fish_ec/tot_fish
            if(tot_fish==0):
                fish_chrom_ratio = 0
            else:
                fish_chrom_ratio = fish_chrom/tot_fish
            df = df.append({'image_name':img_name, 'ec_pixels':tot_ec,
                'chrom_pixels':tot_chrom, 'fish_pixels({})'.format(FISH_COLOR):tot_fish, 'ec+fish pixels':fish_ec, 'chrom+fish pxiels':fish_chrom,
                '(ec+fish pixels)/fish':fish_ec_ratio, '(chrom+fish pixels)/fish': fish_chrom_ratio, '# of ecDNA + fish' : numecDNA[1]}, ignore_index=True)
        df.to_csv((inputfile + '/ec_fish.csv'))

    print("FISH analysis complete, successfully exited...")
    K.clear_session()
if __name__ == "__main__":
    main(sys.argv[1:])
