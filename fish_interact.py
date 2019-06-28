import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, sys, getopt
import pandas as pd
import cv2
from keras.models import load_model
from predict import predict
def main(argv):
    inputfile = './'
    FISH_COLOR = 'green'
    THRESHOLD = 120
    PRED_BOOL = True
    try:
        opts, args = getopt.getopt(argv,"i:c:t:p:")
    except getopt.GetoptError:
        print('ecSeg.py -i <input path> -c <color of FISH> -t <threshold> -p <predict>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            inputfile = arg
        elif opt == '-c':
            FISH_COLOR = arg
        elif opt == '-t':
            THRESHOLD = arg
        elif opt == 'p':
            PRED_BOOL = arg
    #create folders

    if(os.path.exists((inputfile+'coordinates'))):
        pass
    else:
        os.mkdir((inputfile+'coordinates'))

    if(os.path.exists((inputfile+'labels'))):
        pass
    else:
        os.mkdir((inputfile+'labels'))

    if(os.path.exists((inputfile+'dapi'))):
        pass
    else:
        os.mkdir((inputfile+'dapi'))

    if(os.path.exists((inputfile+'red'))):
        pass
    else:
        os.mkdir((inputfile+'red'))

    if(os.path.exists((inputfile+'green'))):
        pass
    else:
        os.mkdir((inputfile+'green'))
    if(PRED_BOOL):
        model = load_model('ecDNA_model.h5') #load model
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
        A = np.load((inputfile+'labels/'+f+'.npy'))
        B = Image.open((inputfile +f))
        red, green, blue = B.split()
        channels = B.split()
        cv2.imwrite((inputfile+'dapi/'+f),cv2.bitwise_not(np.uint8(channels[2])))
        cv2.imwrite((inputfile+'red/'+f),cv2.bitwise_not(np.uint8(channels[0])))
        cv2.imwrite((inputfile+'green/'+f),cv2.bitwise_not(np.uint8(channels[1])))
        nuc = ~(A==1)
        if(FISH_COLOR =='green'):
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
            'chrom_pixels':TOT_CHROM, 'fish_pixels(green)':TOT_FISH, 'ec+fish':FISH_EC, 'chrom+fish':FISH_CHROM,
            '(ec+fish)/fish':FISH_EC_ratio, '(chrom+fish)/fish': FISH_CHROM_RATIO})
        df.to_excel((inputfile + 'hsr_ec.xlsx'))

if __name__ == "__main__":
    main(sys.argv[1:])
