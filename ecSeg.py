#!/usr/bin/env python3
#Author: Utkrisht Rajkumar
#Email: urajkuma@eng.ucsd.edu
#Loads in trained model and produces segmentation maps of images in folder

import os
from predict import predict
from keras.models import Model, load_model
import sys, getopt
from keras import backend as K

if sys.version_info[0] < 3:
    raise Exception("Must run with Python version 3 or higher")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(argv):
    inputfile = './'
    model_name = 'ecDNA_ResUnet.h5'
    
    try:
      opts, args = getopt.getopt(argv,"i:m:")
    except getopt.GetoptError:
      print('ecSeg.py -i <input path> -m <model name>')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-i"):
         inputfile = arg
      if opt in ("-m"):
         model_name = arg

    #create folders
    if(os.path.exists((inputfile+'/coordinates'))):
      pass
    else:
      os.mkdir((inputfile+'/coordinates'))
    
    if(os.path.exists((inputfile+'/dapi'))):
      pass
    else:
      os.mkdir((inputfile+'/dapi'))

    if(os.path.exists((inputfile+'/labels'))):
      pass
    else:
      os.mkdir((inputfile+'/labels'))
    
    print("Loading in trained model...")
    model = load_model(model_name, compile=False) #load model
    print("Model loaded.")
    for f in os.listdir(inputfile): #get all images in path
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.tif':
          print('Segmenting',f)
          predict(model, inputfile, (f))
    print("Successfully exited...")
    K.clear_session()
if __name__ == "__main__":
   main(sys.argv[1:])
