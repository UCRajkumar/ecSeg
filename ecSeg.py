import os
from predict import predict
from keras.models import Model, load_model
import sys, getopt

def main(argv):
    inputfile = './'
    try:
      opts, args = getopt.getopt(argv,"i:")
    except getopt.GetoptError:
      print('ecSeg.py -i <input path>')
      sys.exit(2)
    for opt, arg in opts:
      if opt in ("-i"):
         inputfile = arg

    #create folders
    if(os.path.exists((inputfile+'coordinates'))):
      pass
    else:
      os.mkdir((inputfile+'coordinates'))

    if(os.path.exists((inputfile+'labels'))):
      pass
    else:
      os.mkdir((inputfile+'labels'))

    model = load_model('ecDNA_model.h5') #load model
    for f in os.listdir(inputfile): #get all images in path
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.tif':
          print('Segmenting',f)
          predict(model, inputfile, (f))

if __name__ == "__main__":
   main(sys.argv[1:])