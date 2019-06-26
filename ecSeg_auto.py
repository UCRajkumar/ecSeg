from tkinter import *
import os
from PIL import ImageTk, Image
from io import BytesIO
from predict import predict
from keras.models import Model, load_model
import PIL.ImageOps
import numpy as np
from matplotlib import pyplot as plt

model = load_model('ecDNA_model_dilated_context.h5')
path = './ec/'
imgs = []
tot = []
for f in os.listdir(path): #get all images in path
    ext = os.path.splitext(f)[1]
    if ext.lower() == '.tif':
        tot.append(predict(model, path, (f)))
print(tot)
#predict(model, '3.tif')
#predict(model, '4.tif')

'''
import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    im = cv2.imread("./SNU16/1.tif", cv2.IMREAD_COLOR)
    
    # Select ROI
    #r = cv2.selectROI("Image", im, False, False)
     
    # Crop image
    #imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
    cv2.imshow("Image", imCrop)

    cv2.waitKey(0)
'''