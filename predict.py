#!/usr/bin/env python3
#Author: Utkrisht Rajkumar
#Email: urajkuma@eng.ucsd.edu
#Produces segmentation and post-processes the image

from os import listdir
import os
from os.path import isfile, join
import sys
import numpy as np
from skimage import measure
from skimage.io import imread, imshow, imread_collection, concatenate_images
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from skimage.color import label2rgb, rgb2gray, gray2rgb
from PIL import Image
import scipy.misc
import cv2
from skimage.morphology import diamond, opening, binary_dilation, binary_erosion, remove_small_objects
from matplotlib import pyplot as plt
from skimage.filters import threshold_minimum
from keras.models import Model
import matplotlib.colors as colors


if sys.version_info[0] < 3:
    raise Exception("Must run with Python version 3 or higher")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

DIM = 256
NUM_CLASSES = 4

#post-processing
def inference(img):
    #if ecDNA is touching chromosome/nuclei, mark that whole
    #component as that class
    def merge_comp(img, class_id):  
        I = img
        if(class_id == 1):
            mask_id = 2
        else:
            mask_id = 1
        temp = I == mask_id
        I[temp] = 0
        O = I
        s = generate_binary_structure(2,2)
        labeled_array, num_features = label(I,  structure=s)
        for i in range(1, num_features):
            ind = (labeled_array == i)
            if(np.any(I[ind]==class_id)):
                O[ind] = class_id
        img[opening(O, diamond(1)) == class_id] = class_id #reset nuclei and chromosomes in main image
        img[temp] = mask_id 
        return img

    #fill holes in connected components
    def fill_holes(img, class_id):
        temp = binary_fill_holes(img == class_id)
        img[temp == 1] = class_id
        return img

    #remove ecDNA too small and mark ecDNA that are too large as chromosomes
    def size_thresh(img):
        RP = measure.regionprops(measure.label(img == 3))
        for region in RP:
            if(region.area > 125):
                img[tuple(region.coords.T)] = 2
            if(region.area < 15):
                img[tuple(region.coords.T)] = 0
        return img

    img = fill_holes(fill_holes(fill_holes(img, 1), 2), 3) #fill holes
    img = size_thresh(img)
    img[binary_dilation(img == 3, diamond(1)) ^ binary_erosion(img == 3, diamond(1))] = 0
    img = merge_comp(merge_comp(img, 1), 2)
    img[binary_dilation(img == 3, diamond(1))] = 3
    return img

def pre_proc(img, path):
    if(img.dtype == 'uint16'):
        img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    if(len(img.shape) > 2):
        img = img[:,:,2]
    
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Ensure image is black background by binarizing and checking if number of white pixels 
    #is greater than 50% of total pixels
    if(np.sum(th3)  > img.shape[0]*img.shape[1]*0.5): 
        img = ~img
    
    cv2.imwrite(path, cv2.bitwise_not(img))
    img = np.expand_dims(img, axis=-1)
    return img

def crop(img):
    crops = []
    y = 1; x = 1
    shape = img.shape
    vcrop = int(shape[0]/DIM)
    hcrop = int(shape[1]/DIM)
    for a in range(0, hcrop):
        y = 1
        for k in range(0, vcrop):
            train = img[y:y+DIM, x:x+DIM]
            crops.append(train)
            y = y+DIM
        x = x+DIM
    return crops, vcrop, hcrop

def stitch(pred, vcrop, hcrop):
    stitched_im = np.ones((256*vcrop,1,NUM_CLASSES))
    index = -1
    for j in range (1,hcrop+1):
        index = index +1
        if(index >=hcrop*vcrop):
            break
        row = pred[index]
        for k in range(1,vcrop):
            index = index + 1
            I = pred[index]
            row = np.vstack((row, I))
        stitched_im = np.hstack((stitched_im, row))
    img = np.argmax(stitched_im[:, 1:, :], axis=2)
    return img

def compute_stat(img, path, img_name):
    numecDNA = measure.label(img==3, return_num = True) #compute number of ecDNA
    RP = measure.regionprops(numecDNA[0])
    coord_path = path + '/coordinates/'+ os.path.splitext(img_name)[0]+'.txt'
    with open(coord_path, 'w') as f:
        for prop in RP:
            (x, y) = prop.centroid
            f.write('{}, {}\n'.format(x, y))

#Crops large size image, predicts on patches, and restitches image
def predict(model, path, img_name):
    name = path+'/'+img_name
    img = imread(name)
    img = pre_proc(img, path+'/dapi/'+img_name)
    crops, vcrop, hcrop = crop(img)
    pred = []   
    for i in range(0,len(crops)):
        x = np.expand_dims(crops[i], axis=0)
        comb_pred = np.squeeze(model.predict(x, verbose=0))
        pred.append(comb_pred)
    
    img = stitch(pred, vcrop, hcrop)
    img = inference(img)

    print("Saving ", img_name)
    data_path = path+'/labels/'+ os.path.splitext(img_name)[0]
    im_path = path+'/labels/'+ os.path.splitext(img_name)[0] + '.tif'
#             # blue     # white    #green     #white
    cmap1 = ['#386cb0', '#ffff99', '#7fc97f', '#f0027f']
    cmap = colors.ListedColormap(cmap1) 
    plt.imsave(im_path, img.astype('uint8'), cmap=cmap)
    np.save(data_path, img)

    compute_stat(img, path, img_name)
    return im_path
