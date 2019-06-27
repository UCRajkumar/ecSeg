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
import cv2
from skimage.morphology import diamond, opening, binary_dilation, binary_erosion, remove_small_objects
from matplotlib import pyplot as plt
from skimage.filters import threshold_minimum
from keras.models import Model

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
        for i in range(146, num_features):
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
    img[binary_dilation(img ==3, diamond(1))] = 3
    return img

def predict(model, path, img_name):
    num_classes = 4
    name = path+img_name
    img = imread(name)
    crops = []
    y = 1
    x = 1
    dim = 256
    shape = img.shape
    vcrop = int(shape[0]/256)
    hcrop = int(shape[1]/256)
    for a in range(0,5):
        y = 1
        for k in range(0,4):
            train = img[y:y+dim, x:x+dim]
            crops.append(train)
            y = y+dim
        x = x+dim
    pred = []   
    for i in range(0,len(crops)):
        x = np.expand_dims(crops[i], axis=0)
        comb_pred = np.squeeze(model.predict(x, verbose=0))
        pred.append(comb_pred)
    stitched_im = np.ones((256*vcrop,1,num_classes))
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
    img = inference(img)
    numecDNA = measure.label(img==3, return_num = True)
    RP = measure.regionprops(numecDNA[0])
    coord_path = path + 'coordinates/'+ os.path.splitext(img_name)[0]+'.txt'
    with open(coord_path, 'w') as f:
        for prop in RP:
            (x, y) = prop.centroid
            f.write('{} {}\n'.format(x, y))
    outpath = path+'labels/'+img_name
    np.save(outpath, img)
    plt.imsave(outpath, img)
    return outpath