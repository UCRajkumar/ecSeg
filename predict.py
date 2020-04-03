#!/usr/bin/env python3
#Author: Utkrisht Rajkumar
#Email: urajkuma@eng.ucsd.edu
#Produces segmentation and post-processes the image
#Code to crop with overlays and re-stitch borrowed from https://github.com/neuropoly/axondeepseg

from os import listdir
import os
from os.path import isfile, join
import sys
import numpy as np
from skimage import measure, img_as_ubyte
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

def im2patches_overlap(img, overlap_value=25, scw=256): #See https://github.com/neuropoly/axondeepseg
    '''
    Convert an image into patches.
    :param img: the image to convert.
    :param overlap_value: Int, the number of pixels to use when overlapping the predictions.
    :param scw: Int, input size.
    :return: the original image, a list of patches, and their positions.
    '''

    # First we crop the image to get the context
    cropped = img[overlap_value:-overlap_value, overlap_value:-overlap_value]

    # Then we create patches using the prediction window
    spw = scw - 2 * overlap_value  # size prediction windows

    qh, rh = divmod(cropped.shape[0], spw)
    qw, rw = divmod(cropped.shape[1], spw)

    # Creating positions of prediction windows
    L_h = [spw * e for e in range(qh)]
    L_w = [spw * e for e in range(qw)]

    # Then if there is a remainder we take the last positions (overlap on the last predictions)
    if rh != 0:
        L_h.append(cropped.shape[0] - spw)
    if rw != 0:
        L_w.append(cropped.shape[1] - spw)

    xx, yy = np.meshgrid(L_h, L_w)
    P = [np.ravel(xx), np.ravel(yy)]
    L_pos = [[P[0][i], P[1][i]] for i in range(len(P[0]))]

    # These positions are also the positions of the context windows in the base image coordinates !
    L_patches = []
    for e in L_pos:
        patch = img[e[0]:e[0] + scw, e[1]:e[1] + scw]
        L_patches.append(patch)

    return [img, L_patches, L_pos]

def patches2im_overlap(L_patches, L_pos, overlap_value=25, scw=256): #See https://github.com/neuropoly/axondeepseg

    '''
    Stitches patches together to form an image.
    :param L_patches: List of segmented patches.
    :param L_pos: List of positions of the patches in the image to form.
    :param overlap_value: Int, number of pixels to overlap.
    :param scw: Int, patch size.
    :return: Stitched segmented image.
    '''

    spw = scw - 2 * overlap_value
    # L_pred = [e[cropped_value:-cropped_value,cropped_value:-cropped_value] for e in L_patches]
    # First : extraction of the predictions
    h_l, w_l = np.max(np.stack(L_pos), axis=0)
    L_pred = []
    new_img = np.zeros((h_l + scw, w_l + scw, 4))

    for i, e in enumerate(L_patches):
        if L_pos[i][0] == 0:
            if L_pos[i][1] == 0:
                new_img[0:overlap_value, 0:overlap_value] = e[0:overlap_value, 0:overlap_value]
                new_img[overlap_value:scw - overlap_value, 0:overlap_value] = e[overlap_value:-overlap_value,
                                                                              0:overlap_value]
                new_img[0:overlap_value, overlap_value:scw - overlap_value] = e[0:overlap_value,
                                                                              overlap_value:-overlap_value]
            else:
                if L_pos[i][1] == w_l:
                    new_img[0:overlap_value, -overlap_value:] = e[0:overlap_value, -overlap_value:]
                new_img[0:overlap_value, L_pos[i][1] + overlap_value:L_pos[i][1] + scw - overlap_value] = e[
                                                                                                          0:overlap_value,
                                                                                                          overlap_value:-overlap_value]

        if L_pos[i][1] == 0:
            if L_pos[i][0] != 0:
                new_img[L_pos[i][0] + overlap_value:L_pos[i][0] + scw - overlap_value, 0:overlap_value] = e[
                                                                                                          overlap_value:-overlap_value,
                                                                                                          0:overlap_value]

        if L_pos[i][0] == h_l:
            if L_pos[i][1] == w_l:
                new_img[-overlap_value:, -overlap_value:] = e[-overlap_value:, -overlap_value:]
                new_img[h_l + overlap_value:-overlap_value, -overlap_value:] = e[overlap_value:-overlap_value,
                                                                               -overlap_value:]
                new_img[-overlap_value:, w_l + overlap_value:-overlap_value] = e[-overlap_value:,
                                                                               overlap_value:-overlap_value]
            else:
                if L_pos[i][1] == 0:
                    new_img[-overlap_value:, 0:overlap_value] = e[-overlap_value:, 0:overlap_value]

                new_img[-overlap_value:, L_pos[i][1] + overlap_value:L_pos[i][1] + scw - overlap_value] = e[
                                                                                                          -overlap_value:,
                                                                                                          overlap_value:-overlap_value]
        if L_pos[i][1] == w_l:
            if L_pos[i][1] != h_l:
                new_img[L_pos[i][0] + overlap_value:L_pos[i][0] + scw - overlap_value, -overlap_value:] = e[
                                                                                                          overlap_value:-overlap_value,
                                                                                                          -overlap_value:]

    L_pred = [e[overlap_value:-overlap_value, overlap_value:-overlap_value] for e in L_patches]
    L_pos_corr = [[e[0] + overlap_value, e[1] + overlap_value] for e in L_pos]
    for i, e in enumerate(L_pos_corr):
        new_img[e[0]:e[0] + spw, e[1]:e[1] + spw] = L_pred[i]

    return new_img

def compute_stat(img, path, img_name):
    numecDNA = measure.label(img==3, return_num = True) #compute number of ecDNA
    RP = measure.regionprops(numecDNA[0])
    coord_path = path + '/coordinates/'+ os.path.splitext(img_name)[0]+'.txt'
    with open(coord_path, 'w') as f:
        for prop in RP:
            (x, y) = prop.centroid
            f.write('{}, {}\n'.format(x, y))
    return numecDNA[1]

#Crops large size image, predicts on patches, and restitches image
def predict(model, path, img_name):
    name = path+'/'+img_name
    img = imread(name)
    img = pre_proc(img, path+'/dapi/'+img_name)
    img, patches, pos = im2patches_overlap(img) #crop into patches
    #patches = [np.expand_dims(x, -1) for x in patches] 
    preds = model.predict_on_batch(np.array(patches)) #predict on patches
    img = patches2im_overlap(preds, pos) #stitch image together
    img = img_as_ubyte(img) #convert to uint8
    img = np.argmax(img[:, :, :], axis=2) #flatten the channels
    img = inference(img) 
    print("Saving ", img_name)
    data_path = path+'/labels/'+ os.path.splitext(img_name)[0]
    im_path = path+'/labels/'+ os.path.splitext(img_name)[0] + '.tif'
             # blue     # white    #green     #white

    cmap1 = ['#386cb0', '#ffff99', '#7fc97f', '#f0027f']
    cmap = colors.ListedColormap(cmap1) 
    
    plt.imsave(im_path, img.astype('uint8'), cmap=cmap)
    np.save(data_path, img)

    num_ecDNA = compute_stat(img, path, img_name)
    return img_name, num_ecDNA
