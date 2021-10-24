import cv2, os
import numpy as np
from skimage import measure
from skimage.io import *
from skimage.color import *
from skimage.morphology import *
from skimage.filters.rank import *
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
import scipy.misc
from matplotlib import pyplot as plt

NUM_CLASSES = 4
EC_SIZE_THRESHOLD = 15

def inter_inference(img):
    print('Post-processing...')
    return img

def meta_inference(img):
    #if ecDNA is touching chromosome/nuclei, mark that whole
    #component as that class
    def merge_comp(img, class_id=2):  
        I = img
        mask_id = 1
        if(class_id == 1): mask_id = 2
        temp = I == mask_id
        I[temp] = 0 #remove everything with mask_id
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

    def size_thresh(img):
        nuc_regs = measure.regionprops(measure.label(img == 1)) #Set small nucleic regions as background
        chrom_regs = measure.regionprops(measure.label(img == 2))
        avg_chrom_size = np.mean([c.area for c in chrom_regs])
        for r in nuc_regs:
            if(r.area < avg_chrom_size):
                img[tuple(r.coords.T)] = 0 

        chrom_regs = measure.regionprops(measure.label(img == 2)) #Set small chromosomal regions as ecDNA
        ec_regs = measure.regionprops(measure.label(img == 3))
        avg_ec_size = np.mean([c.area for c in ec_regs])
        for r in chrom_regs:
            if(r.area < avg_ec_size):
                img[tuple(r.coords.T)] = 3

        for r in ec_regs: # set small ecs to background
            if(r.area < EC_SIZE_THRESHOLD):
                img[tuple(r.coords.T)] = 0
        return img

    def break_DM(img): #Break double minutes
        temp = binary_erosion(img == 3, diamond(1)) #break bridges between connected components
        img[img == 3] = 0 #reset ecDNA
        img[temp == 1] = 3 #add new ecDNA
        return img

    img = fill_holes(fill_holes(img, 1), 2) #fill holes
    #img = break_DM(img)
    img = size_thresh(img)
    #dilation XOR erosion of ecDNA. Smoothens ecDNA borders by enlarging and then evenly eroding
    img[binary_dilation(img == 3, diamond(1)) ^ binary_erosion(img == 3, diamond(1))] = 0

    chrom_regs = measure.regionprops(measure.label(img == 2))
    nuc_regs = measure.regionprops(measure.label(img == 1))
    c_y = [c.centroid[0] for c in chrom_regs]; c_x= [c.centroid[1] for c in chrom_regs]
    n_cent = [n.centroid for n in nuc_regs]

    ######check if there is a nuclei in the middle of the metaphase###############
    min_chrom_count = 5; v=70
    for idx, n in enumerate(n_cent):
        count = 0
        ######## check if chroms are close to metaphase and if chroms are surrounding the nuclei###
        left = (len(np.where((c_x > n[1]) & (c_x < n[1]+v))[0])> min_chrom_count)
        right = (len(np.where((c_x < n[1]) & (c_x > n[1]-v))[0]) > min_chrom_count)
        bottom = (len(np.where((c_y < n[0]) & (c_y > n[0]-v))[0])> min_chrom_count)
        top = (len(np.where((c_y > n[0]) & (c_y < n[0]+v))[0]) > min_chrom_count)
        if((left*bottom & right*top) or (bottom*right & top*left)): #check if two opposing quadrants contain chrs
            img[tuple(nuc_regs[idx].coords.T)] = 0
    img = merge_comp(merge_comp(img, 1), 2)
    img[binary_dilation(img == 3, diamond(1))] = 3
    return img

def u16_to_u8(img):
    if(img.dtype == 'uint16'):
        img = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))
    return img

def meta_preprocess(img):

    img = u16_to_u8(img)

    if(len(img.shape) > 2):
        img = img[:,:,2]
    
    ret3,th3 = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Ensure image is black background by binarizing and checking if number of white pixels 
    #is greater than 50% of total pixels
    if(np.sum(th3)  > img.shape[0]*img.shape[1]*0.5): 
        img = ~img

    return img

def nuclei_segment(img, chrom, ec):
    img[chrom] = 0
    img[ec] = 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(9, 9),0)
    img = median(img, disk(30))
    _,th3 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    th3 = remove_small_objects(th3.astype('bool'), 3000) #these components are most likely micronuclei
    return th3

def split_FISH_channels(image_path, sensitivity):
    path_split = os.path.split(image_path)
    I = imread(image_path)
    if(len(I.shape)<3):
        print(image_path, " isn't an RGB image. Therefore, no FISH signals could be identified. Skipping...")
        return 0
    
    if(I.dtype == 'uint16'):
        I = cv2.convertScaleAbs(I, alpha=(255.0/65535.0))

    cv2.imwrite(os.path.join(path_split[0], 'red', path_split[1] + '.png'), cv2.bitwise_not(np.uint8(I[...,0])))
    cv2.imwrite(os.path.join(path_split[0], 'green', path_split[1] + '.png'), cv2.bitwise_not(np.uint8(I[...,1])))

    return (np.array(I[...,0]) > sensitivity), (np.array(I[...,1]) > sensitivity)

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
    new_img = np.zeros((h_l + scw, w_l + scw, NUM_CLASSES))

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