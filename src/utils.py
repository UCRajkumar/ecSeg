import os, glob, cv2, math, scipy
from skimage.io import *
from image_tools import *
from skimage.measure import label, regionprops
from skimage.morphology import *
from skimage.filters.rank import *
from skimage.transform import resize
from skimage import img_as_ubyte
import tensorflow as tf

from model_layers.models import UNET
from model_layers.model_RPN import RPN
from model_layers.anchor_size import anchor_size
from model_layers.rpn_proposal import RPNProposal
from model_layers.marker_watershed import marker_watershed

from nuset_utils.anchors import generate_anchors_reference
from nuset_utils.generate_anchors import generate_anchors
from nuset_utils.normalization import *

import warnings
warnings.filterwarnings('ignore')
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


HSR_SIZE_THRESHOLD = 20
def load_model(model_name):
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join('models', model_name))

def get_imgs(inpath):
    image_paths = glob.glob(os.path.join(inpath, '*.tif'))
    return image_paths

def meta_segment(model, image_path):
    I = imread(image_path)
    I = meta_preprocess(I)
    save_img(cv2.bitwise_not(I), os.path.split(image_path), 'dapi')
    I, patches, pos = im2patches_overlap(np.expand_dims(I, axis=-1)) #crop into patches
    #patches = [np.expand_dims(x, -1) for x in patches] 
    preds = model.predict_on_batch(np.array(patches)) #predict on patches
    I = patches2im_overlap(preds, pos) #stitch image together
    I = img_as_ubyte(I) #convert to uint8
    I = np.argmax(I[:, :, :], axis=2) #flatten the channels
    I = meta_inference(I) 
    return I

def inter_classify(model, image_path, fish_index, CELL_THRESHOLD):
    def split_nuclei_segments(img, overlap = 75, scw = 256): 
        h, w = img.shape[:2]; patches = []
        for i in range(0, math.ceil(h/scw)):
            min_row = i*scw
            if(h < 256): max_row = h
            else:
                max_row = min_row + scw
                if(max_row > h): continue
            for j in range(0, math.ceil(w/scw)):
                min_col = j*scw
                if(w < 256): max_col = w
                else:
                    max_col = min_col + scw
                    if(max_col > w): continue
                patches.append(resize(img[min_row:max_row, min_col:max_col, :], (256, 256), preserve_range=True))
        return patches
        
    I = u16_to_u8(imread(image_path))
    _, _, chrom, ec = read_seg(image_path)
    I_segment = nuclei_segment(I, chrom, ec)
    nucleic_regions = label(I_segment)
    region_props = regionprops(nucleic_regions)
    cells = []
    for region in region_props: #The first unique value is 0, which is just background
        mask = (nucleic_regions == region.label)
        temp = I.copy()
        temp[~mask] = 0
        
        bb = region.bbox
        h = bb[2] - bb[0]; w = bb[3] - bb[1]
        
        if((h <= 256) & (w <= 256)):
            nuclei = temp[bb[0]:(bb[0] + min(256, h)), bb[1]:(bb[1]+ min(256, w)), [fish_index, 2]]
            cells.append(resize(nuclei, (256, 256), preserve_range=True))
        else:
            nuclei = temp[bb[0]:(bb[0] + h), bb[1]:(bb[1]+ w), [fish_index, 2]]
            patches = im2patches_overlap(nuclei)
            for p in patches: cells.append(p)
    cells = np.array(cells)
    prediction_scores = model.predict(cells) > CELL_THRESHOLD
    classification = np.sum(prediction_scores)/len(prediction_scores) > 0.5
    return classification

def save_img(I, path, folder):
    cv2.imwrite(os.path.join(path[0], folder, path[1]), I)

def count_cc(I):
    numcc = measure.label(I, return_num = True) #compute number of ecDNA    
    sizes = []
    for i in np.unique(numcc[0])[1:]:
        sizes.append(np.sum(numcc[0] == i))
    return numcc[1], np.sum(sizes)

def count_colocalization(ob1, ob2):
    regs = measure.label(ob1)
    num_coloc = 0
    for r in np.unique(regs)[1:]:
        mask = (regs == r)
        temp = mask*ob2
        if(np.sum(temp) >= 1):
            num_coloc += 1
    return num_coloc

def count_HSR(chrom, fish):
    fish = remove_small_objects(fish, HSR_SIZE_THRESHOLD)
    chrom_regs = measure.label(chrom)
    num_HSR = 0
    for r in np.unique(chrom_regs)[1:]:
        mask = (chrom_regs == r)
        temp = mask*fish
        if(np.sum(temp) >= 1):
            num_HSR += 1
    return num_HSR

def read_seg(image_path):
    path_split = os.path.split(image_path)
    seg_I = np.load(os.path.join(path_split[0], 'labels', path_split[1].split('.')[0] + '.npy'))
    background = (seg_I==0)
    nuclei = (seg_I==1)
    chrom = (seg_I==2)
    ec = (seg_I==3)
    return background, nuclei, chrom, ec

def nuclei_segment(image, name, bbox_min_score, nms_thresh, resize_scale, sess1, sess2,
pred_masks, train_initial, pred_masks_watershed):

    if resize_scale != 1:
        image = rescale(image, resize_scale, anti_aliasing=True)

    # Clip the height and width to be 16-fold 
    imheight, imwidth = image.shape
    imheight = imheight//16*16
    imwidth = imwidth//16*16
    image = image[:imheight, :imwidth]

    image_normalized_wn = whole_image_norm(image)
    image_normalized_wn = np.reshape(image_normalized_wn, [1,imheight,imwidth,1])
    
    masks1 = sess1.run(pred_masks, feed_dict={train_initial:image_normalized_wn})
        
    # # Restore the foreground normalization model from the trained network
    # saver.restore(sess, './models/nuset/foreground.ckpt')
    
    # sess.run(tf.compat.v1.local_variables_initializer())

    if resize_scale != 1:
        image = rescale(image, resize_scale)
    
    # Clip the height and width to be 16-fold 
    imheight, imwidth = image.shape
    imheight = imheight//16*16
    imwidth = imwidth//16*16
    image = image[:imheight, :imwidth]

    # Final pass, foreground normalization to get final masks
    image_normalized_fg = foreground_norm(image, masks1)
    image_normalized_fg = np.reshape(image_normalized_fg, [1,imheight,imwidth,1])
    masks_watershed = sess2.run(pred_masks_watershed, feed_dict={train_initial:image_normalized_fg})

    masks_watershed = clean_image(masks_watershed)

    # Revert the scale to original display
    if resize_scale != 1:
        masks_watershed = rescale(masks_watershed, 1/resize_scale)
    
    I8 = (((masks_watershed - masks_watershed.min()) / (masks_watershed.max() - masks_watershed.min())) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(name[0], 'nuclei',  name[1]), I8)
    return I8