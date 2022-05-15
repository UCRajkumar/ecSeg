import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from skimage import morphology
from skimage.measure import regionprops


def _anchor_size(mask):
    
    """
    Given the predicted binary mask, using regionprop
    to find the mode of the diameters of all objec as the 
    estimation of anchor base size used for the RPN
    """
    
    im_height = mask.shape[0]
    im_width = mask.shape[1]
    label_mask = np.array(morphology.label(mask))
    blob_scales = []
    for region in regionprops(label_mask):

        # draw rectangle around segmented coins
        minx, miny, maxx, maxy = region['BoundingBox']
        blob_height = maxx - minx
        blob_weight = maxy - miny
        blob_scales.append(np.maximum(blob_weight,blob_height))
    
    blob_scales = np.asarray(blob_scales)
    optimal_scale = np.median(blob_scales)
    return optimal_scale


def anchor_size(mask):
    
    optimal_size = tf.compat.v1.py_func(_anchor_size, [mask], tf.float64)
    
    return optimal_size