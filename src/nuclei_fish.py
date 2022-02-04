#!/usr/bin/env python3
import yaml, glob, os, sys, pickle
import pandas as pd
import numpy as np
from utils import *
from image_tools import *
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from scipy import ndimage as ndi
from skimage import *

import warnings
warnings.filterwarnings('ignore')

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def main(argv):

    config = open("config.yaml")
    var = yaml.load(config, Loader=yaml.FullLoader)['nuclei_fish']
    inpath = var['inpath']
    fish_color = var['FISH_color'].lower()
    sensitivity = var['color_sensitivity']
    bbox_min_score = var['min_score'] 
    nms_thresh = var['nms_threshold']
    resize_scale = var['scale_ratio']

    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
    else:
        if((fish_color != 'green') & (fish_color != 'red')):
            print("FISH_color can only be \"green\" or \"red\". Please update the config.yaml file accordingly.")
            sys.exit(2)
     

    if(os.path.exists(os.path.join(inpath, 'red'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'red'))

    if(os.path.exists(os.path.join(inpath, 'green'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'green'))

    if(os.path.exists(os.path.join(inpath, 'nuclei'))):
        pass
    else:
        os.mkdir(os.path.join(inpath, 'nuclei'))

    image_paths = get_imgs(inpath)

    with tf.Graph().as_default():
        pred_dict_final = {}
    
        train_initial = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None, None, 1])

        input_shape = tf.shape(input=train_initial)
        
        input_height = input_shape[1]
        input_width = input_shape[2]
        im_shape = tf.cast([input_height, input_width], tf.float32)
        
        nb_classes = 2
        
        with tf.compat.v1.variable_scope('model_U-Net') as scope:
            final_logits, feat_map = UNET(nb_classes, train_initial)
        
        # The final_logits has 2 channels for foreground/background softmax scores,
        # then we get prediction with larger score for each pixel
        pred_masks = tf.argmax(input=final_logits, axis=3)
        pred_masks = tf.reshape(pred_masks,[input_height,input_width])
        pred_masks = tf.cast(pred_masks, dtype=tf.float32)
        
        # Dynamic anchor base size calculated from median cell lengths
        base_size = anchor_size(tf.reshape(pred_masks,[input_height,input_width]))
        scales = np.array([ 0.5, 1, 2])
        ratios = np.array([ 0.125, 0.25, 0.5, 1, 2, 4, 8])
        
        # stride is to control how sparse we want to place anchors across the image
        # stride = 16 means to place an anchor every 16 pixels on the original image
        stride = 16
        
        ref_anchors = generate_anchors_reference(base_size, ratios, scales)
        num_ref_anchors = scales.shape[0] * ratios.shape[0]

        feat_height = input_height / stride
        feat_width = input_width / stride
        
        all_anchors = generate_anchors(ref_anchors, stride, [feat_height,feat_width])
        
        with tf.compat.v1.variable_scope('model_RPN') as scope:
            prediction_dict = RPN(feat_map, num_ref_anchors)
        
        # Get the tensors from the dict
        rpn_cls_prob = prediction_dict['rpn_cls_prob']
        rpn_bbox_pred = prediction_dict['rpn_bbox_pred']
        
        proposal_prediction = RPNProposal(rpn_cls_prob, rpn_bbox_pred, all_anchors, im_shape, nms_thresh)
        
        pred_dict_final['all_anchors'] = tf.cast(all_anchors, tf.float32)
        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']
            
        pred_dict_final['rpn_prediction'] = prediction_dict
        scores = pred_dict_final['rpn_prediction']['scores']
        proposals = pred_dict_final['rpn_prediction']['proposals']

        pred_masks_watershed = tf.cast(marker_watershed(scores, proposals, pred_masks, min_score = bbox_min_score), dtype=tf.float32)
        sess1 = tf.compat.v1.Session()
        sess1.run(tf.compat.v1.global_variables_initializer())
        saver1 = tf.compat.v1.train.Saver()
        saver1.restore(sess1, './models/nuset/whole_norm.ckpt')
        sess1.run(tf.compat.v1.local_variables_initializer())

        sess2 = tf.compat.v1.Session()
        sess2.run(tf.compat.v1.global_variables_initializer())
        saver2 = tf.compat.v1.train.Saver()
        saver2.restore(sess2, './models/nuset/foreground.ckpt')
        sess2.run(tf.compat.v1.local_variables_initializer())

        dfs = []
        for i in image_paths:
            path_split = os.path.split(i)
            print("Processing image: ", i)
            
            I = imread(i)
            red, green = split_FISH_channels(I, i, sensitivity)
            blue = I[:,:,2]

            if(fish_color == 'green'):
                fish = green
            else:
                fish = red

            if(isinstance(fish, np.ndarray) == False):
                continue

            segmented_cells = nuclei_segment(blue, path_split, bbox_min_score,
            nms_thresh, resize_scale, sess1, sess2, pred_masks, train_initial, pred_masks_watershed)

            segmented_cells = measure.label(segmented_cells)
            regions = measure.regionprops(segmented_cells)

            fish_sizes = []; cell_sizes = []; centroids = []
            df = pd.DataFrame()
            for region in regions:
                cell = (segmented_cells == region.label)
                fish_sizes.append(count_cc(fish*cell)[1])
                cell_sizes.append(np.sum(cell==1))
                center = region.centroid
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
            df['nucleus center'] = centroids
            df['# of fish pixels'] = fish_sizes
            df['# of nuclei pixels'] = cell_sizes
            df.to_csv(os.path.join(path_split[0],  'nuclei', path_split[1][:-4]+'.csv'), index=False)
        sess1.close()
        sess2.close()


if __name__ == "__main__":
   main(sys.argv[1:])