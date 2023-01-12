import os, glob, cv2, math, scipy
from skimage.io import *
from image_tools import *
from skimage.measure import label, regionprops
from skimage.morphology import *
from skimage.filters.rank import *
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.transform import rescale
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

def load_model(model_name):
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join('models', model_name))

def load_nuset(bbox_min_score, nms_thresh, resize_scale):
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
    return sess1, sess2, pred_masks, train_initial, pred_masks_watershed, resize_scale

def get_imgs(inpath):
    image_paths = glob.glob(os.path.join(inpath, '*.tif')) + glob.glob(os.path.join(inpath, '*.npy'))
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

def save_img(I, path, folder):
    cv2.imwrite(os.path.join(path[0], folder, path[1]), I)

def read_seg(image_path):
    path_split = os.path.split(image_path)
    seg_I = np.load(os.path.join(path_split[0], 'labels', path_split[1][:-4] + '.npy'))
    background = (seg_I==0)
    nuclei = (seg_I==1)
    chrom = (seg_I==2)
    ec = (seg_I==3)
    return background, nuclei, chrom, ec

def nuclei_segment(image, resize_scale, sess1, sess2, pred_masks, train_initial, pred_masks_watershed, NUCLEI_SIZE_T):
    if resize_scale != 1:
        image = rescale(image, resize_scale, anti_aliasing=True)

    imheight, imwidth = image.shape
    imheight = imheight//16*16
    imwidth = imwidth//16*16
    image = image[:imheight, :imwidth]
    
    image_normalized_wn = whole_image_norm(image)
    image_normalized_wn = np.reshape(image_normalized_wn, [1,imheight,imwidth,1])

    masks1 = sess1.run(pred_masks, feed_dict={train_initial:image_normalized_wn})
    
    # Final pass, foreground normalization to get final masks
    image_normalized_fg = foreground_norm(image, masks1)
    image_normalized_fg = np.reshape(image_normalized_fg, [1,imheight,imwidth,1])

    masks_watershed = sess2.run(pred_masks_watershed, feed_dict={train_initial:image_normalized_fg})
    masks_watershed = clean_image(masks_watershed)

    # Revert the scale to original display
    if resize_scale != 1:
        masks_watershed = rescale(masks_watershed, 1/resize_scale)
    
    I8 = (((masks_watershed - masks_watershed.min()) / (masks_watershed.max() - masks_watershed.min())) * 255).astype(np.uint8)
    I8[I8 > 0] = 255
    I8 = morphology.remove_small_objects(I8.astype('bool'), NUCLEI_SIZE_T).astype('int') * 255
    # save_img(I8, name, 'annotated')
    return I8