import numpy as np
import tensorflow as tf

from scipy import ndimage as ndi
from skimage import morphology
from skimage.measure import regionprops


def _watershed(scores, proposals, pred_mask, min_score=0.99):
    im_height = pred_mask.shape[0]
    im_width = pred_mask.shape[1]
    markers = np.zeros([im_height, im_width], dtype=np.float32)
    mask = pred_mask.reshape([im_height, im_width])
    
    # Set up a edge mask that has ones at the edges of the matrix
    edge_len = 20
    edge_mask = np.zeros([im_height, im_width])
    edge_mask[edge_len:im_height-edge_len, edge_len:im_width-edge_len] = 1
    # flip
    edge_mask = 1 - edge_mask
    
    if scores.size > 0:
        if np.max(scores) > min_score:
            top_scores_idx = scores > min_score
            scores = scores[top_scores_idx]
            proposals = proposals[top_scores_idx]
            sorted_idx = scores.argsort()
            scores = scores[sorted_idx]
            proposals = proposals[sorted_idx]

            num_bboxes = len(sorted_idx)

            p = 1

            for topn, (score, proposal) in enumerate(zip(scores, proposals)):
                
                bbox = list(proposal)
                #max_cood = np.max(im_width, im_height)
                #bbox = np.clip(bbox, 0, max_cood)
                
                x_pos = int(round((bbox[3] + bbox[1]) / 2))
                y_pos = int(round((bbox[2] + bbox[0]) / 2))
                
                # Make sure markers are only placed on foreground and 
                # each bounding box can only has 1 marker placed
                xmin = int(round(bbox[1]))
                xmax = int(round(bbox[3])) 
                ymin = int(round(bbox[0]))
                ymax = int(round(bbox[2]))
                
                # Since the RPN prediction at the edges are not very accurate, therefore supress the
                # placement of markers at the edges
                if edge_mask[x_pos, y_pos] < 1:
                    #lower scores are always overwritten by higher scores
                    markers[xmin:xmax, ymin:ymax] == 0
                    markers[x_pos, y_pos] = p
                    p = p + 1
            
            
            label_mask = np.array(morphology.label(mask))

            for region in regionprops(label_mask):

                # skip small dirts
                if region['Area'] < 10:
                    continue

                # draw rectangle around segmented nuclei
                minx, miny, maxx, maxy = region['BoundingBox']
                
                minx = np.clip(minx, 0, im_height - 1)
                miny = np.clip(miny, 0, im_width - 1)
                maxx = np.clip(maxx, 0, im_height - 1)
                maxy = np.clip(maxy, 0, im_width - 1)
                
                if np.sum(markers[minx:maxx, miny:maxy]) == 0:
                    x_pos = int(round((minx + maxx) / 2))
                    y_pos = int(round((miny + maxy) / 2))
                    markers[x_pos, y_pos] = p
                    p = p + 1
                    
            markers_rw = morphology.dilation(markers, morphology.disk(3))
            distance = ndi.distance_transform_edt(ndi.binary_fill_holes(mask))
            contour = morphology.watershed(-distance, markers_rw, mask = mask, watershed_line = True)
            contour[contour != 0] = 1
        else:
            contour = np.ones([im_height, im_width], dtype=np.int32)
    else:
        contour = np.ones([im_height, im_width], dtype=np.int32)
    
    _pred_mask = (pred_mask * contour).astype(np.int32) 
    
    # Fill unused watershed markers and holes
    # _pred_mask = ndi.morphology.binary_fill_holes(_pred_mask).astype(np.int32)
    
    return _pred_mask


def marker_watershed(scores, proposals, pred_mask, min_score=0.99):
    
    pred_mask = tf.compat.v1.py_func(_watershed, [scores, proposals, pred_mask, min_score], tf.int32)
    
    return pred_mask
