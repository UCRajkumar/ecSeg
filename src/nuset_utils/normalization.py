import numpy as np
from skimage import morphology

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def whole_image_norm(image):
    return (image - np.mean(image)) / np.std(image)

def foreground_norm(image, mask):
    foreground = image * mask
                                
    # Normalization based on the foreground average and std, not the global average
    foreground_flat = np.reshape(foreground, -1)
    foreground_nonzero = np.asarray(remove_values_from_list(foreground_flat, 0))

    im_median = np.median(foreground_nonzero)
    im_std = np.std(foreground_nonzero)

    # normalize
    im_foreground_norm = (image - im_median) / (im_std + 1e-5)
    
    return im_foreground_norm

def clean_image(image):
   """ perform small region removal and fill holes """
   image = image.astype(np.bool)
   im_label = morphology.label(image, connectivity=1)
   num_cells = np.max(im_label)
   mean_area = np.sum(image).astype(np.float32)/num_cells

   # if a region < 1/5 of a normal cell, remove it
   image = morphology.remove_small_objects(image, min_size=mean_area/5, connectivity=2)
   # if a hole < 1/5 of a normal cell, remove it
   image = morphology.remove_small_holes(image, area_threshold=mean_area/5, connectivity=2)
   return image.astype(np.uint8)
