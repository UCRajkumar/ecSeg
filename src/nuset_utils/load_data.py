import pickle
from utils.normalization import whole_image_norm, foreground_norm 
from tqdm import tqdm
from os import listdir
from PIL import Image
import numpy as np
from scipy.ndimage.measurements import label
from scipy import ndimage
import copy

def list_files(directory, extension):
    return [f for f in listdir(directory) if f.endswith('.' + extension)]


def load_data_test(path_to_file):
    all_test = list_files(path_to_file, 'tif')
    # Get the data.
    num_testing = len(all_test)

    # The testing data
    x_test = []
    x_id = []
    for j in range(0,num_testing):
        im = Image.open(path_to_file + all_test[j])
        im = np.asarray(im)
        # if the image is rgb, convert to grayscale
        if len(im.shape) == 3:
            im = im[:,:,2]
        # fix height and width 
        height,width = im.shape
        # The fix_dimension function has been moved inside the test.py
        #width = width//16*16
        #height = height//16*16
        im = im[:height,:width]
        x_test.append(im)
        x_id.append(all_test[j])

    return x_id,x_test



