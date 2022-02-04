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

def classBalancemap(mask):

    """
    Give low weights to most irrelevant pixels and high weights to pixels around
    the features 
    """
    maskShape = mask.shape
    weight = np.ones(maskShape)

    for i in range(10):
        mask = ndimage.binary_dilation(mask).astype(mask.dtype)
    weight[mask==1] = 100
    mask1 = copy.deepcopy(mask)

    for i in range(10):
        mask1 = ndimage.binary_dilation(mask1).astype(mask1.dtype)
    mask = mask1 - mask
    weight[mask==1] = 50
    mask2 = copy.deepcopy(mask1)

    for i in range(10):
        mask2 = ndimage.binary_dilation(mask2).astype(mask2.dtype)
    mask = mask2 - mask1
    weight[mask==1] = 20

    return weight

def unetwmap(mask, w0=10, sigma=5):

    """
    Calculate the U-Net Weight Map
    Adapted from unetwmap function written by Fidel A. Guerrero Pena
    """
    
    uvals = np.unique(mask)
    wmp = np.zeros(len(uvals))

    for i, uv in enumerate(uvals):
        wmp[i] = 1/ np.sum( mask == uv)

    # Normalize
    wmp=wmp/wmp.max()
    wc = copy.deepcopy(mask)
    wc = wc.astype(np.float32)

    for i, uv in enumerate(uvals):
        wc[ mask == uv ] = wmp[i]

    # Convert to bwlabel.
    cells,_ = label(mask)

    # Cell distance map.
    bwgt = np.zeros(mask.shape)
    maps = np.zeros((mask.shape[0],mask.shape[1], cells.max()))
    if cells.max() >= 2:
        for ci in range(cells.max()):
            maps[:, :, ci] = ndimage.distance_transform_edt(np.invert(cells == ci + 1))

        maps = np.sort(maps, 2)
        d0 = maps[:, :, 0]
        d1 = maps[:, :, 1]
        bwgt = w0 * np.exp(-np.square(d0 + d1) / (2 * sigma)) * (cells == 0)
    # Unet weights
    weight = wc + bwgt
    
    return weight

def bounding_box(mask):
    
    mask_label,num_cells = label(mask)
    height = mask.shape[0]
    width = mask.shape[1]
    # The coordinates of bounding box denotes x,y,w,h
    # x,y is the center of the box, and w is the width and h is the height
    b_box = np.zeros((num_cells,5))
    for k in range (0,num_cells):
        coords_x, coords_y = np.where(mask_label == k+1)
        # The last column is the label, also we give an edge of 10 pix for bbox
        ymin = max(coords_y.min()-25,0)
        xmin = max(coords_x.min()-25,0)
        ymax = min(coords_y.max()+25,width)
        xmax = min(coords_x.max()+25,height)
        b_box[k,:] = [ymin,xmin,ymax,xmax,1]

    return b_box[np.argsort(b_box[:, 0])]

def load_data_train(self, normalization_method='fg'):

    """
       Load and normalize the training data from the integrated .pckl file

       argument: normalization_method(str): can choose between 'wn'(whole image normalization)
       and 'fg'(foreground normalization)
       
       return: the formatted input ready for network
    """

    # First load the training image
    img_dir = self.train_img_path
    imlabel_dir = self.train_label_path

    img_val_dir = self.val_img_path
    imlabel_val_dir = self.val_label_path

    # load the val image
    if len(list_files(img_dir, 'png')) > 0:
        all_train = list_files(img_dir, 'png')
        all_val = list_files(img_val_dir, 'png')
    else:
        all_train = list_files(img_dir, 'tif')
        all_val = list_files(img_val_dir, 'tif')

    if len(list_files(imlabel_dir, 'png')) > 0:
        all_train_label = list_files(imlabel_dir, 'png')
        all_val_label = list_files(imlabel_val_dir, 'png')
    else:
        all_train_label = list_files(imlabel_dir, 'tif')
        all_val_label = list_files(imlabel_val_dir, 'tif')

    if self.usingCL:
        print('Computing weight matrix ...')
    else:
        self.training_results.set('Computing weight matrix ...')
        self.window.update()

    # Get the data
    num_training = len(all_train)
    num_training_label = len(all_train_label)

    num_val = len(all_val)
    num_val_label = len(all_val_label)
    # The number of training images and training labels should be the same
    assert num_training == num_training_label
    assert num_val == num_val_label

    all_train.sort()
    all_train_label.sort()
    all_val.sort()
    all_val_label.sort()
 
    # The training data
    x_train = []
    x_val = []
    # The training label
    y_train = []
    y_val = []
    w_train = []
    w_val = []
    bbox_train = []
    bbox_val = []

    for j in tqdm(range(0,num_training)):
        im = Image.open(img_dir + all_train[j])
        im = np.asarray(im)
        if len(im.shape) > 2:
            r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
            im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # fix height and width 
        height,width = im.shape
        width = width//16*16
        height = height//16*16
        im = im[:height,:width]

        iml = Image.open(imlabel_dir + all_train_label[j])
        iml = np.asarray(iml)
        if len(iml.shape) > 2:
            r, g, b = iml[:,:,0], iml[:,:,1], iml[:,:,2]
            iml = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # fix height and width 
        height,width = iml.shape
        width = width//16*16
        height = height//16*16
        iml = iml[:height,:width]

        # Remove training images with blank labels/annotations
        # to avoid dividing by 0
        if np.max(iml) > 1:
            y_train.append(iml/np.max(iml))
            w_train.append(unetwmap(iml/np.max(iml)))
            # for mitosis detection
            #w_train.append(classBalancemap(iml/np.max(iml)))
            bbox_train.append(bounding_box(iml/np.max(iml)))
            x_train.append(im)   
  
        elif np.max(iml) == 1:
            y_train.append(iml)
            w_train.append(unetwmap(iml))
            # for mitosis detection
            #w_train.append(classBalancemap(iml))
            bbox_train.append(bounding_box(iml))
            x_train.append(im)
    
    for j in tqdm(range(0,num_val)):
        im = Image.open(img_val_dir + all_val[j])
        im = np.asarray(im)
        if len(im.shape) > 2:
            r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
            im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # fix height and width 
        height,width = im.shape
        width = width//16*16
        height = height//16*16
        im = im[:height,:width]

        iml = Image.open(imlabel_val_dir + all_val_label[j])
        iml = np.asarray(iml)
        if len(iml.shape) > 2:
            r, g, b = iml[:,:,0], iml[:,:,1], iml[:,:,2]
            iml = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # fix height and width 
        height,width = iml.shape
        width = width//16*16
        height = height//16*16
        iml = iml[:height,:width]

        # Remove training images with blank labels/annotations
        # to avoid dividing by 0
        if np.max(iml) > 1:
            y_val.append(iml/np.max(iml))
            w_val.append(unetwmap(iml/np.max(iml)))
            # for mitosis detection
            #w_train.append(classBalancemap(iml/np.max(iml)))
            bbox_val.append(bounding_box(iml/np.max(iml)))
            x_val.append(im)   
  
        elif np.max(iml) == 1:
            y_val.append(iml)
            w_val.append(unetwmap(iml))
            # for mitosis detection
            #w_train.append(classBalancemap(iml))
            bbox_val.append(bounding_box(iml))
            x_val.append(im)

    if self.usingCL:
        print('Normalizing ...')
    else:
        self.training_results.set('Normalizing ...')
        self.window.update()
    if normalization_method == 'wn':
        # Normalizing the training data
        for i in tqdm(range(len(x_train))):
            x_train[i] = whole_image_norm(x_train[i])

        # Normalizing the validation data
        for i in tqdm(range(len(x_val))):
            x_val[i] = whole_image_norm(x_val[i])

    if normalization_method == 'fg':
        # Normalizing the training data
        print('fg norm on training data...')
        for i in tqdm(range(len(x_train))):
            x_train[i] = foreground_norm(x_train[i], y_train[i])

        # Normalizing the validation data: notice it is normalized 
        # based on whole image norm model predictions
        print('fg norm on val data...')
        for i in tqdm(range(len(x_val))):
            x_val[i] = foreground_norm(x_val[i], self.whole_norm_y_pred[i])

    return (x_train, x_val, y_train, y_val, w_train, w_val, bbox_train, bbox_val)

def load_data_test(path_to_file):
    if len(list_files(path_to_file, 'png')) > 0:
        all_test = list_files(path_to_file, 'png')
    else:
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
            r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]
            im = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # fix height and width 
        height,width = im.shape
        # The fix_dimension function has been moved inside the test.py
        #width = width//16*16
        #height = height//16*16
        im = im[:height,:width]
        x_test.append(im)
        x_id.append(all_test[j])

    return x_id,x_test



