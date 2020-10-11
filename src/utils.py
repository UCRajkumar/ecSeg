import os, glob, cv2
from skimage.io import *
from image_tools import *
from skimage import img_as_ubyte

HSR_SIZE_THRESHOLD = 20
def load_model(model_name):
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join('models', model_name))

def get_imgs(inpath):
    image_paths = glob.glob(os.path.join(inpath, '*.tif'))
    return image_paths

def segment(model, image_path, phase):
    I = imread(image_path)
    I = pre_proc(I)
    save_img(cv2.bitwise_not(np.squeeze(I, -1)), os.path.split(image_path), 'dapi')
    I, patches, pos = im2patches_overlap(I) #crop into patches
    #patches = [np.expand_dims(x, -1) for x in patches] 
    preds = model.predict_on_batch(np.array(patches)) #predict on patches
    I = patches2im_overlap(preds, pos) #stitch image together
    I = img_as_ubyte(I) #convert to uint8
    I = np.argmax(I[:, :, :], axis=2) #flatten the channels
    if(phase == 'meta'): I = meta_inference(I) 
    if(phase == 'inter'): I = inter_inference(I)
    return I

def save_img(I, path, folder):
    cv2.imwrite(os.path.join(path[0], folder, path[1]), I)

def count_cc(I):
    numcc = measure.label(I, return_num = True) #compute number of ecDNA
    return numcc[1]

def count_EC_FISH(ec, fish):
    ec_regs = measure.label(ec)
    num_ec_fish = 0
    for r in np.unique(ec_regs)[1:]:
        mask = (ec_regs == r)
        temp = mask*fish
        if(np.sum(temp) >= 1):
            num_ec_fish += 1
    return num_ec_fish

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