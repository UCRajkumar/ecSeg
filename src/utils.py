import os, glob, cv2, math, scipy
from skimage.io import *
from image_tools import *
from skimage.measure import label, regionprops
from skimage.morphology import *
from skimage.filters.rank import *
from skimage.transform import resize
from skimage import img_as_ubyte

HSR_SIZE_THRESHOLD = 20
def load_model(model_name):
    import tensorflow as tf
    return tf.keras.models.load_model(os.path.join('models', model_name))

def get_imgs(inpath):
    image_paths = glob.glob(os.path.join(inpath, '*.tif'))
    return image_paths

def meta_segment(model, image_path):
    I = imread(image_path)
    I = pre_proc(I)
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
    print(image_path)
    I = imread(image_path)
    if(I.dtype == 'uint16'):
        I = cv2.convertScaleAbs(I, alpha=(255.0/65535.0))
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
        
        nuclei = temp[bb[0]:(bb[0] + min(256, h)), bb[1]:(bb[1]+ min(256, w)), [fish_index, 2]]
        cells.append(resize(nuclei, (256, 256), preserve_range=True));
    cells = np.array(cells)
    prediction_scores = model.predict(cells)
    classification = scipy.stats.mode(prediction_scores > CELL_THRESHOLD)[0][0][0]
    return 0 #classification

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