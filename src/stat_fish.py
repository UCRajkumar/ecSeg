#!/usr/bin/env python3
import yaml, glob, os, sys, datetime, shutil
from pathlib import Path
config = open("config.yaml")
stat_fish_params_file = open("src/stat_fish_params.yaml")
var = yaml.load(config, Loader=yaml.FullLoader)['stat_fish']
stat_fish_params = yaml.load(stat_fish_params_file, Loader=yaml.FullLoader)
Path("README.md").touch()

import pandas as pd
import numpy as np
from utils import *
import max_flow_binary_mask
from image_tools import *
import seaborn as sns
from skimage import *
import scipy.stats
import cv2
from tqdm import tqdm
import subprocess as sp
import warnings
warnings.filterwarnings('ignore')

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def scipy_sampled_gaussian_kernel(kernel_shape, sigma=1):
    if not isinstance(kernel_shape, np.ndarray):
        kernel_shape = np.array(kernel_shape)
    if not kernel_shape[0] % 2 or not kernel_shape[1]:
        print("\n\tWarning: Even Kernel Used in Convolution\n")
    
    centers = (kernel_shape / 2) - 0.5
    kernel_axis_y, kernel_axis_x = [np.arange(kernel_axis_size) - center for kernel_axis_size, center in zip(kernel_shape, centers)]
    grid = np.linalg.norm(np.dstack(np.meshgrid(kernel_axis_x, kernel_axis_y)), axis=2).astype(np.float64)
    gaussian = scipy.stats.norm.pdf(grid, scale=sigma)
    return gaussian / gaussian.sum()


def get_gaussian_proj_kernel(kernel_size, sigma):
    g_kernel = scipy_sampled_gaussian_kernel(kernel_size, sigma=sigma)
    c_kernel = np.ones(kernel_size)

    # Normalizing constant kernel
    c_kernel = c_kernel / np.linalg.norm(c_kernel)
    
    # Projecting and Normalizing gaussian kernel
    g_kernel_proj = np.dot(g_kernel.flatten(), c_kernel.flatten()) * c_kernel
    g_kernel_perp = g_kernel - g_kernel_proj
    g_kernel_perp /= np.linalg.norm(g_kernel_perp)

    while len(g_kernel_perp.shape) < 4:
        g_kernel_perp = np.expand_dims(g_kernel_perp, -1)
    return g_kernel_perp


def get_sensitivity(I, segmented_cells, intensity_threshold_std_coeff):
    # Get Color Sensitivity
    seg_copy = segmented_cells.copy().astype(bool).astype(np.float32)
    mean = np.array([(seg_copy * I[:,:,chan]).sum() / seg_copy.sum() for chan in range(1, I.shape[-1])])
    seg_copy[seg_copy == 0] = np.nan
    stdev = np.array([np.nanstd((seg_copy * I[:,:,chan])) for chan in range(1, I.shape[-1])])
    color_sensitivity = mean + (intensity_threshold_std_coeff * stdev)
    return color_sensitivity


def get_boundary_sum_kernel(gaussian_stdev, kernel_side_size):
    arr = scipy_sampled_gaussian_kernel((kernel_side_size, kernel_side_size), gaussian_stdev)
    return np.sum((arr[0].sum(), (arr[-1].sum() if arr.shape[0] > 1 else 0), arr[1:-1,0].sum(), arr[1:,-1].sum()))


def get_thresholded(I, segmented_cells, gaussian_stdev, normal_threshold, color_sensitivity, gaussian_kernel_shape):
    g_kernel_perp = get_gaussian_proj_kernel(gaussian_kernel_shape, gaussian_stdev)
    num_channels = I.shape[-1]
    inter = np.expand_dims([I[...,channel] for channel in range(1, num_channels)], -1).astype(np.float64)
    normal_coefficients = tf.nn.conv2d(inter, g_kernel_perp, strides=1, padding="SAME").eval(session=tf.compat.v1.Session())
    assert normal_coefficients.shape[-1] == 1
    normal_coefficients = np.dstack(normal_coefficients[...,0])

    # If pixel is max brightness or is brighter than neighbors, classify as center
    max_pixels = np.dstack([(channel == channel.max()) * bool(channel.max()) for channel in inter]).astype(int)
    centers = ((normal_coefficients > normal_threshold) + max_pixels).astype(bool)

    thresholded = (centers * (I[...,1:] > color_sensitivity)).astype(int)
    thresholded *= np.dstack([segmented_cells] * (num_channels - 1))

    return thresholded


def get_boundaries(s, line_thickness=1):
    s = np.expand_dims(s.astype(np.int32), (0, -1))
    
    lr_kernel = np.array(([1] * line_thickness) + ([-1] * line_thickness))
    tb_kernel = np.array(([1] * line_thickness) + ([-1] * line_thickness))

    lr_kernel = tf.convert_to_tensor(np.expand_dims(lr_kernel, axis=(0, 2, 3)), dtype=tf.int32)
    tb_kernel = tf.convert_to_tensor(np.expand_dims(tb_kernel, axis=(1, 2, 3)), dtype=tf.int32)
    
    lr_edges = (tf.nn.conv2d(s, lr_kernel, strides=1, padding="SAME").eval(session=tf.compat.v1.Session()) == 0).astype(int)[0]
    tb_edges = (tf.nn.conv2d(s, tb_kernel, strides=1, padding="SAME").eval(session=tf.compat.v1.Session()) == 0).astype(int)[0]
    boundaries = (lr_edges + tb_edges != 2).astype(int) * 255
    
    zeros = np.zeros(boundaries.shape).astype(int)
    boundaries = np.dstack([boundaries, -boundaries, boundaries])
        
    return boundaries


def merge_channels(img, aqua_rgb):
    if img.shape[-1] == 3:
        return img
    assert img.shape[-1] == 4
    img = img[...,:-1] + np.dstack([coeff * img[...,-1] / 255 for coeff in aqua_rgb[::-1]])
    return np.minimum(img, 255).astype(np.uint8)


def cell_splice_segmentation(i, thresh, s, region):
    y_splice, x_splice = region.slice
    img_splice = i[y_splice.start:y_splice.stop,x_splice.start:x_splice.stop,:]
    thresh_splice = thresh[y_splice.start:y_splice.stop,x_splice.start:x_splice.stop,:]
    seg_splice = (s[y_splice.start:y_splice.stop,x_splice.start:x_splice.stop] == region.label).astype(int)
    return img_splice, thresh_splice, seg_splice, (y_splice, x_splice)



def get_scale(labeled_segmented_cells, target_median_nuclei_size):
    areas = [nuclei.area for nuclei in measure.regionprops(labeled_segmented_cells)]
    median_size = np.median(areas)
    scaling_factor = np.sqrt(target_median_nuclei_size / median_size)
#     print(f"Scale: {scaling_factor}")
    return scaling_factor

def count_blobs(fish_splice, cell_seg, min_cc_size):
    labeled_array, blob_count = scipy.ndimage.measurements.label(fish_splice * cell_seg)
    for blob in measure.regionprops(labeled_array):
        if blob.area < min_cc_size:
            blob_y_splice, blob_x_splice = blob.slice
            component = (labeled_array[blob_y_splice.start:blob_y_splice.stop, blob_x_splice.start:blob_x_splice.stop] == blob.label).astype(int)
            fish_splice[blob_y_splice.start:blob_y_splice.stop, blob_x_splice.start:blob_x_splice.stop] -= 255 * component
            blob_count -= 1
    return blob_count

def main(argv):    
    inpath = var['inpath']
    
    # Intensity and Normal Distribution Scaled Thresholds
    normal_threshold = stat_fish_params['normal_threshold']
    # intensity_threshold_std_coeff = var['intensity_threshold_std_coeff']
    color_sensitivity = stat_fish_params['color_sensitivity']

        
    # Image Resizing Parameters
    scaling_factor = var['scale']
    target_median_nuclei_size = stat_fish_params["target_median_nuclei_size"]
    kernel_shape = stat_fish_params['kernel_size']

    # Gaussian Kernel Parameters
    gaussian_sigma = stat_fish_params['gaussian_sigma']

    # Cosmetic: thickness of segmentation lines
    line_thickness = stat_fish_params['line_thickness']
    aqua_rgb = [233, 137, 54]
    
    # NuSeT parameters
    bbox_min_score = stat_fish_params['min_score'] 
    nms_thresh = stat_fish_params['nms_threshold']
    resize_scale = stat_fish_params['scale_ratio']
    nuclei_size_t = var['nuclei_size_T']
    flow_limit = stat_fish_params['flow_limit']
    cell_size_threshold_coeff = stat_fish_params['cell_size_threshold_coeff']


    #check input parameters
    if(os.path.isdir(os.path.join(inpath)) == False):
        print("Input folder does not exist. Exiting...")
        sys.exit(2)
     

    output_folder = f"tmp_{datetime.datetime.now().strftime('%m-%d_%H:%M:%S')}"
    if(os.path.exists(os.path.join(inpath, output_folder))):
        pass
    else:
        os.mkdir(os.path.join(inpath, output_folder))
        
    current_git_commit = sp.run('git log -1 | head -1', shell=True, capture_output=True).stdout.decode().strip().split(' ')[-1]
    shutil.copyfile('config.yaml', os.path.join(inpath, output_folder, f'config_{current_git_commit}.yaml'))
    shutil.copyfile('src/stat_fish_params.yaml', os.path.join(inpath, output_folder, 'stat_fish_params.yaml'))

    image_paths = get_imgs(inpath)
    first_fish = 'green'
    second_fish = 'red'
    third_fish = 'aqua'

    with tf.Graph().as_default():
        sess1, sess2, pred_masks, train_initial, pred_masks_watershed, resize_scale = load_nuset(bbox_min_score, nms_thresh, resize_scale)

        dfs = []
        for i in image_paths:
            path_split = os.path.split(i)
            print("Processing image: ", i)
            img_name = os.path.basename(i)[:-4]
            annotated_path = os.path.join(inpath, output_folder, img_name)
            os.makedirs(annotated_path, exist_ok=True)
            
            if i.endswith('.tif'):
                I = u16_to_u8(cv2.imread(i))
            elif i.endswith('.npy'):
                I = u16_to_u8(np.load(i))
            else:
                raise AssertionError
            blue = I[:,:,0]

            segmented_cells = nuclei_segment(blue, resize_scale, sess1, sess2, pred_masks, train_initial, pred_masks_watershed, nuclei_size_t)
            
            imheight, imwidth = segmented_cells.shape
            I = I[:imheight,:imwidth,:]
            segmented_cells = segmented_cells[:I.shape[0],:I.shape[1]] # In case of small resizing errors during upscaling

            # color_sensitivity = get_sensitivity(I, segmented_cells, intensity_threshold_std_coeff)
            if var['use_min_cut']:
                labeled_segmented_cells, labeled_segmented_cells_visualization = max_flow_binary_mask.binary_seg_to_instance_min_cut(segmented_cells, flow_limit, cell_size_threshold_coeff)
            else:
                labeled_segmented_cells = measure.label(segmented_cells, connectivity=None)

            regions = measure.regionprops(labeled_segmented_cells)
            
            scaling_factor = scaling_factor if scaling_factor != 'auto' else get_scale(labeled_segmented_cells, target_median_nuclei_size)
            
            segmented_cells_copy = segmented_cells.copy()
            num_channels = I.shape[-1]
            if not np.isnan(scaling_factor):
                gaussian_stdev = gaussian_sigma / scaling_factor
                min_cc_size = int(stat_fish_params['min_cc_size'] // (scaling_factor * scaling_factor))
                gaussian_kernel_shape = [int(dim // scaling_factor) if (dim // scaling_factor % 2) else int(dim // scaling_factor) + 1 for dim in kernel_shape]       

                thresholded = get_thresholded(I, segmented_cells, gaussian_stdev, normal_threshold, color_sensitivity, gaussian_kernel_shape)
            else:
                thresholded = np.zeros_like(I)[...,1:]
                gaussian_stdev = min_cc_size = gaussian_kernel_shape = np.nan
            thresholded_copy = thresholded.copy().astype(np.uint8)

            names = []; cell_sizes = []; centroids = [];  green_red_pixels = []; green_red_blobs = []
            
            fish_sizes, fish_blobs, avg_fish, max_fish = [[[] for _ in range(num_channels-1)] for _ in range(4)]
            df = pd.DataFrame()
            exec_summary = pd.DataFrame()
            
            for region in tqdm(regions):
                raw_cell, thresh_cell, cell_seg, (y_splice, x_splice) = cell_splice_segmentation(I, thresholded, labeled_segmented_cells, region)
                fish = [thresh_cell[...,channel] for channel in range(num_channels-1)]
                raw_fish = [raw_cell[...,channel].astype(np.int64) * cell_seg for channel in range(1, num_channels)]
                for raw_fish_ch, avg_fish_ch, max_fish_ch, fish_sizes_ch, fish_blobs_ch, fish_splice, color_sensitivity_ch in zip(raw_fish, avg_fish, max_fish, fish_sizes, fish_blobs, fish, color_sensitivity):         
                    
                    blob_count = count_blobs(fish_splice, cell_seg, min_cc_size)
                    
                    fish_blobs_ch.append(blob_count)
                    fish_pixels = (fish_splice * cell_seg).sum() / 255
                    assert fish_pixels == int(fish_pixels)
                    fish_sizes_ch.append(int(fish_pixels))
                    avg_fish_intensity, max_fish_intensity = intensity_metrics(raw_fish_ch)
                    avg_fish_ch.append(avg_fish_intensity if not np.isnan(avg_fish_intensity) else 0)
                    max_fish_ch.append(max_fish_intensity)
                
                cell_sizes.append(region.area)
                center = region.centroid
                centroids.append(str(int(center[0])) + '_' + str(int(center[1])))
                names.append(path_split[-1][:-4])
                
                green_red_splice = ((fish[0]) * (fish[1] / 255))
                blob_count = count_blobs(green_red_splice, cell_seg, min_cc_size)
                fish_pixels = (green_red_splice * cell_seg).sum() / 255
                assert fish_pixels == int(fish_pixels)
                green_red_pixels.append(int(fish_pixels))
                green_red_blobs.append(blob_count)

            df['image_name'] = np.array(names)
            df['nucleus_center'] = np.array(centroids)
            
            for channel_name, fish_sizes_ch, fish_blobs_ch, avg_fish_ch, max_fish_ch in zip((first_fish, second_fish, third_fish), fish_sizes, fish_blobs, avg_fish, max_fish): 
                df[f'#_FISH_pixels ({channel_name})'] = np.array(fish_sizes_ch)
                df[f'#_FISH_foci ({channel_name})'] = np.array(fish_blobs_ch)
                df[f'Avg fish intensity ({channel_name})'] = np.array(avg_fish_ch)
                df[f'Max fish intensity ({channel_name})'] = np.array(max_fish_ch)

            df['#_DAPI_pixels'] = np.array(cell_sizes)
            df[f'#_FISH_pixels (green and red)'] = np.array(green_red_pixels)
            df[f'#_FISH_foci (green and red)'] = np.array(green_red_blobs)
            dfs.append(df)
            
            thresholds_abbreviation = '_'.join([f"{letter}{format(x, '.1f')}" for letter, x in zip(['g', 'r', 'aq'], color_sensitivity)])
            image_least_squares_path = f"{annotated_path}/{img_name}_lsq_n{normal_threshold}_std{format(gaussian_stdev, '.2f')}_s{min_cc_size}_{thresholds_abbreviation}.tif"
            boundaries = get_boundaries(labeled_segmented_cells, line_thickness=line_thickness)
            
            I = merge_channels(I, aqua_rgb).astype(np.uint8)
            img_with_segmentation = np.minimum(I + boundaries, 255).astype(np.uint8)
            blob_labeled_img = np.dstack([boundaries[:,:,0], thresholded])
            if blob_labeled_img.shape[-1] > 3:
                blob_labeled_img = merge_channels(blob_labeled_img, aqua_rgb)
            blob_labeled_img = blob_labeled_img.astype(np.uint8)
            
            np.save(f"{annotated_path}/{img_name}__segmentation_min_cut.npy", labeled_segmented_cells)
            assert cv2.imwrite(f"{annotated_path}/{img_name}_segmentation.tif", segmented_cells_copy)
            if var['use_min_cut']:
                assert cv2.imwrite(f"{annotated_path}/{img_name}_segmentation_corrected_min_cut.tif", labeled_segmented_cells_visualization)
            assert cv2.imwrite(f"{annotated_path}/{img_name}_original_with_segmentation.tif", img_with_segmentation)
            assert cv2.imwrite(f"{annotated_path}/{img_name}_original.tif", I)
            assert cv2.imwrite(image_least_squares_path, blob_labeled_img)
            
        dfs = pd.concat(dfs)
        dfs.to_csv(os.path.join(path_split[0],  output_folder, 'stat_fish_lsq.csv'), index=False)
        sess1.close()
        sess2.close()

    if os.path.isdir(f"{inpath}/annotated"):
        os.rename(f"{inpath}/annotated", f"{inpath}/annotated_{str(datetime.datetime.now())[5:-10].replace(' ', '-')}")
    os.rename(f"{inpath}/{output_folder}", f"{inpath}/annotated")

if __name__ == "__main__":
    main(sys.argv[1:])

