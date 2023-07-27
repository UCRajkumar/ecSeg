#!/usr/bin/env python3

import os, sys, glob
import yaml
import numpy as np
import pandas as pd
import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import skimage
import seaborn as sns


def get_distances_img(lsq, segmentation, presets):
    centromere_probe_index, fish_probe_index, max_centromeric_spots = presets
    distances = []
    for cell in skimage.measure.regionprops(segmentation):
        seg_cutout = (segmentation[cell.slice] == cell.label).astype(int)
        if (lsq[cell.slice][...,0] * seg_cutout).any() and (lsq[cell.slice][...,1] * seg_cutout).any():
            sqrt_cell_area = np.sqrt(seg_cutout.sum())
            lsq_cutout = lsq[cell.slice] * np.expand_dims(seg_cutout, 2)
            
            grid = np.dstack(np.meshgrid(*[np.arange(dim) for dim in seg_cutout.shape[::-1]]))
            distance_transformed = np.zeros(seg_cutout.shape)

            fish_probe = lsq_cutout[...,fish_probe_index].astype(bool)
            centromere_probe = lsq_cutout[...,centromere_probe_index].astype(bool)

            labeled_fish_probe = skimage.measure.label(fish_probe)
            if labeled_fish_probe.max() > max_centromeric_spots:
                continue

            fish_coords = grid[fish_probe.astype(bool)]
            centromere_coords = grid[centromere_probe.astype(bool)]

            for fish_coord in fish_coords:
                distance_transformed[fish_coord[1], fish_coord[0]] = (np.linalg.norm(centromere_coords - fish_coord, axis=1).min() / sqrt_cell_area)
        
        
            distances.append(float('inf'))
            for spot in skimage.measure.regionprops(labeled_fish_probe):
                spot_cutout = labeled_fish_probe[spot.slice] == spot.label
                distances[-1] = min(distances[-1], distance_transformed[spot.slice][spot_cutout].min())
    return distances


def get_distances_path(root_directory, *presets):
    distances = []
    for img_path in tqdm.tqdm(glob.glob(f"{root_directory}/*.tif")):
        img_name = os.path.basename(img_path)[:-4]
        img_directory = f'{root_directory}/annotated/{img_name}'
        assert os.path.isdir(img_directory)
        segmentation_path = f'{img_directory}/{img_name}__segmentation_min_cut.npy'
        lsq_path = glob.glob(f'{img_directory}/{img_name}_lsq*.tif')[0]

        segmentation = np.load(segmentation_path)
        lsq = skimage.io.imread(lsq_path)

        distances.append(get_distances_img(lsq, segmentation, presets))
    return [y for x in distances for y in x]


def main():
    with open('config.yaml') as infile:
        var = yaml.safe_load(infile)['fish_distance_calculation']

    color_to_index = {
        'red': 0,
        'green': 1,
        'blue': 2
    }


    directory = var['inpath']
    assert os.path.exists(f'{directory}/annotated')

    centromere_probe_index = color_to_index[var['centromere_probe_color']]
    fish_probe_index = color_to_index[var['fish_probe_color']]
    max_centromeric_spots = var['max_centromeric_spots']

    distances = get_distances_path(directory, centromere_probe_index, fish_probe_index, max_centromeric_spots)
    pd.DataFrame({'normalized_distance': distances}).to_csv(f'{directory}/centromere_distances.csv', index=False)


if __name__ == '__main__':
    main()


