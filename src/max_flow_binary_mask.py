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
from skimage import *
import skimage

from model_layers.models import UNET
from model_layers.model_RPN import RPN
from model_layers.anchor_size import anchor_size
from model_layers.rpn_proposal import RPNProposal
from model_layers.marker_watershed import marker_watershed

from nuset_utils.anchors import generate_anchors_reference
from nuset_utils.generate_anchors import generate_anchors
from nuset_utils.normalization import *
from skimage.measure import label, regionprops, find_contours
from collections import deque
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import random

class Edge:
    def __init__(self, start, end, capacity, flow=0):
        self.start = start
        self.end = end
        self.capacity = capacity
        self.flow = flow
        self.reverse = None # pointer to reverse edge

def add_forward_reverse_edges(residual_graph, start_node, end_node, forward_capacity=1):
    # Creates a forward and reverse edge(capacity 0) 
    # for each edge in path matrix
    forward_edge = Edge(start_node, end_node, forward_capacity)
    reverse_edge = Edge(end_node, start_node, 0)

    # Pairs the edges by making each edge the reverse of the other
    forward_edge.reverse = reverse_edge
    reverse_edge.reverse = forward_edge

    # Adds to Residual Graph
    residual_graph[start_node].append(forward_edge)
    residual_graph[end_node].append(reverse_edge)

def get_graph(img, start, target, dist):
    residual_graph = defaultdict(list)
    neighbors = lambda y, x: [(y+y_diff, x+x_diff) for y_diff, x_diff in ((1, 0), (0, 1), (-1, 0), (0, -1)) if (0 <= y+y_diff < len(img) and 0 <= x+x_diff < len(img[0]))]
    for i, row in enumerate(img):
        for j, intensity in enumerate(row):
            if intensity and (i, j) != start and (i, j) != target:
                if (i, j) != start and np.linalg.norm(np.array(start) - np.array((i, j)), ord=1) <= dist:
                    add_forward_reverse_edges(residual_graph, start, (i, j), 1)
                elif (i, j) != target and np.linalg.norm(np.array(target) - np.array((i, j)), ord=1) <= dist: 
                    add_forward_reverse_edges(residual_graph, (i, j), target, 1)
                for neighbor in neighbors(i, j):
                    if img[neighbor[0], neighbor[1]]:
                        add_forward_reverse_edges(residual_graph, (i, j), neighbor, 1)
    return residual_graph

def bfs(residual_graph, start, target, return_reachable=False):
    prev = {start: None}
    queue = deque([start])
    while queue:
        curr = queue.pop()
        for edge in residual_graph[curr]:
            if edge.end not in prev and edge.flow < edge.capacity:
                prev[edge.end] = edge
                queue.appendleft(edge.end)
    if return_reachable:
        return set(prev.keys())
                
    path = [prev[target]] if target in prev else []
    while path and path[-1].start != start:
        path.append(prev[path[-1].start])
    return list(reversed(path))

def max_flow(res, start, target):
    # stores current flow
    current_flow = 0
    edge_path = bfs(res, start, target)
    while edge_path:
        flow_difference = float('inf')
        for edge in edge_path:
            flow_difference = min(flow_difference, edge.capacity - edge.flow)
        
        for edge in edge_path:
            edge.flow += flow_difference
            edge.reverse.flow -= flow_difference
        current_flow += flow_difference
        edge_path = bfs(res, start, target)
    return current_flow

def partition_min_cut(img, res, start, target):
    current_flow = max_flow(res, start, target)
    group_1 = np.zeros_like(img)
    for y, x in bfs(res, start, target, return_reachable=True):
        group_1[y, x] = 1
    group_2 = img - group_1
    return group_1, group_2


def segment_min_cut(mask, centers, dist):
    if not centers:
        return []
    elif len(centers) == 1:
        return [mask]
    center_1, center_2 = centers[:2]
    res = get_graph(mask, center_1, center_2, dist)
    group_1, group_2 = partition_min_cut(mask, res, center_1, center_2)
    if group_1.sum() == 1:
        group_1 = np.zeros_like(mask)
        group_2 = mask
        centers.remove(center_1)
    elif group_2.sum() == 1:
        group_2 = np.zeros_like(mask)
        group_1 = mask
        centers.remove(center_2)
    
    color_1_group = [x for x in centers if group_1[x[0], x[1]]]
    color_2_group = [x for x in centers if group_2[x[0], x[1]]]
    groups_1 = segment_min_cut(group_1, color_1_group, dist)
    groups_2 = segment_min_cut(group_2, color_2_group, dist)
    return groups_1 + groups_2

def get_centers(segmented_cells, min_rad=10):
#     segmented_cells = 255 * (tf.nn.conv2d(np.expand_dims(segmented_cells.astype(np.int32), (0, 3)), np.ones((3, 3, 1, 1)), strides=1, padding='SAME')[0,...,0].numpy() > 0).astype(np.uint8)
    distance_transformed = cv2.distanceTransform(segmented_cells.astype(np.uint8), cv2.DIST_L1, 3)
    distance_transformed = np.expand_dims(distance_transformed, (0, 3))
    grad = [segmented_cells[1:-1,1:-1]]
    kernel = [1, -1]
    for dim in (0, 1):
        s_kernel = np.expand_dims(kernel, (1-dim, 2, 3))
        conv = tf.nn.conv2d(distance_transformed, s_kernel, strides=1, padding='VALID')[0, :, :, 0].numpy() 
        if not dim:
            boolean = (conv[1:,1:-1] >= 0) * (conv[:-1,1:-1] <= 0)
        else:
            boolean = (conv[1:-1,1:] >= 0) * (conv[1:-1,:-1] <= 0)     
        grad.append(boolean)
        
        
    kernel = [
        [1, 0],
        [0, -1]
    ]
    s_kernel = np.expand_dims(kernel, (2, 3))
    conv = tf.nn.conv2d(distance_transformed, s_kernel, strides=1, padding='VALID')[0, :, :, 0].numpy()
    grad.append((conv[1:,1:] >= 0) * (conv[:-1,:-1] <= 0))
    
    kernel = [
        [0, 1],
        [-1, 0]
    ]   
    s_kernel = np.expand_dims(kernel, (2, 3))
    conv = tf.nn.conv2d(distance_transformed, s_kernel, strides=1, padding='VALID')[0, :, :, 0].numpy()
    grad.append((conv[1:,:-1] >= 0) * (conv[:-1,1:] <= 0))
    grad.append(distance_transformed[0,1:-1,1:-1,0] > min_rad)
    grad = np.expand_dims(np.prod(np.array(grad), axis=0), (0, 3)).astype(np.int32)
    
    min_rad = max((distance_transformed[0,1:-1,1:-1,0][grad[0,...,0] > 0]).min(), min_rad)
    centers = 255 * (distance_transformed[0,1:-1,1:-1,0] >= min_rad)
    return np.pad(centers, 1)


def binary_seg_to_instance_min_cut(segmented_cells, flow_limit, cell_size_threshold_coeff):
    labeled_segmented_cells, num_cells = skimage.measure.label(segmented_cells, connectivity=1, return_num=True)
    areas = [region.area for region in skimage.measure.regionprops(labeled_segmented_cells)]
    expected_cell_size = np.median(areas)
    distance = (-1 + int(np.sqrt(1 + (2 * flow_limit)))) // 2
    print(f"DISTANCE: {distance}")
    assert distance > 0
    # print(f"MAX NUMBER OF EDGES REMOVED: {2 * distance * (distance + 1)}")

    visualization = np.zeros_like(labeled_segmented_cells)
    updated_labeled_segmented_cells = labeled_segmented_cells.copy()

    for region in tqdm(skimage.measure.regionprops(labeled_segmented_cells)):
        mask = (labeled_segmented_cells[region.slice] == region.label).astype(int)
        composite, cells = mask, [0]
        if region.area > cell_size_threshold_coeff * expected_cell_size:

            center_conv = get_centers(mask)

            center_ls = []

            labeled_center_conv = skimage.measure.label(center_conv, connectivity=2)
            for cell_region in skimage.measure.regionprops(labeled_center_conv):
                centroid = (np.round(cell_region.centroid)).astype(int)
                if not mask[centroid[0], centroid[1]]:
                    alternatives = [(i, j) for i, row in enumerate(labeled_center_conv) for j, val in enumerate(row) if val == cell_region.label]
                    alternative = alternatives[random.randint(0, len(alternatives)-1)]
                    assert mask[alternative[0], alternative[1]]
                    print(f"PICKED ALTERNATE CENTROID {centroid} {alternative}")
                    centroid = alternative
                center_ls.append(centroid)
            center_ls = list(map(lambda center: tuple(np.round(center).astype(int)), center_ls))
            
            
            ##### REMOVE LATER
            center_vis = np.zeros_like(mask)
            for center in center_ls:
                center_vis[center] = 1

            composite = mask.copy()
            if len(center_ls) > 1:
                composite = np.zeros_like(mask)
                cells = segment_min_cut(mask, center_ls, dist=distance)
                updated_labeled_segmented_cells[region.slice] -= mask * region.label

                for i, cell in enumerate(cells, start=1):
                    if i == 1:
                        updated_labeled_segmented_cells[region.slice] += cell * region.label
                    else:
                        num_cells += 1
                        updated_labeled_segmented_cells[region.slice] += cell * (num_cells)
                    composite += cell * i
            
            
            
#             fig, axs = plt.subplots(1, 3, dpi=150)            
#             axs[0].imshow(np.dstack([mask * 255, center_conv * 255, np.zeros_like(mask)]))
#             axs[1].imshow(np.dstack([mask * 255, center_vis * 255, np.zeros_like(mask)]))
#             axs[2].imshow(composite)
#             for axis in axs:
#                 axis.axis(False)
#             plt.show()
        visualization[region.slice] += (composite * (255 // len(cells)))
    assert num_cells == updated_labeled_segmented_cells.max()
    return updated_labeled_segmented_cells, visualization

