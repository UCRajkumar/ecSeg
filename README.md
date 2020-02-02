# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

This branch is the latest development of ecSeg. Updates: Replacement of convolutional blocks with residual blocks and the incorporation of [mutli-scale context aggregation by dilated convolutions](https://arxiv.org/abs/1511.07122).

Rajkumar, U. et al. *ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.* iScience. 21, 428-435. (2019)

## Installation
This platform requires a modern processor with support for AVX instructions and python 3.5 or greater. 

We highly recommend installing ecSeg through a conda environemnt.
```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
conda create -n ecseg python=3.7 opencv tqdm scikit-image keras Pillow=5.4.1 matplotlib=3.0.3
conda activate ecseg
```

## Run ecSeg
To produce segmentations, run ecSeg.py:
```
python3 ecSeg.py -i "input_path"
```

### Input specifications
1. input_path must be enclosed by double quotes "". For example: `python3 ecSeg.py -i "C:/Users/Utkrisht/path/to/images"`
2. Software will only read the `.tif` images
3. Input images will be resized to `1040x1392`
4. Optionally, you can provide a model name using "-m"

Note: The flag, -i, must be provided.

### Output 
1. **Coordinates folder** will be created which will contain a coordinate file for each image. Each coordinate file will have the coordinates of all the ecDNA present in the corresponding image in the form `(x, y)`.
2.  **Labels folder** will contain the RGB version of the post-processed segmentation. It will also contain raw values saved as a `.npy` file.

#### Segmentation details

The segmented images will be in dimension `1024x1280`.

To extract individual classes:

```
seg_I = np.load('example_results/seg.npy')
background = (seg_I==0)
nuclei = (seg_I==1)
chromosome = (seg_I==2)
ecDNA = (seg_I==3)
```

## Run ecSeg_fish
To analyze fish interaction run ecSef_fish.py:
```
python3 ecSeg_fish.py -i "input_path"
```

### Input specifications

Arguments | Description 
---| ---|
`-h` | Displays argument options
`-i` | Path to folder containing images. Must be wrapped in double quotes. See example above on how to run ecSeg.py.
`-c` | Fish color (optional). Must be 'green' or 'red'
`-t` | Threshold value (optional). Threshold values must be [0, 255]. Indicates sensitivity of FISH interaction. 0 and 255 are the least and highest sensitivity, respectively
`-p` | Segment boolean (optional). Must be 'True' or 'False'. Indicates whether to re-segment images. Enter 'False' if you have already segmented the images

## Training/test Dataset
Training and test dataset can be downloaded from:
```
https://data.mendeley.com/datasets/m7n3zvg539/draft?a=30ace699-6d6a-4c49-a770-29b09f759795
```

Dataset | Description
---|---|
train_im| RGB patches used to train neural network 
train_mask| mask for train_im patches 
test_im|  RGB patches used to evaluate neural network 
test_mask| mask for test_im patches 
full images | full sized images used for evaluation
