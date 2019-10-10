# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

Reference implementation of methods outlined in ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.
```
ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA
Utkrisht Rajkumar, Kristen Turner, Jens Luebeck, Viraj Deshpande, Manmohan Chandraker, Paul Mischel, and Vineet Bafna
```

## Installation
This platform requires a modern processor with support for AVX instructions and python 3.5 or greater. 

We highly recommend installing ecseg through a conda environemnt.
```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
conda create -n ecseg python=3.7
source activate ecseg
conda install Pillow tqdm scikit-image
conda install -c menpo opencv 
pip install keras=2.2.5
pip install matplotlib=2.2.4
```

If not using conda, execute the following commands from a terminal: 

```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
pip3 install -r requirements.txt
```

## Run ecSeg
To produce segmentations, run ecSeg.py:
```
python3 ecSeg.py -i "input_path"
```

### Input specifications
1. input_path (must be enclosed by double quotes ""). 

For example: `python3 ecSeg.py -i "C:/Users/Utkrisht/path/to/images"`
2. Software will only read the `.tif` images
3. Input images must be `1040x1392x3` (RGB images)

Note: The flag, -i, must be provided.

### Output 
1. **Coordinates folder** will be created which will contain a coordinate file for each image. Each coordinate file will have the coordinates of all the ecDNA present in the corresponding image in the form `(x, y)`.
2.  **Labels folder** will contain the RGB version of the post-processed segmentation. It will also contain raw values saved as a `.npy` file.

#### Segmentation details

The segmented images (`.npy`) will be in dimension `1024x1280`.

To extract individual classes (`seg.npy` can be found in "example_results" folder):

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
