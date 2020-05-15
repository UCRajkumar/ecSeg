# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

ecSeg is a tool used to segment and analyze metaphase images containing ecDNA. It also has an extension to analyze FISH probes on metaphase images. ecSeg is the latest version of ECdetect used to perform the analysis in [Extrachromosomal oncogene amplification drives tumour evolution and genetic heterogeneity, Turner 2017](https://www.nature.com/articles/nature21356). 

Rajkumar, U. et al. *ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.* iScience. 21, 428-435. (2019)

## Installation
This platform requires a modern processor with support for AVX instructions and python 3.5 or greater. 

We highly recommend installing ecSeg through a conda environemnt.
```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
conda create -n ecseg python=3.7 opencv tqdm scikit-image keras Pillow=5.4.1 matplotlib=3.0.3 pandas
conda activate ecseg
```

## Run ecSeg
To produce segmentations, run ecSeg.py:
```
python ecSeg.py -i "input_path"
```

### Input specifications
1. input_path must be enclosed by double quotes "". For example: `python3 ecSeg.py -i "C:/Users/Utkrisht/path"`
2. Software will only read the `.tif` images in the input folder
4. Optionally, you can provide a model name using "-m" if you train a new model.

Note: The flag, "-i," must be provided.

### Output 
1. **Coordinates folder** - Contains coordinate files for each image. Each coordinate file will have the coordinates of all the ecDNA present in the corresponding image in the form `(x, y)`.
2.  **Labels folder** - Contains the RGB version of the post-processed segmentation and the raw values saved as a `.npy` file.
3. **dapi folder** - Contains the gray-scale dapi version of the images.
4. **num_ecDNAs.csv** - Contains a list of all the images that were processed and the number of ecDNA in each image. 

#### Segmentation details

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
python ecSeg_fish.py -i "input_path"
```

### Input specifications

Arguments | Description 
---| ---|
`-h` | Displays argument options
`-i` | Path to folder containing images. Must be wrapped in double quotes. See example above on how to run ecSeg.py.
`-c` | Fish color (optional). Must be 'green' or 'red'. Default color is green.
`-t` | Threshold value (optional). Threshold values must be [0, 255]. Indicates sensitivity of FISH interaction. 0 and 255 are the least and highest sensitivity, respectively
`-p` | Segment boolean (optional). Must be 'True' or 'False'. Indicates whether to re-segment images. Enter 'False' if you have already segmented the images
`-m` | Model name (optional). Name of the trained model. Must have '.h5' extension.

### Output
1. All the outputs from ecSeg.py. See above.
2. **green folder** will contain the gray-scale version of the green fish signal.
3. **red folder** will contain the gray-scale version of red fish signal.
4. **ec_fish.csv** will contain fish interaction stats for each image. See [ecSeg_fish_analysis](https://github.com/UCRajkumar/ecSeg/edit/master/ecSeg_fish_analysis.md). 
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
