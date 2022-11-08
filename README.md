# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

[BBI DEVELOPMENTAL BRANCH]

This repository is the official version of ecSeg, a tool used to quantify ecDNA in DAPI-stained images. It also has an extension to analyze FISH probes. Please cite (Bibtex at the bottom) the following reference if using ecSeg in your work: 

Rajkumar, U. et al. *ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.* iScience. 21, 428-435. (2019)

## Directory structure

| File             | Description                                 |
| ---------------- | ------------------------------------------- |
| config.yaml      | File to set parameters for different tasks  |
| Makefile         | Makefile to run the different scripts       |
| env.yml | Yaml file for setting up environment |
| setup.sh         | Script to install ecSeg package             |

...

| Folder | Description                        |
| ------ | ---------------------------------- |
| src    | Contains python scripts            |
| models | Contains required models. Download models.zip from [here](https://data.mendeley.com/public-files/datasets/m7n3zvg539/files/dd0cdd8a-9763-4a82-adcd-62be932e85ad/file_downloaded) and unzip the folder inside the ecseg/ folder|
| example | Example images to test ecSeg       |

## Installation

This platform requires a modern processor with support for AVX instructions, python 3.5+, and conda installed. 

```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
conda env create -f env.yml
conda activate ecseg
pip install numpy==1.19.2
pip install imagecodecs
```

Always make sure the `ecseg` environment is activated before executing any tasks:

```
conda activate ecseg
```

## Input Image Specifications

1.  Software will only read the `.tif` images in the input folder.
2.  FISH analysis will only consider green and red FISH

## Tasks

### `make metaseg`

Segment metaphase images stained with DAPI. Identified background, nuclei, chromosomes, and ecDNA.

Set parameters in config.yaml under `metaseg`:

`inpath : path to folder containing images (make sure path is incapsulated inside single quotes. e.g. `path\to\folder`)`

#### Output

1.  **labels folder** - Contains the RGB version of the post-processed segmentation and the raw values saved as a `.npy` file.
2. **dapi folder** - Contains the gray-scale dapi version of the images.
3. **ec_quantifications.csv** - Contains a list of all the images that were processed and the number of ecDNA in each image. 

#### Segmentation details

To extract individual classes:

```
seg_I = np.load('example_results/seg.npy')
background = (seg_I==0)
nuclei = (seg_I==1)
chromosome = (seg_I==2)
ecDNA = (seg_I==3)
```

### `make meta_overlay`

Analyze fish interaction on metaphase images stained with DAPI. Supports green and red FISH. `make metaseg` must be run before `make meta_overlay` can be executed.

Set parameters in config.yaml under `meta_overlay`:

````
inpath : path to folder containing images
color_sensitivity : Sensitivity to FISH color. Value between 0 (most sensitive) and 255 (least sensitive)
````

#### Output

2. **green folder** will contain the gray-scale version of the green fish signal.
3. **red folder** will contain the gray-scale version of red fish signal.
3. **fish_quantification.csv** will contain fish interaction stats for each image. Column headers are as follows:
    1. “image_name” - Name of image
    2. “# of ecDNA” - # of ecDNA based on DAPI only. 
    3. “# of ecDNA (FISH_color)” - # of ecDNA based on that FISH color only.
    4. “# of ecDNA(DAPI and  FISH_color)” - # of ecDNA based on DAPI colocated with that FISH_color
    5. “# of HSR (FISH)” - # of homogeneously stained regions based on FISH_color.


### `make nuclei_fish`

Identifies nuclei in the image and analyzes ratio of fish to dapi pixels. Provides rough approximation of oncogene amplification per cell in interphase images. Supports green and red FISH. 

```
ecseg
|
|--models
|--|  metaseg.h5
|  |--nuset
|     |  foreground.ckpt.data-00000-of-00001
|     |  foreground.ckpt.index
|     |  ...
|--src
|  |  ...
```

Set parameters in config.yaml under `nuclei_fish`:

````
inpath : path to folder containing images
color_sensitivity : Sensitivity to FISH color. Value between 0 (most sensitive) and 255 (least sensitive)
min_score : 0.95
nms_threshold : Threshold to suppress oversegmentation. (Recommended value: 0.01)
scale_ratio : Ratio of cell size to image size. (Recommended value: 0.4)
nuclei_size_t : Size threshold for finding nuclei. Nuclei smaller than this value will be considered erroneous signals. Recommendation: cultured images, use 5000. For tissue use 500.
````

#### Output

1. **nuclei_fish.csv** - Each row represents a single nucleus. Column headers are as follows:
    1. “image_name” - Name of image
    2. “nuclei_center” - Center of each nuclei
    3. “#_FISH_pixels (FISH_color)" - # of FISH pixels inside nuclei
    4. “#_FISH_blobs (FISH_color)" - # of FISH connected components. 
    5. “#_DAPI_pixels” - # of DAPI pixels, i.e. size of nucleus.

**PROBABILITY OF ANEUPLOIDY** The probability of aneuploidy can be computed by averaging of the #_FISH_blobs for the centromere probe. If the average is close to 2, then there is low evidency of aneuploidy. If the average is >>2, then there is a high probability of aneuploidy. Visually inspect to verify that the cell segmentation predominantly segmented each individual cell. If the image contains extremely clumped cells, the segmentations might be inaccurate and consequently the aneuploidy score will also be skewed.

**IDENTIFYING POOR STAINING ARTIFACTS** If the #_FISH_pixels for the oncogene (or centromere) probe is approximately equal to the #_DAPI_pixels then this is typically a cell with poor staining.


## Bibtex
```
@article{Rajkumar2019,
    author = {Rajkumar, Utkrisht and Turner, Kristen and Luebeck, Jens and Deshpande, Viraj and Chandraker, Manmohan and Mischel, Paul and Bafna, Vineet},
    journal = {iScience},
    title = {{EcSeg: Semantic Segmentation of Metaphase Images Containing Extrachromosomal DNA}},
    url = {https://github.com/ucrajkumar/ecseg},
    year = {2019}
}
```
