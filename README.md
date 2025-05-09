# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

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
| models | Contains required models. Download models.zip from [here](https://data.mendeley.com/datasets/m7n3zvg539) and unzip the folder inside the ecseg/ folder|
| example | Example images to test ecSeg       |

## Installation

This platform requires a modern processor with support for AVX instructions, python 3.5+, and conda installed. 

```
git clone https://github.com/ucrajkumar/ecSeg
cd ecSeg
conda env create -f env.yml
conda activate ecseg
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


### `make stat_fish`

Identifies nuclei in the image and analyzes ratio of fish to dapi pixels. Provides rough approximation of oncogene amplification per cell in interphase images. Supports green and red FISH. 

Recommended folder structure:
```
ecseg
|
|--models
|--|  metaseg.h5
|--|--nuset
|     |  foreground.ckpt.data-00000-of-00001
|     |  foreground.ckpt.index
|     |  ...
|--src
|  |  ...
```

Set parameters in config.yaml under `stat_fish`:

````
inpath: path to folder containing images
scale: Average square root of ratio of nuclei size relative to target_median_nuclei_size (Recommended value: 1)
use_min_cut: Whether to use the min_cut algorithm to perform instance segmentation to break up overlapping nuclei.
nuclei_size_t: Size threshold for finding nuclei.

Note: Nuclei smaller than nuclei_size_t will be considered erroneous signals. 
Rec: cultured images, use nuclei_size_t=5000. For tissue, use nuclei_size_t=500.
````

#### Output

1. **stat_fish_lsq.csv** - Each row represents a single nucleus. Column headers are as follows:
    1. “image_name” - Name of image
    2. “nuclei_center” - Center of each nucleus
    3. “#_FISH_pixels (FISH_color)" - # of FISH pixels inside nucleus
    4. “#_FISH_foci (FISH_color)" - # of FISH connected components.
    5. “Avg fish intensity (FISH_color)" - Avg intensity of the corresponding FISH pixels inside nucleus
    6. “Max fish intensity (FISH_color)" - Max intensity of the corresponding FISH pixels inside nucleus
    7. “#_DAPI_pixels” - # of DAPI pixels, i.e. size of nucleus.

1. **annotated/img_name/img_name_lsq.tif**
    1. Visualization of fish foci calls for each fish probe with cell segmentation.
2. **annotated/img_name/img_name_original_with_segmentation.tif**
    1. Visualization of nuclei segmentation outline overlayed with original image.
3. **annotated/img_name/img_name_segmentation.tif**
    1. NuSet Binary Segmentation visualization.
4. **annotated/img_name/img_name_segmentation_corrected_min_cut.tif**
    1. NuSet Binary Segmentation after Min Cut algorithm refinement, where each color represents a distinct nuclei.


### `make interseg`
Predicts the probability of each nucleus having no amplification, HSR amplification, and ecDNA amplification for the oncogene (i.e. FISH probe) of interest.

Recommended folder structure:
```
ecseg
|
|--models
|--|  metaseg.h5
|--|  interseg
|--|--nuset
|     |  foreground.ckpt.data-00000-of-00001
|     |  foreground.ckpt.index
|     |  ...
|--src
|  |  ...
```

Set parameters in config.yaml under `interseg`:

````
inpath : path to folder containing images
FISH_color : Fish probe of interest ('green' or 'red')
has_centromeric_probe: Whether to input both the target and centromeric probes (ecSeg-i and ecSeg-c) or in target probe only mode (ecSeg-i).
````

#### Output

1. **interphase_prediction_(FISH_color).csv** - Each row represents a single nucleus. Column headers are as follows:
    1. “image_name” - Name of image
    2. “nuclei_center” - Center of each nucleus
    3. “ecSeg-i_label" - Prediction value using target probe only (No-amp, EC-amp, HSR-amp)
    4. “ecSeg-c_label" - Prediction value of focal-amplification/no-amplification using target and centromeric probe (No-amp, Focal-amp)
    5. “interSeg_label" - Combined ecSeg-i and ecSeg-c prediction (No-amp, EC-amp, HSR-amp)

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
