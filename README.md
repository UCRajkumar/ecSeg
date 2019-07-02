# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

Reference implementation of methods outlined in ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.
```
ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA
Utkrisht Rajkumar, Kristen Turner, Jens Luebeck, Viraj Deshpande, Manmohan Chandraker, Paul Mischel, and Vineet Bafna
```

## Installation
This platform was built using Python 3.6.7. 

To download project dependencies, execute: 

```
pip install -r requirements.txt
```

## Dataset
Training and test dataset can be downloaded from:
```
https://drive.google.com/open?id=10owNEZA1vrbNcunPfve1rHlwPalNnXmB
```

## Run ecSeg
To produce segmentations, run ecSeg.py:
```
python ecSeg.py -i "input_path"
```

### Input specifications
1. input_path (must be enclosed by double quotes ""). For example: `python ecSeg.py -i "C:\Users\Utkrisht\images_folder"`
2. Software will only read the `.tif` images
3. Input images must be `1040x1392x3` (RGB images)

Note, support for different dimensions coming soon.

### Output 
1. **Coordinates folder** will be created which will contain a coordinate file for each image. Each coordinate file will have the coordinates of all the ecDNA present in the corresponding image in the form `(x, y)`.
2.  **Labels folder** will contain the RGB version of the post-processed segmentation. It will also contain raw values saved as a `.npy` file.

#### Segmentation details

The segmented images (`.npy`) will be in dimension `1024x1280`.

To extract individual classes (`example_seg.npy` can be found in "example results" folder):

```
seg_I = np.load('example_seg.npy')
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
`-i` | Path to folder containing images. Must be wrapped in double quotes. See above.
`-c` | Fish Color. Must be 'green' or 'red'
`-t` | Threshold value. Threshold values must be [0, 255]. Indicates sensitivity of FISH interaction. 0 and 255 are the least and highest sensitivity, respectively
`-p` | Segment boolean. Must be 'True' or 'False'. Indicates whether to re-segment images. Enter 'False' if you have already segmented the images

