# ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA

Reference implementation of methods outlined in ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA.
```
ecSeg: Semantic Segmentation of Metaphase Images containing Extrachromosomal DNA
Utkrisht Rajkumar, Kristen Turner, Jens Luebeck, Viraj Deshpande, Manmohan Chandraker, Paul Mischel, and Vineet Bafna
```

# Installation
This platform was built using Python 3.6.7. 

To download project dependencies, execute: 

```
pip install requirements.txt
```

Training and test dataset can be downloaded from:
```
https://drive.google.com/open?id=10owNEZA1vrbNcunPfve1rHlwPalNnXmB
```

To produce segmentations, run ecSeg.py:
```
python ecSeg.py -i input_path
```

### Input specifications
1. `input_path` (must end in "\")
2. Software will only read the `.tif` images
3. Input images must be `1040x1392x4`

Note, support for different dimensions coming soon.

### Output specifications
1. Coordinates folder will be created which will contain a coordinate file for each image. Each coordinate file will have the coordinates of all the ecDNA present in the corresponding image in the form `(x, y)`.
2.  Labels folder will contain the RGB version of the post-processed segmentation. It will also contain raw values saved as a `.npy` file.


