# ecSeg_fish.py output analsysis

One of the outputs produced by `ecSeg_fish.py` is `ec_fish.csv`. The columns in this folder are explained below:

1. `image_name` - Image name
2. `ec_pixels` - Total number of ecDNA pixels
3. `chrom_pixels` - Total number of chromosomal pixels
4. `fish_pixel(<fish color>)` - Total number of fish pixels (fish pixels on nuclei are not considered). The fish color is determined by what is provided as the input argument. 
5. `ec+fish pixels` - Total number of fish pixels co-localized with ecDNA pixels
6. `chrom+fish pixels` - Total number of fish pixels co-localized with chromosomal pixels
7. `# of ecDNA` - Total number of ecDNA (not ecDNA pixels, but full-sized ecDNA components)
8. `# of ecDNA + fish` - Total number of ecDNA co-localized with fish.

For other stat requests, please email urajkuma@eng.ucsd.edu
