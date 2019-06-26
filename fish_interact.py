import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import pandas as pd
import seaborn as sns
import cv2
'''
path = './ec/'
imgs = []
tot = []
for f in os.listdir(path): #get all images in path
    ext = os.path.splitext(f)[1]
    if ext.lower() == '.tif':
        tot.append()
'''
path = './hsr/'
img_name = []
tot_ec = []
tot_chrom = []
egfr_ec_ratio = []
egfr_chrom_ratio = []
im_type = []
tot_fish = []
egfr_ec = []
egfr_chrom = []
for f in os.listdir(path):
	ext = os.path.splitext(f)[1]
	if ext.lower() == '.tif':
		img_name.append(f)
		im_type.append('hsr')
		A = np.load(('./labels/'+f+'.npy'))
		B = Image.open(('./hsr/' +f))
		red, green, blue = B.split()
		cv2.imwrite(('./dapi/'+f),cv2.bitwise_not(np.uint8(blue)))
		nuc = ~(A==1)
		green = (np.array(green) > 120)[:1024,:1280]
		green = green * nuc
		ec = (A==3)
		chrom = (A==2)
		tot_fish.append(len(np.where(green)[0]))
		tot_ec.append(len(np.where(ec)[0]))
		tot_chrom.append(len(np.where(chrom)[0]))
		egfr_ec.append(len(np.where((green*ec))[0]))
		egfr_chrom.append(len(np.where((green*chrom))[0]))
		if(tot_fish[-1]==0):
			egfr_ec_ratio.append(0)
		else:
			egfr_ec_ratio.append(len(np.where((green*ec))[0])/tot_fish[-1])
		if(tot_fish[-1]==0):
			egfr_chrom_ratio.append(0)
		else:
			egfr_chrom_ratio.append(len(np.where((green*chrom))[0])/tot_fish[-1])
		

path = './ec/'
for f in os.listdir(path):
	ext = os.path.splitext(f)[1]
	if ext.lower() == '.tif':
		img_name.append(f)
		im_type.append('ec')
		A = np.load(('./labels/'+f+'.npy'))
		B = Image.open(('./ec/' +f))
		red, green, blue = B.split()
		cv2.imwrite(('./dapi/'+f),cv2.bitwise_not(np.uint8(blue)))
		nuc = ~(A==1)
		green = (np.array(green) > 120)[:1024,:1280]
		green = green * nuc
		ec = (A==3)
		chrom = (A==2)
		tot_fish.append(len(np.where(green)[0]))
		tot_ec.append(len(np.where(ec)[0]))
		tot_chrom.append(len(np.where(chrom)[0]))
		egfr_ec.append(len(np.where((green*ec))[0]))
		egfr_chrom.append(len(np.where((green*chrom))[0]))
		if(tot_fish[-1]==0):
			egfr_ec_ratio.append(0)
		else:
			egfr_ec_ratio.append(len(np.where((green*ec))[0])/tot_fish[-1])
		if(tot_fish[-1]==0):
			egfr_chrom_ratio.append(0)
		else:
			egfr_chrom_ratio.append(len(np.where((green*chrom))[0])/tot_fish[-1])

df = pd.DataFrame({'image_name':img_name, 'image_type': im_type, 'ec_pixels':tot_ec,
	'chrom_pixels':tot_chrom, 'fish_pixels(green)':tot_fish, 'ec+fish':egfr_ec, 'chrom+fish':egfr_chrom,
	'(ec+fish)/fish':egfr_ec_ratio, '(chrom+fish)/fish': egfr_chrom_ratio})
df.to_excel('hsr_ec.xlsx')
print('HSR: egfr+ec median',np.percentile(df[df['image_type']=='hsr'].iloc[:,-2], 50))
print('HSR: egfr+chrom median',np.percentile(df[df['image_type']=='hsr'].iloc[:,-1], 50))
print('DN: egfr+ec median',np.percentile(df[df['image_type']=='ec'].iloc[:,-2], 50))
print('DN: egfr+chrom median',np.percentile(df[df['image_type']=='ec'].iloc[:,-1], 50))