import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, os.path
from roipoly import roipoly
import pylab as pl
import msvcrt as m

count = 1
folder = 'Training_set'
folder_out = 'TrainingData'
folder_colors = ['Red','Black','Brown','Yellow','Red_NoBarrel']

if not os.path.exists(folder_out):
	for color in folder_colors:
		os.makedirs(folder_out+'/'+color)
else:
	for color in folder_colors:
		subfolder = folder_out+'/'+color
		if not os.path.exists(subfolder):
			os.makedirs(subfolder)

for filename in os.listdir(folder):
	if filename not in ("2.2.png","2.6.png","2.8.png","5.4.png"):
		continue
	image = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
	if image is not None:
		img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		
		pl.imshow(img_rgb)
		pl.title('draw red ROI')
		
		redROI = roipoly(roicolor='g')
		mask = redROI.getMask(img_rgb)
		red_data = img_rgb[mask]
		np.save(folder_out + '/' + folder_colors[0] + '/' + filename, red_data)
		
		pl.imshow(img_rgb)
		pl.title('draw black ROI')

		blackROI = roipoly(roicolor='r')
		mask = blackROI.getMask(img_rgb)
		black_data = img_rgb[mask]
		np.save(folder_out + '/' + folder_colors[1] + '/' + filename, black_data)

		pl.imshow(img_rgb)
		pl.title('draw brown ROI')

		brownROI = roipoly(roicolor='g')
		mask = brownROI.getMask(img_rgb)
		brown_data = img_rgb[mask]
		np.save(folder_out + '/' + folder_colors[2] + '/' + filename, brown_data)

		pl.imshow(img_rgb)
		pl.title('draw yellow ROI')

		yellowROI = roipoly(roicolor='r')
		mask = yellowROI.getMask(img_rgb)
		yellow_data = img_rgb[mask]
		np.save(folder_out + '/' + folder_colors[3] + '/' + filename, yellow_data)

		pl.imshow(img_rgb)
		pl.title('draw red (No Barrel) ROI')

		nredROI = roipoly(roicolor='g')
		mask = nredROI.getMask(img_rgb)
		nred_data = img_rgb[mask]
		np.save(folder_out + '/' + folder_colors[4] + '/' + filename, nred_data)

		print(count)
		count = count + 1

