import cv2
import numpy as np
import os, os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gmm import *
from train_camera import *
from train_color import *

#####################################
# Default parameters / models
folder = 'Test_set'
model_file = 'model.p'
f = 0.55
Train_Camera = False
Train_Color = False
Save_Output = False
num_clusters = 3
#####################################

if Train_Color:
	print 'Training the colors'
	from train_color import *
	model_file = train(num_clusters)

if Train_Camera:
	print 'Training the camera model'
	f = get_camera_model(model_file, num_clusters)
	print f

if Save_Output:
	output_folder = 'Test_output'
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

model = pickle.load(open(model_file, "rb"))
print model

frame = 1

for filename in os.listdir(folder):

	image = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_COLOR)
	if image is not None:
		H, W, dim = np.shape(image)
		img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		ny, nx, dim = np.shape(img_rgb)
		data = np.reshape(img_rgb, (nx*ny, dim))

		p_red = prob_data_given_color(data, model['Red']['alpha'], model['Red']['mu'], model['Red']['cov'], num_clusters)
		p_black = prob_data_given_color(data, model['Black']['alpha'], model['Black']['mu'], model['Black']['cov'], num_clusters)
		p_brown = prob_data_given_color(data, model['Brown']['alpha'], model['Brown']['mu'], model['Brown']['cov'], num_clusters)
		p_yellow = prob_data_given_color(data, model['Yellow']['alpha'], model['Yellow']['mu'], model['Yellow']['cov'], num_clusters)
		p_rednb = prob_data_given_color(data, model['Red_NoBarrel']['alpha'], model['Red_NoBarrel']['mu'], model['Red_NoBarrel']['cov'], num_clusters)

		p = np.vstack((p_red, p_black, p_brown, p_yellow, p_rednb))
		color = np.argmax(p, axis=0)
		color[color==0] = 255
		color[color==1] = 0
		color[color==2] = 0
		color[color==3] = 0
		color[color==4] = 0
		img_data = np.reshape(color, (ny, nx))

		cv_img = img_data.astype(np.uint8)
		cv_img = cv2.medianBlur(cv_img, 5)
		ret, thresh = cv2.threshold(cv_img, 127, 255, 0)
		im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		max_index = np.argmax(areas)
		cnt = contours[max_index]
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		centroid = np.mean(box, axis=0)

		cv2.drawContours(image,[box],0,(0,0,255),2)

		cv2.imshow("Show",image)
		if Save_Output:
			cv2.imwrite(output_folder+'/'+filename, image)
		cv2.waitKey()
		cv2.destroyAllWindows()
		BottomLeft = box[0,:]
		TopLeft = box[1,:]
		TopRight = box[2,:]
		BottomRight = box[3,:]
		
		w = np.sqrt(np.sum((BottomRight-BottomLeft)*(TopRight-TopLeft)))
		h = np.sqrt(np.sum((TopLeft-BottomLeft)*(TopRight-BottomRight)))
		d = f/np.sqrt(((w/W)**2 + (h/H)**2)/2)
		print 'ImageNo = %d' %frame
		print 'BottomLeft = (%f,%f)' %(BottomLeft[0], BottomLeft[1])
		print 'BottomRight = (%f,%f)' %(BottomRight[0], BottomRight[1])
		print 'TopLeft = (%f,%f)' %(TopLeft[0], TopLeft[1])
		print 'TopRight = (%f,%f)' %(TopRight[0], TopRight[1])
		print 'Centroid = (%f,%f)' %(centroid[0], centroid[1])
		print 'Width = %f, Height = %f, Distance = %f' %(w, h ,d)

	frame = frame + 1