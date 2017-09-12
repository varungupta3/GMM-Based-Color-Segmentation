import cv2
import numpy as np
import os, os.path
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gmm import *

# from skimage import data, util
# from skimage.measure import label, regionprops

def get_camera_model(model_file, num_clusters):
	""" Function that returns the camera parameters estimated using least squares. Based on the GMM results on the training images.
	"""
	model = pickle.load(open(model_file, "rb"))
	print model

	folder = 'Training_set'
	frame = 1
	f_est = 0.
	num_barrels = 0

	for filename in os.listdir(folder):
		fn = filename.split('.')[0]
		d = np.array(fn.split('_'), dtype='|S4')
		d = d.astype(float)
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
			ret, thresh = cv2.threshold(cv_img,127,255,0)
			# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
			# thresh = cv2.dilate(thresh,kernel,iterations = 3) # dilate
			im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			areas = [cv2.contourArea(c) for c in contours]
			areas_copy = np.copy(areas)

			for i in range(0, len(d)):
				max_index = np.argmax(areas_copy)
				areas_copy[max_index] = 0
				cnt = contours[max_index]
				rect = cv2.minAreaRect(cnt)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				centroid = np.mean(box, axis=0)

				cv2.drawContours(image,[box],0,(0,0,255),2)
				
				cv2.imshow("Show",image)
				cv2.waitKey()
				cv2.destroyAllWindows()
				BottomLeft = box[0,:]
				TopLeft = box[1,:]
				TopRight = box[2,:]
				BottomRight = box[3,:]
				
				w = np.sqrt(np.sum((BottomRight-BottomLeft)*(TopRight-TopLeft)))
				h = np.sqrt(np.sum((TopLeft-BottomLeft)*(TopRight-BottomRight)))
				f_est = f_est + np.sqrt(((w/W)**2 + (h/H)**2)/2)*d[i]
				print 'ImageNo = %d' %frame
				print 'BottomLeft = (%f,%f)' %(BottomLeft[0], BottomLeft[1])
				print 'BottomRight = (%f,%f)' %(BottomRight[0], BottomRight[1])
				print 'TopLeft = (%f,%f)' %(TopLeft[0], TopLeft[1])
				print 'TopRight = (%f,%f)' %(TopRight[0], TopRight[1])
				print 'Centroid = (%f,%f)' %(centroid[0], centroid[1])
				print 'Width = %f, Height = %f, Distance = %f' %(w, h ,d[i])
				print 'Estimate of focal length = %f' %f_est
				num_barrels = num_barrels + 1
				
		frame = frame + 1
	f = f_est/num_barrels
	return f