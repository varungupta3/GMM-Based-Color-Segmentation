import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, os.path
from mpl_toolkits.mplot3d import Axes3D
from gmm import *
import cPickle as pickle

def get_unique(data):
	"""
	Utility function to remove repeating data
	"""
	# Perform lex sort and get sorted data
	sorted_idx = np.lexsort(data.T)
	sorted_data =  data[sorted_idx,:]
	# Get unique row mask
	row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
	# Get unique rows
	out = sorted_data[row_mask]
	return out

def train(num_clusters):
	"""
	Function to construct a GMM model for 5 colors using 'num_clusters' number of clusters
	"""
	folder_in = 'TrainingData/Set1'
	folder_colors = ['Red','Black','Brown','Yellow','Red_NoBarrel']

	if not os.path.exists(folder_in):
		exit()	

	dict = {}
	for subfolder in folder_colors:
		folder = folder_in + '/' + subfolder + '/';
		for filename in os.listdir(folder):
			try:
				temp
			except NameError:
				temp = np.load(folder+filename)
			else:
				temp = np.append(temp, np.load(folder+filename), axis=0)
		dict[subfolder] = temp
		del temp

	red_data = dict.get('Red')
	black_data = dict.get('Black')
	brown_data = dict.get('Brown')
	yellow_data = dict.get('Yellow')
	rednb_data = dict.get('Red_NoBarrel')

	red_unique = get_unique(red_data).astype(float)
	black_unique = get_unique(black_data).astype(float)
	brown_unique = get_unique(brown_data).astype(float)
	yellow_unique = get_unique(yellow_data).astype(float)
	rednb_unique = get_unique(rednb_data).astype(float)

	idx = np.random.randint(np.shape(red_unique)[0], size=10000)
	train_red = red_unique[idx,:]
	idx = np.random.randint(np.shape(black_unique)[0], size=2500)
	train_black = black_unique[idx,:]
	idx = np.random.randint(np.shape(brown_unique)[0], size=2500)
	train_brown = brown_unique[idx,:]
	idx = np.random.randint(np.shape(yellow_unique)[0], size=2500)
	train_yellow = yellow_unique[idx,:]
	idx = np.random.randint(np.shape(rednb_unique)[0], size=2500)
	train_rednb = rednb_unique[idx,:]

	# train_red = red_unique
	# train_black = black_unique
	# train_brown = brown_unique
	# train_yellow = yellow_unique
	# train_rednb = rednb_unique

	model = {'Red':{}, 'Black':{}, 'Brown':{}, 'Yellow':{}, 'Red_NoBarrel':{}}

	model['Red']['alpha'], model['Red']['mu'], model['Red']['cov'] = build_gmm(train_red, num_clusters)
	print 'Red Colour Trained'

	model['Black']['alpha'], model['Black']['mu'], model['Black']['cov'] = build_gmm(train_black, num_clusters)
	print 'Black Colour Trained'

	model['Brown']['alpha'], model['Brown']['mu'], model['Brown']['cov'] = build_gmm(train_brown, num_clusters)
	print 'Brown Colour Trained'

	model['Yellow']['alpha'], model['Yellow']['mu'], model['Yellow']['cov'] = build_gmm(train_yellow, num_clusters)
	print 'Yellow Colour Trained'

	model['Red_NoBarrel']['alpha'], model['Red_NoBarrel']['mu'], model['Red_NoBarrel']['cov'] = build_gmm(train_rednb, num_clusters)
	print 'Red (Not Barrel) Colour Trained'

	model_file = 'model_new.p'
	pickle.dump(model, open(model_file, "wb"))

	return model_file

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(red_unique[:,0], red_unique[:,1], red_unique[:,2], zdir='z', c='r')
	# ax.scatter(black_unique[:,0], black_unique[:,1], black_unique[:,2], zdir = 'z', c='k')
	# ax.scatter(brown_unique[:,0], brown_unique[:,1], brown_unique[:,2], zdir = 'z', c='brown')
	# ax.scatter(yellow_unique[:,0], yellow_unique[:,1], yellow_unique[:,2], zdir='z', c='yellow')
	# ax.scatter(rednb_unique[:,0], rednb_unique[:,1], rednb_unique[:,2], zdir='z', c='pink')
	# plt.show()
