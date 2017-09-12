import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_likelihood(x, mu, cov):
	""" Evaluates the likelihood of the data given a gaussian distribution with mean mu (3 by 1) and covariance matrix cov (3 by 3) , for a set of n datapoints. Returns an n by 1	
	"""
	dim = np.shape(cov)[1]
	coeff = 1./(np.sqrt(((2.*np.pi)**dim)*np.linalg.det(cov)))
	phi = coeff*np.exp(-np.matmul(np.matmul((x-mu),np.linalg.inv(cov)), x-mu)/2)
	return phi 

def eStep(alpha, mu, cov, x, k):
	""" Expectation step to compute the membership probabilities
	"""
	len_data = np.shape(x)[0]
	phi = np.zeros((k,))
	r = np.zeros((len_data, k))
	for j in range(0, len_data):
		for i in range(0,k): 
			phi[i] = get_likelihood(x[j,:], mu[i,:], cov[i,:,:]) 
		l = phi * alpha
		r[j,:] = l/np.sum(l) 
	return r


def mStep(r, x, k):
	""" Maximization step to compute the alpha's, means and covariances 
	"""
	len_data = np.shape(x)[0] 
	num_channels = np.shape(x)[1]
	mu = np.zeros((k, num_channels))
	cov = np.zeros((k, num_channels, num_channels))
	alpha = np.zeros((k,))
	for i in range(0,k): 
		sum_r = np.sum(r[:,i])
		mu[i,:] = np.dot(r[:,i], x)/sum_r
		r_mat = np.repeat(r[:,i], num_channels).reshape(len_data, num_channels)
		cov[i,:,:] = np.matmul(np.transpose(r_mat*(x-mu[i,:])),(x-mu[i,:]))/sum_r
		alpha[i] = sum_r/len_data
		
	return mu, cov, alpha


def build_gmm(data, num_clusters):
	""" Main function that builds the GMM model by calling eStep and mStep alternately until convergence of cost function
	"""

	alpha = np.linspace(1.0/num_clusters, 1.0/num_clusters, num_clusters)
	num_channels = np.shape(data)[1]

	len_data = np.shape(data)[0]
	kmeans = KMeans(n_clusters = num_clusters).fit(data)
	
	mu = kmeans.cluster_centers_ 
	cov = np.zeros((num_clusters, num_channels, num_channels))
	pred = kmeans.predict(data)
	for y in range(0,num_clusters):
		idx = np.where(pred==y)[0]
		num_elements = np.shape(idx)[0]
		cov[y,:,:] = np.matmul(np.transpose((data[idx,:]-mu[y,:])),(data[idx,:]-mu[y,:]))/num_elements

	prev_cost_fn = 0
	phi = np.zeros((len_data, num_clusters))
	for i in range(0, num_clusters):
		for j in range(0, len_data):
			phi[j,i] = get_likelihood(data[j,:], mu[i,:], cov[i,:,:])
	alpha_mat = np.repeat(alpha, len_data).reshape(num_clusters, len_data).T
	p = np.sum(phi*alpha_mat, axis=1)
	cost_fn = np.sum(np.log(p))
	n_iter = 1
	obj_fn = np.array([cost_fn])
	while np.fabs(cost_fn-prev_cost_fn) > 10**-2:
		r = eStep(alpha, mu, cov, data, num_clusters)
		mu, cov, alpha = mStep(r, data, num_clusters)
		alpha_mat = np.repeat(alpha, len_data).reshape(num_clusters, len_data).T
		for i in range(0, num_clusters):
			for j in range(0, len_data):
				phi[j,i] = get_likelihood(data[j,:], mu[i,:], cov[i,:,:])

		p = np.sum(phi*alpha_mat, axis=1)
		prev_cost_fn = cost_fn
		cost_fn = np.sum(np.log(p))
		print n_iter, cost_fn, prev_cost_fn
		n_iter = n_iter + 1
		obj_fn = np.append(obj_fn, cost_fn)
	
	plt.plot(obj_fn)
	plt.show()
	return alpha, mu, cov


def prob_data_given_color(x, alpha, mu, cov, k):
	""" Evaluates the GMM model and returns the probabilities of each class given the data
	"""
	len_data = np.shape(x)[0]
	num_channels = np.shape(x)[1]
	alpha_mat = np.repeat(alpha, len_data).reshape(k, len_data).T
	likelihood = np.zeros((len_data, k))
	dim = np.shape(cov)[1]
	for i in range(0, k):
		coeff = 1./(np.sqrt(((2.*np.pi)**dim)*np.linalg.det(cov[i,:,:])))
		chol = np.linalg.cholesky(np.linalg.inv(cov[i,:,:]))
		g = np.matmul(x-mu[i,:], chol)
		likelihood[:,i] = coeff*np.exp(-0.5*np.sum(g*g, axis=1))
	p = np.sum(likelihood*alpha_mat, axis=1)
	return p
