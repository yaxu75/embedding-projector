from __future__ import division
#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 12:46:25 2017

@author: ehsanamid
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:17:14 2017

@author: ehsanamid
"""

import sys
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD
import numpy as np
from annoy import AnnoyIndex

def generate_triplets(X, n_inlier, n_outlier, n_random, fast_trimap = True, weight_adj = False, verbose = True):
    n, dim = X.shape
    if dim > 100:
        X = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)
        dim = 100
    exact = n <= 10000
    n_extra = min(max(n_inlier, 200),n)
    if exact: # do exact knn search
        knn_tree = knn(n_neighbors= n_extra, algorithm='auto').fit(X)
        distances, nbrs = knn_tree.kneighbors(X)
    elif fast_trimap: # use annoy
        tree = AnnoyIndex(dim)
        for i in range(n):
            tree.add_item(i, X[i,:])
        tree.build(50)
        nbrs = np.empty((n,n_extra), dtype=np.int64)
        distances = np.empty((n,n_extra), dtype=np.float64)
        dij = np.empty(n_extra, dtype=np.float64)
        for i in range(n):
            nbrs[i,:] = tree.get_nns_by_item(i, n_extra)
            for j in range(n_extra):
                dij[j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
            sort_indices = np.argsort(dij)
            nbrs[i,:] = nbrs[i,sort_indices]
            # for j in range(n_extra):
            #     distances[i,j] = tree.get_distance(i, nbrs[i,j])
            distances[i,:] = dij[sort_indices]
    else:
        n_bf = 10
        n_extra += n_bf
        knn_tree = knn(n_neighbors= n_bf, algorithm='auto').fit(X)
        _, nbrs_bf = knn_tree.kneighbors(X)
        nbrs = np.empty((n,n_extra), dtype=np.int64)
        nbrs[:,:n_bf] = nbrs_bf
        tree = AnnoyIndex(dim)
        for i in range(n):
            tree.add_item(i, X[i,:])
        tree.build(60)
        distances = np.empty((n,n_extra), dtype=np.float64)
        dij = np.empty(n_extra, dtype=np.float64)
        for i in range(n):
            nbrs[i,n_bf:] = tree.get_nns_by_item(i, n_extra-n_bf)
            unique_nn = np.unique(nbrs[i,:])
            n_unique = len(unique_nn)
            nbrs[i,:n_unique] = unique_nn
            for j in range(n_unique):
                dij[j] = euclid_dist(X[i,:], X[nbrs[i,j],:])
            sort_indices = np.argsort(dij[:n_unique])
            nbrs[i,:n_unique] = nbrs[i,sort_indices]
            distances[i,:n_unique] = dij[sort_indices]
    if verbose:
        print("found nearest neighbors")
    sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
    P = find_p(distances, sig, nbrs)
    triplets = sample_knn_triplets(P, nbrs, n_inlier, n_outlier)
    n_triplets = triplets.shape[0]
    outlier_dist = np.empty(n_triplets, dtype=np.float64)
    if exact or  not fast_trimap:
        for t in range(n_triplets):
            outlier_dist[t] = np.sqrt(np.sum((X[triplets[t,0],:] - X[triplets[t,2],:])**2))
    else:
        for t in range(n_triplets):
            outlier_dist[t] = tree.get_distance(triplets[t,0], triplets[t,2])
    weights = find_weights(triplets, P, nbrs, outlier_dist, sig)
    if n_random > 0:
        rand_triplets = sample_random_triplets(X, n_random, sig)
        rand_weights = rand_triplets[:,-1]
        rand_triplets = rand_triplets[:,:-1].astype(np.int64)
        triplets = np.vstack((triplets, rand_triplets))
        weights = np.hstack((weights, rand_weights))
    weights /= np.max(weights)
    weights += 0.0001
    if weight_adj:
        weights = np.log(1 + 50 * weights)
        weights /= np.max(weights)
    return (triplets, weights)


def euclid_dist(x1, x2):
    """
    Fast Euclidean distance calculation between two vectors.
    """
    result = 0.0
    for i in range(x1.shape[0]):
        result += (x1[i]-x2[i])**2
    return np.sqrt(result)

def find_p(distances, sig, nbrs):
    """
    Calculates the similarity matrix P
    Input
    ------
    distances: Matrix of pairwise distances
    sig: Scaling factor for the distances
    nbrs: Nearest neighbors
    Output
    ------
    P: Pairwise similarity matrix
    """
    n, n_neighbors = distances.shape
    P = np.zeros((n,n_neighbors), dtype=np.float64)
    for i in range(n):
        for j in range(n_neighbors):
            P[i,j] = np.exp(-distances[i,j]**2/sig[i]/sig[nbrs[i,j]])
    return P

def sample_random_triplets(X, n_random, sig):
    """
    Sample uniformly random triplets
    Input
    ------
    X: Instance matrix
    n_random: Number of random triplets per point
    sig: Scaling factor for the distances
    Output
    ------
    rand_triplets: Sampled triplets
    """
    n = X.shape[0]
    rand_triplets = np.empty((n * n_random, 4), dtype=np.float64)
    for i in range(n):
        for j in range(n_random):
            sim = np.random.choice(n)
            while sim == i:
                sim = np.random.choice(n)
            out = np.random.choice(n)
            while out == i or out == sim:
                out = np.random.choice(n)
            p_sim = np.exp(-euclid_dist(X[i,:],X[sim,:])**2/(sig[i] * sig[sim]))
            if p_sim < 1e-20:
                p_sim = 1e-20
            p_out = np.exp(-euclid_dist(X[i,:],X[out,:])**2/(sig[i] * sig[out]))
            if p_out < 1e-20:
                p_out = 1e-20
            if p_sim < p_out:
                sim, out = out, sim
                p_sim, p_out = p_out, p_sim
            rand_triplets[i * n_random + j,0] = i
            rand_triplets[i * n_random + j,1] = sim
            rand_triplets[i * n_random + j,2] = out
            rand_triplets[i * n_random + j,3] = p_sim/p_out
    return rand_triplets

def sample_knn_triplets(P, nbrs, n_inlier, n_outlier):
    """
    Sample nearest neighbors triplets based on the similarity values given in P
    Input
    ------
    nbrs: Nearest neighbors indices for each point. The similarity values
        are given in matrix P. Row i corresponds to the i-th point.
    P: Matrix of pairwise similarities between each point and its neighbors
        given in matrix nbrs
    n_inlier: Number of inlier points
    n_outlier: Number of outlier points
    Output
    ------
    triplets: Sampled triplets
    """
    n, n_neighbors = nbrs.shape
    triplets = np.empty((n * n_inlier * n_outlier, 3), dtype=np.int64)
    for i in range(n):
        sort_indices = np.argsort(-P[i,:])
        for j in range(n_inlier):
            sim = nbrs[i,sort_indices[j+1]]
            samples = rejection_sample(n_outlier, n, sort_indices[:j+2])
            for k in range(n_outlier):
                index = i * n_inlier * n_outlier + j * n_outlier + k
                out = samples[k]
                triplets[index,0] = i
                triplets[index,1] = sim
                triplets[index,2] = out
    return triplets

def rejection_sample(n_samples, max_int, rejects):
    """
    Samples "n_samples" integers from a given interval [0,max_int] while
    rejecting the values that are in the "rejects".
    """
    result = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        reject_sample = True
        while reject_sample:
            j = np.random.randint(max_int)
            for k in range(i):
                if j == result[k]:
                    break
            for k in range(rejects.shape[0]):
            	if j == rejects[k]:
            		break
            else:
                reject_sample = False
        result[i] = j
    return result

def find_weights(triplets, P, nbrs, distances, sig):
    """
    Calculates the weights for the sampled nearest neighbors triplets
    Input
    ------
    triplets: Sampled triplets
    P: Pairwise similarity matrix
    nbrs: Nearest neighbors
    distances: Matrix of pairwise distances
    sig: Scaling factor for the distances
    Output
    ------
    weights: Weights for the triplets
    """
    n_triplets = triplets.shape[0]
    weights = np.empty(n_triplets, dtype=np.float64)
    for t in range(n_triplets):
        i = triplets[t,0]
        sim = 0
        while(nbrs[i,sim] != triplets[t,1]):
            sim += 1
        p_sim = P[i,sim]
        p_out = np.exp(-distances[t]**2/(sig[i] * sig[triplets[t,2]]))
        if p_out < 1e-20:
            p_out = 1e-20
        weights[t] = p_sim/p_out
    return weights




def trimap(X, num_dims=2, num_neighbs=50, num_out=10, num_rand=5, eta=2000.0, Yinit=[]):
    n, dim = X.shape
    print("running TriMap on %d points with dimension %d" % (n, dim))
    print("PLEASE DO NOT DISTRIBUTE THE CODE!")
    X -= np.min(X)
    X /= np.max(X)
    X -= np.mean(X, axis=0)
    if dim > 50:
        #        pca = PCA(n_components=50)
        #        pca.fit(X)
        #        X = np.dot(X, pca.components_.transpose())
        #        cov = np.dot(X.transpose(), X)
        #        pca = PCA(n_components=50)
        #        pca.fit(cov)
        #        X = np.dot(X, pca.components_.transpose())
        X = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
    if np.size(Yinit) > 0:
        Y = Yinit
    else:
        Y = np.random.normal(size=[n, num_dims]) * 0.0001
    C = np.inf
    best_C = np.inf
    best_Y = Y
    tol = 1e-7
    num_iter = 1000
    #    eta = 500.0 # learning rate

    triplets, weights = generate_triplets(X, num_neighbs, num_out, num_rand)


    return (triplets, weights)





