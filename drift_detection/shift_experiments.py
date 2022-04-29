import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import select, extract
from sqlalchemy.sql.expression import and_
from sklearn.feature_selection import SelectKBest
import math
import random
import sys


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta_total=1.0, clip=True):
    """ 
    gaussian_noise_subset creates gaussian noise of 
    specificed parameters in input data

    x: covariate data
    noise_amt: standard deviation of gaussian noise
    normalization: normalization parameter to divide noise by (e.g. 255 for images)
    delta_total: fraction of data affected
    """
    x_df = pd.DataFrame(x)
    bin_cols = x_df.loc[:, (x_df.isin([0,1])).all()].columns.values
    c_cols = [x for x in x_df.columns if x not in bin_cols]
    indices = np.random.choice(x.shape[0], math.ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[np.ix_(indices,c_cols)]      
    if len(c_cols) == 1:
        noise = np.random.normal(0, noise_amt / normalization, x_mod.shape[0]) 
    else:
        noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], len(c_cols))) 
    if clip:
        x_mod = np.clip(x_mod + noise, 0., 1.)
    else:
        x_mod = x_mod + noise
    x[np.ix_(indices,c_cols)] = x_mod
    return x,indices

# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    """
    knockout shift creates class imbalance by removing 
    a fraction of samples from a class
    
    x: covariate data
    y: label data
    cl: class (e.g. 0,1,2,3, etc.)
    delta: fraction of samples removed
    
    """
    del_indices = np.where(y == cl)[0]
    until_index = math.ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)
    return x, y

def changepoint_shift(X_s, y_s, X_t,y_t,cl,n_shuffle=0.25,rank=False):
    """
    changepoint shift swaps features on a changepoint axis

    X_s: source data 
    y_s: source label
    X_t: target data
    y_t: target label
    cl: class (e.g. 0,1,2,3, etc.)
    n_shuffle: number of features to shuffle
    rank: should features should be ranked or not? 
    """
    n_feats = X_s.shape[1]
    n_shuffle_feats=int(n_shuffle*n_feats)
                
    ## Get importance values - should sub for model-specific
    selector=SelectKBest(k=n_feats)
    selection=selector.fit(X_s,y_s)
    ranked_x = sorted(zip(selection.scores_,selection.get_support(indices=True)),reverse=True)
    shuffle_list=list(range(0,n_feats))
                                        
    # Get shuffled features
    if rank:
        prev=shuffle_list[ranked_x[0][1]]
        for i in range(n_shuffle_feats):
            shuffle_list[ranked_x[i][1]]=shuffle_list[ranked_x[(i+1)%n_shuffle_feats][1]]
        shuffle_list[ranked_x[i][1]]=prev
    else:
        prev=shuffle_list[ranked_x[n_feats-n_shuffle_feats][1]]
        for i in range(n_feats-n_shuffle_feats, n_feats):
            shuffle_list[ranked_x[i][1]]=shuffle_list[ranked_x[(i+1)%n_feats][1]]
        shuffle_list[ranked_x[i][1]]=prev
        
    # Shuffle features
    for i in range(len(X_t)):
        if y_t[i]==cl:
            X_t[i,:]=X_t[i,shuffle_list]
    return(X_t, y_t)

def multiway_feat_association_shift(X_t, y_t, n_shuffle=0.25, keep_rows_constant=True, repermute_each_column=True):
    """
    multiway_feat_association_shift swaps individuals within features

    X_t: target data
    y_t: target label
    cl: class (e.g. 0,1,2,3, etc.)
    n_shuffle: number of individuals to shuffle
    keep_rows_constant: are the permutations the same across features?
    repermute_each_column: are the individuals selected for permutation the same across features?
    """
        
    n_inds = X_t.shape[0]
    n_shuffle_inds=int(n_shuffle*n_inds)
    shuffle_start=np.random.randint(n_inds-n_shuffle_inds)
    shuffle_end = shuffle_start+n_shuffle_inds 
    shuffle_list=np.random.permutation(range(shuffle_start,shuffle_end))
    for i in range(X_t.shape[1]):
        rng = np.random.default_rng(i)
        if repermute_each_column:
            rng.random(1)
            shuffle_start=np.random.randint(n_inds-n_shuffle_inds)
            shuffle_end = shuffle_start+n_shuffle_inds    
        if not keep_rows_constant:   
            rng.random(1)
            shuffle_list= np.random.permutation(range(shuffle_start,shuffle_end))
        indices = list(range(0,shuffle_start))+list(shuffle_list) + list(range(shuffle_end,n_inds))
        # Implement so that it changes only for a specific class
        X_t[:,i]=X_t[indices,i]
            
    return(X_t, y_t)    

def binary_shift(x, y, cl, delta, p_frac):
    """
    binary shift 
    
    x: covariate data
    y: label data
    cl: class (e.g. 0,1,2,3, etc.)
    delta: proportion of 1:0
    p_frac: fraction of features changed
    
    """
    p=math.ceil(x.shape[1]*p_frac)
    del_indices = np.where(y == cl)[0]
    for i in range(0,p):
        i = int(i)
        unique, counts = np.unique(x[del_indices, i],return_counts=True)
        if set(unique) == {0,1} or set(unique) == {0} or set(unique) == {1}:
            until_index = math.ceil(delta * len(del_indices))
            #until_index = math.ceil(delta * counts[1])
            if until_index % 2 != 0:
                until_index = until_index + 1
            x[del_indices[:until_index], i] = 1
            x[del_indices[until_index:], i] = 0
    return x, y

def binary_noise_subset(x, p, delta_total=1.0):
    """ 
    binary_noise_subset creates binary noise of 
    specificed parameters in input data

    x: covariate data
    p: proportion of 1s
    delta_total: fraction of data affected
    """
    x_df = pd.DataFrame(x)
    bin_cols = x_df.loc[:, (x_df.isin([0,1])).all()].columns.values
    indices = np.random.choice(x.shape[0], math.ceil(x.shape[0] * delta_total), replace=False)
    x_mod = x[indices,:][:,bin_cols]
    if x_mod.shape[1] == 1:
        noise = np.random.binomial(1, p, x_mod.shape[0])
    else:
        noise = np.random.binomial(1, p, (x_mod.shape[0], x_mod.shape[1]))
    x[np.ix_( indices, bin_cols )] = noise
    return x,indices

def apply_shift(X_s_orig, y_s_orig, X_te_orig, y_te_orig, shift):
    
    X_te_1 = None
    y_te_1 = None
    
    if shift == 'large_gn_shift_1.0':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=1.0, delta_total=1.0,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_gn_shift_1.0':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=1.0, delta_total=1.0,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_gn_shift_1.0':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=1.0, delta_total=1.0,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_gn_shift_0.5':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=1.0, delta_total=0.5,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_gn_shift_0.5':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=1.0, delta_total=0.5,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_gn_shift_0.5':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=1.0, delta_total=0.5,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_gn_shift_0.1':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 100.0, normalization=1.0, delta_total=0.1,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_gn_shift_0.1':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 10.0, normalization=1.0, delta_total=0.1,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_gn_shift_0.1':
        X_te_1, _ = gaussian_noise_subset(X_te_orig, 1.0, normalization=1.0, delta_total=0.1,clip=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'ko_shift_0.1':
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.1)
    elif shift == 'ko_shift_0.5':
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 0.5)
    elif shift == 'ko_shift_1.0':
        X_te_1, y_te_1 = knockout_shift(X_te_orig, y_te_orig, 0, 1.0)
    elif shift == 'cp_shift_0.75':
        X_te_1, y_te_1 = changepoint_shift(X_s_orig,y_s_orig, X_te_orig, y_te_orig, 0,n_shuffle=0.75,rank=True)
        y_te_1 = y_te_orig.copy()
    elif shift == 'cp_shift_0.25':
        X_te_1, y_te_1 = changepoint_shift(X_s_orig,y_s_orig, X_te_orig, y_te_orig, 0,n_shuffle=0.25,rank=True)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.75_krc_rec':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=1,keep_rows_constant=True, repermute_each_column=True)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.5_krc_rec':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.5, keep_rows_constant=True, repermute_each_column=True)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.25_krc_rec':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.1, keep_rows_constant=True, repermute_each_column=True)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.75_krc':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=1.0,keep_rows_constant=True, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.5_krc':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.5, keep_rows_constant=True, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.25_krc':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.1, keep_rows_constant=True, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.75':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=1.0,keep_rows_constant=False, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.5':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.5, keep_rows_constant=False, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'mfa_shift_0.25':
        X_te_1, y_te_1 = multiway_feat_association_shift(X_te_orig, y_te_orig, n_shuffle=0.1, keep_rows_constant=False, repermute_each_column=False)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_bn_shift_1.0':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_bn_shift_1.0':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_bn_shift_1.0':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 1.0)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_bn_shift_0.5':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_bn_shift_0.5':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_bn_shift_0.5':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 0.5)
        y_te_1 = y_te_orig.copy()
    elif shift == 'large_bn_shift_0.1':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.5, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == 'medium_bn_shift_0.1':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.1, 0.1)
        y_te_1 = y_te_orig.copy()
    elif shift == 'small_bn_shift_0.1':
        X_te_1, _ = binary_noise_subset(X_te_orig, 0.01, 0.1)
        y_te_1 = y_te_orig.copy()
    return (X_te_1, y_te_1)
