import numpy as np

def dist_between(vec, vec_arr):
    '''Computes Euclidean distance btwn
     vec and each row in vec_arr'''
    vec_repeat = np.tile(vec, (vec_arr.shape[0], 1))
    diff = vec_arr - vec_repeat
    dists = np.linalg.norm(diff, axis=1)
    return dists

def get_dmax(class_embeds):
    '''Computes maximum of distance
    from class centroid to each positive
    data point''' 
    centroid = np.mean(class_embeds, axis=0) # Compute class centroid
    ds_win_class = dist_between(centroid, class_embeds)
    dmax = np.max(ds_win_class).astype(float) # Get dmax
    return dmax

def dmax_error(class_embeds, neg_embeds, neg_ecs):
    '''Computes fraction of errors, negative samples
    contained within the hypersphere of radius dmax
    where dmax is the greatest distance between the class
    centroid and a positive sample''' 
    centroid = np.mean(class_embeds, axis=0) # Compute class centroid
    ds_win_class = dist_between(centroid, class_embeds)
    dmax = np.max(ds_win_class) # Get dmax
    ds_neg = dist_between(centroid, neg_embeds) # Distances from negative samples to class centroid
    less_dmax = ds_neg <= dmax
    fps = neg_embeds[np.array(less_dmax)] # "False positives" fall w/in hypersphere
    tns = neg_embeds[~np.array(less_dmax)] # "True negatives" fall outside of hypersphere
    fp_ecs = neg_ecs[np.array(less_dmax)] 
    tn_ecs = neg_ecs[~np.array(less_dmax)]
    error_frac = fp_ecs.shape[0] / neg_embeds.shape[0]
    # return error_frac, fps, tns, fp_ecs, tn_ecs
    return error_frac