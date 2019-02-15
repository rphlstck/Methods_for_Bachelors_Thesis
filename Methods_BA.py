# header
# calculating
import numpy as np

# plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, scale
from sklearn.preprocessing import Imputer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# fitting 
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData # ODR is orthogonal distance regression -> allows to fit both in x and y direction

# pandas data visualization
import pandas as pd

from IPython.display import display

# extra container types
import collections as co

import sys
from copy import deepcopy
#print(sys.version)
#input()

# the indices and the corresponding feature of the legus data 
dir_legus_featurename_and_id = {0:"source id", 1:"x corr", 2:"y corr", 3:"RA", 4:"DEC", 5:"final total mag in WFC3/F275W", 6:"final photometric error in WFC3/F275W", 7:"final total mag in WFC3/F336W", 8:"final photometric error in WFC3/F336W", 9:"final total mag in ASC/F435W", 10:"final photometric error in ASC/F435W", 11:"final total mag in ASC/F555W", 12:"final photometric error in ASC/F555W", 13:"final total mag in ASC/F814W", 14:"final photometric error in ASC/F814W", 15:"CI=mag(1px)-mag(3px)", 16:"best age in yr", 17:"max age in yr (within 68% confidence level)", 18:"min age in yr (within 68% confidence level)", 19:"best mass in solar masses", 20:"max mass in solar masses (within 68% confidence level)", 21:"min mass in solar masses (within 68% confidence level)", 22:"best E(B-V)", 23:"max E(B-V) (within 68% confidence level)", 24:"min E(B-V) (within 68% confidence level)", 25:"chi2 fit residual in F275W", 26:"chi2 fit residual in F336W", 27:"chi2 fit residual in F435W", 28:"chi2 fit residual in F555W", 29:"chi2 fit residual in F814W", 30:"reduced chi2", 31:"Q probability", 32:"Number of filter", 33:"Final assigned class", 34:"Final assigned class after visual inspection"}
# the indices and the corresponding feature of the sitelle data 
dir_sitelle_featurename_and_id= {0:"ID of Region", 1:"RA", 2:"DEC", 3:"Galactocentric radius", 4:"Ha total luminosity corrected for extinction", 5:"Ha mean diffuse ionized gaz backgound level", 6:"Region category", 7:"I0", 8:"Amp", 9:"sig", 10:"alpha", 11:"R2", 12:"size; pc size of the regions", 13:"EBV; extinction; E(B-V)", 14:"EBV_err; extinction error; E(B-V error)", 15:"log [NII]6583/Ha; log line ratio", 16:"error on log [NII]6583/Ha; log line ratio error", 17:"SNR_cross [NII]6583/Ha; line ratio best SNR", 18:"log [SII]6716+6731/Ha; log line ratio", 19:"error on log [SII]6716+6731/Ha; log line ratio error", 20:"SNR_cross [SII]6716+6731/Ha; line ratio best SNR", 21:"log [SII]6716+6731/[NII]6583; log line ratio", 22:"error on log [SII]6716+6731/[NII]6583; log line ratio error", 23:"SNR_cross on [SII]6716+6731/[NII]6583; line ratio best SNR", 24:"log [OIII]5007/Hb ; log line ratio", 25:"error on log [OIII]5007/Hb ; log line ratio error", 26:"SNR_cross on [OIII]5007/Hb; line ratio best SNR", 27:"log [OII]3727/Hb ; log line ratio", 28:"error on log [OII]3727/Hb ; log line ratio error", 29:"SNR_cross on [OII]3727/Hb; line ratio best SNR", 30:"log ([OII]3727+[OIII]5007)/Hb ; log line ratio", 31:"error on log ([OII]3727+[OIII]5007)/Hb ; log line ratio error", 32:"SNR_cross on ([OII]3727+[OIII]5007)/Hb; line ratio best SNR", 33:"log [OIII]5007/[OII]3727 ; log line ratio", 34:"error on log [OIII]5007/[OII]3727 ; log line ratio error", 35:"SNR_cross on [OIII]5007/[OII]3727; line ratio best SNR", 36:"log [OIII]5007/[NII]6583 ; log line ratio", 37:"error on log [OIII]5007/[NII]6583 ; log line ratio error", 38:"SNR_cross on [OIII]5007/[NII]6583; line ratio best SNR", 39:"log [OII]3727/[NII]6583 ; log line ratio", 40:"error on log [OII]3727/[NII}6583 ; log line ratio error", 41:"SNR_cross on [OII]3727/[NII}6583; line ratio best SNR", 42:"[SII]6716/[SII]6731; line ratio", 43:"error on [SII]6716/[SII]6731; line ratio error", 44:"SNR_cross on [SII]6716/[SII]6731; line ratio best SNR"}

def euclidean_distance(a, b):
    '''
    returns the euclidean distance between vector a and vector b
    
    a, b vector or number
    '''
    return np.sqrt(a**2+b**2)

def transform_pc_in_ra_dec(size, dist=9.9e6):
    '''
    returns the angle between the two objects seen from the observer in degrees
    
    size is the radius of the object in pc
    dist is the distance that the observed object is located in pc
    '''
    gamma = 2*np.arcsin(size/(2*dist))
    gamma = np.rad2deg(gamma)
    return gamma
    
def calculate_distance(angle, dist=9.9e6):
    '''
    returns the distance between two objects at same distance in pc(dist) given the angle in degree (angle)
    '''
    # source of formula: https://de.wikipedia.org/wiki/Gleichschenkliges_Dreieck
    # hope np.sin is stable for small angles
    angle = np.deg2rad(angle)
    return 2*dist*np.sin(angle/2)
    
def calculate_angle(RA1, DEC1, RA2, DEC2):
    '''
    returns the angle between two objects in deg
    
    all angles should be given in decimal degrees
    RA1 right ascension of object 1
    DEC1 declination of object 1
    RA2 right ascension of object 2
    DEC2 declination of object 2
    '''
    # the decimal degrees should be transformed to radian so that np.sin/np.cos operates correctly
    RA1 = np.deg2rad(RA1)
    RA2 = np.deg2rad(RA2)
    DEC1 = np.deg2rad(DEC1)
    DEC2 = np.deg2rad(DEC2)
    # source of formula: http://www.gyes.eu/calculator/calculator_page1.htm
    res = np.arccos(np.sin(DEC1)*np.sin(DEC2)+np.cos(DEC1)*np.cos(DEC2)*np.cos(RA1-RA2))
    return np.rad2deg(res)

def calculate_r_squared(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x-x_mean)*(y-y_mean))
    denumerator = np.sqrt(np.sum((x-x_mean)**2)*(np.sum((y-y_mean)**2)))
    r = numerator/denumerator
    return r**2
    
def calculate_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x-x_mean)*(y-y_mean))
    denumerator = np.sqrt(np.sum((x-x_mean)**2)*(np.sum((y-y_mean)**2)))
    r = numerator/denumerator
    return r

def matches_against_r2(r_asc_le_ov, decl_le_ov, r_asc_le_ov_dc, decl_le_ov_dc, thresholds, filters, overlap_legus_c, overlap_legus_dc):
    # iterate over a thresholdrange
    num_of_matches = []
    summed_mean_r2 = []
    for threshold in thresholds:
        a = Nearest_Neighbor([-r_asc_le_ov, decl_le_ov], [-r_asc_le_ov_dc, decl_le_ov_dc], double_use=False, axis=1, threshhold=threshold)
        num_of_matches.append(len(a))
        #filters = [5,7,9,11,13]#[22]# extinction#
        r_arr = []
        # calculated for every filter the correlation r^2
        a = np.asarray(a)
        for f in filters:
            mag_c = overlap_legus_c[a[:,0]][:,f]
            mag_dc = overlap_legus_dc[a[:,1]][:,f]
            mask1 = np.asarray(mag_c < 30) #(mag_c > 0)#
            mask2 = np.asarray(mag_dc < 30)#(mag_c > 0)#
            mask = np.asarray(mask1.astype(int)+mask2.astype(int) > 1)
            mag_c = mag_c[mask]
            mag_dc = mag_dc[mask]
            r_arr.append(calculate_correlation(mag_c, mag_dc))
        # average the obtained values of r^2
        summed_mean_r2.append(np.mean(r_arr))      
    return num_of_matches, summed_mean_r2

def create_distance_matrix_in_pc(A, B, dist):
    '''
    returns a distance matrix of the elements in A and B
    
         A[0] A[1] ...
           --------------------> A
    B[0]  |
    B[1]  |
      .   |
      .   |
      .   |
          v
          B
    
    A,B describe arrays with information like the position i.e [right ascesion, declination]
    dist describes the distance of the objects
    '''
    RA1 = A[0]
    RA2 = B[0]
    DEC1 = A[1]
    DEC2 = B[1]
    distance_matrix = np.array([calculate_distance(calculate_angle(RA1, DEC1, RA2[i], DEC2[i]),dist) for i in range(len(B[0]))])
    return distance_matrix

def create_distance_matrix(A, B):
    '''
    returns a distance matrix of the elements in A and B
    
         A[0] A[1] ...
           --------------------> A
    B[0]  |
    B[1]  |
      .   |
      .   |
      .   |
          v
          B
    
    A,B describe arrays with information like the position i.e [right ascesion, declination]
    '''
    distance_matrix = np.array([euclidean_distance(A[0]-B[0][i],A[1]-B[1][i]) for i in range(len(B[0]))])
    
    # tests for correctness
    assert(np.shape(distance_matrix)[1] == len(A[0]))
    assert(np.shape(distance_matrix)[0] == len(B[0]))
    assert(np.shape(distance_matrix)[1] == len(A[1]))
    assert(np.shape(distance_matrix)[0] == len(B[1]))
    assert(distance_matrix.all() >= 0)
    
    return distance_matrix

def create_distance_matrix_2(r_asc_1 , decl_1, r_asc_2, decl_2):
    '''
    returns a distance matrix of the elements of x_1 (r_asc_1), y_1 (decl_1) and x_2 (r_asc_2), y_2 (decl_2) 
    
         A[0] A[1] ...
           --------------------> A
    B[0]  |
    B[1]  |
      .   |
      .   |
      .   |
          v
          B
    
    A,B describe arrays with information like the position i.e [right ascesion, declination]
    '''
    A = np.array([r_asc_1,decl_1])
    B = np.array([r_asc_2,decl_2])
    distance_matrix = np.array([euclidean_distance(A[0]-B[0][i],A[1]-B[1][i]) for i in range(len(B[0]))])
    
    # tests for correctness
    assert(np.shape(distance_matrix)[1] == len(A[0]))
    assert(np.shape(distance_matrix)[0] == len(B[0]))
    assert(np.shape(distance_matrix)[1] == len(A[1]))
    assert(np.shape(distance_matrix)[0] == len(B[1]))
    assert(distance_matrix.all() >= 0)
    
    return distance_matrix

def Nearest_Neighbor(A, B, threshhold, double_use=True, axis = 0, actual_dist=None):
    '''
    returns the indices (not the ID of the objects!) of the points which are nearest to each other in a [(A_idx1, B_idx1),(A_idx2, B_idx2),...]
    
    A describes an array with information like the position i.e [right ascesion, declination]
    B describes an array which will be compared with A in order to find the nearest neighbor corresponding to A
    if double_use is True all neighbors within threshhold will be saved in a
    if double_use is False only the nearest neighbor will be saved in a
    axis = 0 : the same object of B can be the closest neighbor of different instances in A)
    axis = 1 : the same object of A can be the closest neighbor of different instances in B)
    usually order data_sitelle = A and data_legus = B
    '''
    # calculate the distance matrix
    if actual_dist!=None:
        distance_matrix = create_distance_matrix_in_pc(A,B,actual_dist)
    if actual_dist == None:
        distance_matrix = create_distance_matrix(A,B)
    
    a = []
    # all neighbors within the given threshhold will be saved in a
    if double_use == True:
        for i in range(len(A[0])):
            for j in range(len(B[0])):
                if distance_matrix[j][i] <= threshhold:
                    a.append((i,j)) # i corresponds to sitelle and j to legus if A=sitele and j=legus
    
    # new approach with np.argmin along axis
    # only the closest neighbor of A will be saved in a 
    
    if double_use == False:
        # (care! the same object of B can be the closest neighbor of different instances in A)
        if axis == 0:
            minima = np.argmin(distance_matrix, axis=axis) # contains indx of minimum distance of every legus-coordinate to a sitelle
            for j in range(len(minima)): # j is indx corresponding to a column of the distance matrix 
                d = distance_matrix[minima[j]][j] # get value of minimum along the column-axis
                if (d <= threshhold): # check if smaller than threshhold
                    a.append((j,minima[j])) # save in a ((sitelle, legus),...)
        
        # (care! the same object of A can be the closest neighbor of different instances in B)
        if axis == 1:
            minima = np.argmin(distance_matrix, axis=axis) # contains indx of minimum distance of every sitelle-coordinate to a legus
            for i in range(len(minima)): # i is indx corresponding to a row of the distance matrix
                d = distance_matrix[i][minima[i]] # get value of minimum along the column-axis
                if (d <= threshhold): # check if smaller than threshhold
                    a.append((minima[i],i)) # save in a ((sitelle, legus),...)
           
    return a

def feature_selection(data, indx_of_features):
    '''
    returns ...
    
    data is the array that should be pruned to the features of interest
    indx_of_features are the indicies of the features of interes
    
    for the sitelle and legus data the features are saved in the arrays like [[feat.1, feat.2], [feat.1, feat.2], ...]
    '''
    data = np.asarray(data)
    return data[:,indx_of_features]

def select_pairs(data, a, data_order):
    '''
    returns ...
    
    data is the array of features
    a is the array with indicies of the pairs found
    data_order int of the 1st or 2nd catalog that should be selected (usually when a is build sitelle has indx 0)
    '''
    a = np.asarray(a)
    data = np.asarray(data)
    return data[a[:,data_order]]

def count_common_matches(a,b):
    '''
    returns the number of equal elements in array a and array b
    a,b array (i.e like created by Nearest_Neighbor)
    '''
    count = 0
    for elt_a in a:
        for elt_b in b:
            if ((abs(elt_a[0] - elt_b[0]) == 0) and (abs(elt_a[1] - elt_b[1]) == 0)):
                count +=1
    return count

def clusters_embedded_in_HII_region(SITELLE, LEGUS, save_positional_index=False):
    '''
    returns an dictionary. The key corresponds to the ID (the position in the array SITELLE) of the HII-region -1 (zero based indexing here), the values are (true) ID (as saved in the first column in the LEGUS data) of the clusters that are embedded in that HII-region
    so in order to get the positional index in the dictionary set the save_positional_index True
    
    SITELLE is the SITELLE data set
    LEGUS  is the LEGUS data set
    '''
    
    r_asc_st = np.asarray(SITELLE)[:,1] # right ascesion of SITELLE data 
    decl_st = np.asarray(SITELLE)[:,2] # declination of SITELLE data
    size_st = np.asarray(SITELLE)[:,12] # this is given in pc
    size_st_deg = transform_pc_in_ra_dec(size_st) # transform the distance from pc into deg
    
    r_asc_le = np.asarray(LEGUS)[:,3] # right ascesion of LEGUS data
    decl_le = np.asarray(LEGUS)[:,4] # declination of LEGUS data
    
    features_sitele = np.array([r_asc_st,decl_st])
    features_legus = np.array([r_asc_le,decl_le])
    
    # creates distance matrix where features of sitelle are the number of outer arrays and the number of instances in the inner array corresponds to the legus data
    distance_matrix = create_distance_matrix(features_sitele, features_legus)
    
    dic = {}
    dm = np.asarray(distance_matrix.T)
    
    # as values in the dictionary the positional index will be saved of the legus data
    assert(len(dm[0]) == len(LEGUS))
    if save_positional_index == True:
        for i in range(len(dm)):
            dic[i] = np.asarray(np.where(dm[i] <= size_st_deg[i])).flatten()
        return dic
    
    # as values in the dictionary the ID of the legus data will be saved
    else:
        for i in range(len(dm)):
            dic[i] = LEGUS[:,0][dm[i] < size_st_deg[i]]
        return dic
        
# this is an older version that got a little bit different functionality
# some data returned in data_legus can occur multiple times this is in the new version fixed       

#def reduce_dic_to_n_embedded(num_of_clusters_in_HII_region, dic, data_sitele, data_legus, save_positional_index=False):
#    '''
#    returns the pruned data according to dic so that only the data remains which have num_of_clusters_in_HII_region clusters in one HII-Region
#    
#    num_of_clusters_in_HII_region number of clusters in one HII-Region that will only be considered 
#    data_sitele is the same data as passed in clusters_embedded_in_HII_region
#    data_legus is the same data as passed in clusters_embedded_in_HII_region
#    only use save_positional_index = False if whole dataset is used
#    '''
#    # create mask that only 
#    mask_for_corr_legus_data = np.array([v for v in dic.values()])[[(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]]
#    mask_for_corr_legus_data = np.asarray([mask_for_corr_legus_data[i][j] for i in range(len(mask_for_corr_legus_data)) for j in range(len(mask_for_corr_legus_data[i]))]).astype(int)
#    if save_positional_index == False:
#        data_legus = [data_legus[i-1] for i in mask_for_corr_legus_data]
#    else:
#        data_legus = [data_legus[i] for i in mask_for_corr_legus_data]
#    data_legus = np.asarray(data_legus)
#    data_sitele = np.asarray(data_sitele)[[(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]]
#    #print('len sitelle data', len(data_sitele))
#    #if num_of_clusters_in_HII_region != 0:
#    #    print('len unique data', len(np.unique(data_legus[:,0])))
#    assert(sum([(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]) == len(data_sitele))
#    assert(len(mask_for_corr_legus_data) == sum([(len(v) == num_of_clusters_in_HII_region) for v in dic.values()])*num_of_clusters_in_HII_region)
#    assert(len(data_legus) == len(data_sitele)*num_of_clusters_in_HII_region)
#    return data_sitele, data_legus

def reduce_dic_to_n_embedded(num_of_clusters_in_HII_region, dic, data_sitele, data_legus, save_positional_index=False):
    '''
    returns the pruned data according to dic so that only the data remains which have num_of_clusters_in_HII_region clusters in one HII-Region
    
    num_of_clusters_in_HII_region number of clusters in one HII-Region that will only be considered 
    data_sitele is the same data as passed in clusters_embedded_in_HII_region
    data_legus is the same data as passed in clusters_embedded_in_HII_region
    only use save_positional_index = False if whole dataset is used
    '''
    # create mask that only
    mask_for_corr_legus_data = np.array([v for v in dic.values()])[[(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]]
    mask_for_corr_legus_data = np.asarray([mask_for_corr_legus_data[i][j] for i in range(len(mask_for_corr_legus_data)) for j in range(len(mask_for_corr_legus_data[i]))]).astype(int)
    if save_positional_index == False:
        data_legus = [data_legus[i-1] for i in mask_for_corr_legus_data]
    else:
        data_legus = [data_legus[i] for i in mask_for_corr_legus_data]
    
    data_legus_temp = []
    already_used = []
    for i in range(len(data_legus)):
        ID = data_legus[i][0]
        if ID not in already_used:
            data_legus_temp.append(data_legus[i])
            already_used.append(ID)
    data_legus = np.asarray(data_legus_temp)
    data_sitele = np.asarray(data_sitele)[[(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]]
    
    # some quick tests for correctness
    # check that len sitelle data is equal to clusters with num_of_clusters_in_HII_region
    assert(sum([(len(v) == num_of_clusters_in_HII_region) for v in dic.values()]) == len(data_sitele))
    assert(len(mask_for_corr_legus_data) == sum([(len(v) == num_of_clusters_in_HII_region) for v in dic.values()])*num_of_clusters_in_HII_region)
    # avoid that same data is represented more than once in array
    if len(data_legus) > 0:
        u = np.unique(data_legus[:,0])
        try:
            assert(len(np.unique(data_legus[:,0]))==len(data_legus))
        except:
            print('length of Legus data does not correspond to unique data - check if the ID of the combined datasets of legus data is ongoing?')
            
    return data_sitele, data_legus

def prune_to_relevant_data_using_NN(data_sitele, data_legus, num_of_mag_error_less_than_0p3, max_age, classes_sitelle, classes_legus, sep = None,class_idx_for_legus_classes=33, axis=1, min_age=0):
    '''
    returns the pruned data sets of data_sitele and data_legus as well as the matching array a
    '''
    classes = classes_legus
    classes_st = classes_sitelle
    r_asc_st = np.asarray(data_sitele)[:,1] # right ascesion of SITELLE data 
    decl_st =  np.asarray(data_sitele)[:,2] # declination of SITELLE data
    if sep == None:
        sep = np.max(data_sitele[:,12])

    r_asc_le = np.asarray(data_legus)[:,3] # right ascesion of LEGUS data
    decl_le =  np.asarray(data_legus)[:,4] # declination of LEGUS data

    # get data which error is less than 0.3 mag in at least num_of_mag_error_less_than_0p3 bands
    bool1 = np.array(data_legus[:,6] < 0.3, dtype=np.int32)
    bool2 = np.array(data_legus[:,8] < 0.3, dtype=np.int32)
    bool3 = np.array(data_legus[:,10] < 0.3, dtype=np.int32)
    bool4 = np.array(data_legus[:,12] < 0.3, dtype=np.int32)
    bool5 = np.array(data_legus[:,14] < 0.3, dtype=np.int32)
    
    sum_bool = bool1 + bool2 + bool3 + bool4 + bool5

    mask_leg = sum_bool >= num_of_mag_error_less_than_0p3 # previously 4 and in legus paper have a look at end of page 6

    # prune for max_age
    bool6 = np.array(data_legus[:,16] <= max_age, dtype=np.int32)
    # prune for min_age
    bool7 = np.array(data_legus[:,16] >= min_age, dtype=np.int32)
    sum_bool2 = mask_leg.astype(np.int32) + bool6 + bool7
    mask_leg = sum_bool2 >=3 #2

    r_asc_le_red = np.asarray(r_asc_le)[mask_leg]
    decl_le_red = np.asarray(decl_le)[mask_leg]
    data_legus_red = data_legus[mask_leg]

    # for legus two different approaches were used when classifing the clusters into the 5 categories 
    # for the mode_class idx 33 for the mean_class idx 34
    class_idx = class_idx_for_legus_classes
    mask_class = [data_legus_red[:,class_idx] == c for c in classes]
    mask_class = np.sum(mask_class, axis=0).astype(bool)

    # Now also prune the SITELLE data according to their region definitions
    mask_class_st = [data_sitele[:,6] == c for c in classes_st]
    mask_class_st = np.sum(mask_class_st, axis=0).astype(bool)

    data_sitele_red = data_sitele[mask_class_st]
    #data_legus_red_class = data_legus_red[mask_class]

    # sitelle data
    # right ascesion of sitelle
    r_asc_st_r = [i[1] for i in data_sitele_red]

    # Declination of sitelle
    decl_st_r = [i[2] for i in data_sitele_red]

    # find nearest neighbor and of the searched for class of clusters
    data_legus_red_class = data_legus_red[mask_class]

    r_asc_le_r_c = np.asarray(r_asc_le_red)[mask_class]
    decl_le_r_c = np.asarray(decl_le_red)[mask_class]

    features_sitele = np.array([r_asc_st_r,decl_st_r])
    features_legus = np.array([r_asc_le_r_c,decl_le_r_c])

    print('shape legus', np.shape(data_legus))
    print('shape legus red', np.shape(data_legus_red))
    print('shape legus_red_class', np.shape(data_legus_red_class))
    print('shape sitelle', np.shape(data_sitele))
    print('shape sitelle red', np.shape(data_sitele_red))
    a = Nearest_Neighbor(features_sitele, features_legus, sep, axis=axis,double_use=False)# prev 0.00015 , from topcat 0.000281
    print('matching: ', np.shape(a))
    return data_sitele_red, data_legus_red_class, a
    
def prune_to_relevant_data(data_sitele, data_legus, num_of_mag_error_less_than_0p3, max_age, classes_sitelle, classes_legus, class_idx_for_legus_classes=33, axis=1, min_age=0):
    '''
    returns the pruned data sets of data_sitele and data_legus as well as the matching array a
    '''
    classes = classes_legus
    classes_st = classes_sitelle
    r_asc_st = np.asarray(data_sitele)[:,1] # right ascesion of SITELLE data 
    decl_st =  np.asarray(data_sitele)[:,2] # declination of SITELLE data

    r_asc_le = np.asarray(data_legus)[:,3] # right ascesion of LEGUS data
    decl_le =  np.asarray(data_legus)[:,4] # declination of LEGUS data

    # get data which error is less than 0.3 mag in at least num_of_mag_error_less_than_0p3 bands
    bool1 = np.array(data_legus[:,6] < 0.3, dtype=np.int32)
    bool2 = np.array(data_legus[:,8] < 0.3, dtype=np.int32)
    bool3 = np.array(data_legus[:,10] < 0.3, dtype=np.int32)
    bool4 = np.array(data_legus[:,12] < 0.3, dtype=np.int32)
    bool5 = np.array(data_legus[:,14] < 0.3, dtype=np.int32)
    
    sum_bool = bool1 + bool2 + bool3 + bool4 + bool5

    mask_leg = sum_bool >= num_of_mag_error_less_than_0p3 # previously 4 and in legus paper have a look at end of page 6

    # prune for max_age
    bool6 = np.array(data_legus[:,16] <= max_age, dtype=np.int32)
    # prune for min_age
    bool7 = np.array(data_legus[:,16] > min_age, dtype=np.int32)
    sum_bool2 = mask_leg.astype(np.int32) + bool6 + bool7
    mask_leg = sum_bool2 >=3 #2

    r_asc_le_red = np.asarray(r_asc_le)[mask_leg]
    decl_le_red = np.asarray(decl_le)[mask_leg]
    data_legus_red = data_legus[mask_leg]

    # for legus two different approaches were used when classifing the clusters into the 5 categories 
    # for the mode_class idx 33 for the mean_class idx 34
    class_idx = class_idx_for_legus_classes
    mask_class = [data_legus_red[:,class_idx] == c for c in classes]
    mask_class = np.sum(mask_class, axis=0).astype(bool)

    # Now also prune the SITELLE data according to their region definitions
    mask_class_st = [data_sitele[:,6] == c for c in classes_st]
    mask_class_st = np.sum(mask_class_st, axis=0).astype(bool)

    data_sitele_red = data_sitele[mask_class_st]
    #data_legus_red_class = data_legus_red[mask_class]

    # sitelle data
    # right ascesion of sitelle
    r_asc_st_r = [i[1] for i in data_sitele_red]

    # Declination of sitelle
    decl_st_r = [i[2] for i in data_sitele_red]

    # find nearest neighbor and of the searched for class of clusters
    data_legus_red_class = data_legus_red[mask_class]

    r_asc_le_r_c = np.asarray(r_asc_le_red)[mask_class]
    decl_le_r_c = np.asarray(decl_le_red)[mask_class]

    features_sitele = np.array([r_asc_st_r,decl_st_r])
    features_legus = np.array([r_asc_le_r_c,decl_le_r_c])

    print('shape legus', np.shape(data_legus))
    print('shape legus red', np.shape(data_legus_red))
    print('shape legus_red_class', np.shape(data_legus_red_class))
    print('shape sitelle', np.shape(data_sitele))
    print('shape sitelle red', np.shape(data_sitele_red))
    a = create_matching_for_embedded(data_sitele_red, data_legus_red_class)
    print('matching: ', np.shape(a))
    return data_sitele_red, data_legus_red_class, a

# some function for the FOV plot
def rotate_image(x, y, angle):
    '''
    Rotates the points x,y around the origin of the axis
    '''
    angle = np.deg2rad(angle)
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    res = np.matmul(rotation_matrix, [x,y])
    return res[0], res[1]

def count_max_min_within_percentage(v, percentage):
    v = np.asarray(v)
    maxi = np.max(v)
    count_max = len(v[v >= maxi-percentage*maxi])
    mini = np.min(v)
    count_min = len(v[v <= mini+percentage*mini])
    return count_max+count_min

def find_angle(x,y, percentage=0.00005, stepsize=0.05):
    curr_best = 0
    best_angle = None
    for angle in np.arange(0,45,stepsize):
        x_rot, y_rot = rotate_image(x,y,angle)
        cur = count_max_min_within_percentage([x_rot,y_rot], percentage)
        if cur > curr_best:
            best_angle = angle
            curr_best = cur
    return best_angle

def get_indices_of_objects_in_FOV(rasc, decl, rasc_check, decl_check, tolerance_factor=1):
    '''
    returns the indices of the objects which lay within the boundaries of the FOV of the x,y observation (x,y first needs to be rotated with rotate_image by an angle which is defined in find_angle)
    
    the tolerance_factor is defining the sharpness of the FOV. 1 means sharpe edges and everything above 1 is the fraction of tolerance, e.g. 1.2 means up to 20% wider edges.
    '''
    # x and y observations which will be rotated
    x = rasc
    y = decl
    # 
    x_check = rasc_check
    y_check = decl_check
    
    # find the angle the rotation should be made
    angle = find_angle(x, y)
    # rotate both datasets according to the found angle
    x_check, y_check = rotate_image(x_check, y_check, angle)
    x, y = rotate_image(x, y, angle)
    
    max_x = np.max(x)
    max_y = np.max(y)
    min_x = np.min(x)
    min_y = np.min(y)
    indices = []
    for i in range(len(x_check)):
        if x_check[i] < max_x*tolerance_factor and x_check[i] > min_x*tolerance_factor and y_check[i] < max_y*tolerance_factor and y_check[i] > min_y*tolerance_factor:
            indices.append(i)
    return np.asarray(indices)

    

#    
def create_matching_array_out_of_dictionary(dic):
    a = []
    for key in dic.keys():
        for element in dic[key]:
            a.append([key, element])
    return a

def create_matching_for_embedded(SITELLE, LEGUS):
    dic = clusters_embedded_in_HII_region(SITELLE, LEGUS, save_positional_index=True)
    a = create_matching_array_out_of_dictionary(dic)
    a = np.asarray(a)
    return a

def prune_machine_learning_data_set_for_nan_and_inf(X,Y, prune_zeros=False):
    '''
    returns the pruned X and Y data
    
    X is the input features
    Y are the corresponding keys
    '''
    mask_for_invalid = np.ones(len(X))
    for idx,e in enumerate(X):
        idx_inf_nan = np.isfinite(e)
        if isinstance(idx_inf_nan, list) or isinstance(idx_inf_nan, np.ndarray):
            if False in idx_inf_nan:
                mask_for_invalid[idx] = 0
            if prune_zeros == True:
                idx_0 = np.where(e == 0)
                mask_for_invalid[idx_0] = 0
        else:
            if False == idx_inf_nan:
                mask_for_invalid[idx] = 0
            if prune_zeros == True:
                idx_0 = np.where(e == 0)
                mask_for_invalid[idx_0] = 0
    
    for idx,e in enumerate(Y):
        idx_inf_nan = np.isfinite(e)
        if False == idx_inf_nan:
            mask_for_invalid[idx] = 0
        if prune_zeros == True:
            idx_0 = np.where(e == 0)
            mask_for_invalid[idx_0] = 0
    
    mask_for_invalid = mask_for_invalid.astype(bool)

    X = X[mask_for_invalid]
    Y = Y[mask_for_invalid]
    return X, Y
    
# some functions for the scree plot and clustering    
def return_not_nan_or_inf(x, y):
    '''
    returns x, y but delets the positions containing nans or infs occuring in either array and deletes this position in both arrays
    
    i.e: x_in = [nan, 1, 2], y_in = [1,2,inf] -> x_out = [1], y_out = [2]
    '''
    check_for = [x, y]
    mask_for_invalid = np.ones(len(x))
    for e in check_for:
        idx_inf_nan = [i for i, arr in enumerate(e) if not np.isfinite(arr).all()]
        idx_0 = np.where(e == 0)

        mask_for_invalid[idx_0] = 0
        mask_for_invalid[idx_inf_nan] = 0

    mask_for_invalid = mask_for_invalid.astype(bool)

    x = x[mask_for_invalid]
    y = y[mask_for_invalid]
    return x,y

def return_not_nan_or_inf_n_features(feature_array, return_mask=False):
    '''
    returns x, y but delets the positions containing nans or infs occuring in either array and deletes this position in both arrays
    
    i.e if dim feature_array=2: x_in = [nan, 1, 2], y_in = [1,2,inf] -> x_out = [1], y_out = [2]
    '''
    check_for = feature_array
    mask_for_invalid = np.ones(len(feature_array[0]))
    for e in check_for:
        idx_inf_nan = [i for i, arr in enumerate(e) if not np.isfinite(arr).all()]
        idx_0 = np.where(e == 0)

        mask_for_invalid[idx_0] = 0
        mask_for_invalid[idx_inf_nan] = 0

    mask_for_invalid = mask_for_invalid.astype(bool)

    res = [feature_array[i][mask_for_invalid] for i in range(len(feature_array))]
    if return_mask == True:
        return np.asarray(res), mask_for_invalid
    else:
        return np.asarray(res)

def euclidean_norm(vec):
    '''
    returns the euclidean norm of the vector v
    
    source:
    https://en.wikipedia.org/wiki/Norm_(mathematics)
    '''
    return np.sqrt(np.sum(np.asarray(vec)**2, axis=1))

def calculate_within_cluster_variance(x,y):
    '''
    returns the variance of the cluster which is defined over the x and y coordinates
    
    source:
    https://stats.stackexchange.com/questions/86645/variance-within-each-cluster
    '''
    x_c = np.mean(x)
    y_c = np.mean(y)
    centroid = np.array([x_c, y_c])
    vec_i = np.asarray([[x[i],y[i]] for i in range(len(x))])
    return np.sum(euclidean_norm(vec_i-centroid))
    
def scree_plot(x, y, max_clusters=20, return_labels=False):
    '''
    creates a scree_plot according to the x and y data
    '''
    var = []
    if return_labels == True:
        labels_arr = []
    for i in range(1,max_clusters):
        X = np.asarray([np.append(x[i],y[i]) for i in range(len(x))])
        clustering = KMeans(n_clusters=i, random_state=0, n_init=10).fit(X)
        labels = clustering.labels_
        if return_labels == True:
            labels_arr.append([labels])
        var_temp = []
        for l in range(i):
            mask = np.asarray(labels) == l
            var_temp.append(calculate_within_cluster_variance(x[mask], y[mask]))
        var_mean = np.sum(var_temp)
        var.append(var_mean)
    i = range(1,max_clusters)
    plt.xlabel('number of clusters')
    plt.ylabel('sum of within cluster variance')
    plt.plot(i, var)
    if return_labels == True:
        return labels_arr

def calculate_within_cluster_variance_for_n_features(X):
    '''
    returns the variance of the cluster which is defined over the x and y coordinates
    
    source:
    https://stats.stackexchange.com/questions/86645/variance-within-each-cluster
    '''
    # multidimensional centroid
    centroid = [np.mean(X[:,i]) for i in range(len(X[0]))]
    # now create an vector for every element in X 
    vec_i = X
    return np.sum(euclidean_norm(vec_i-centroid))    

def scree_plot_n_features(Input, max_clusters=20, return_labels=False):
    '''
    creates a scree_plot according to the x and y data
    
    Input: [[feature1],[feature2],[feature3],...]
    '''
    var = []
    Input = np.asarray(Input)
    if return_labels == True:
        labels_arr = []
    for i in range(1,max_clusters):
        X = np.asarray([Input[:,i] for i in range(len(Input[0]))])
        clustering = KMeans(n_clusters=i, random_state=0, n_init=10).fit(X)
        labels = clustering.labels_
        if return_labels == True:
            labels_arr.append([labels])
        var_temp = []
        for l in range(i):
            mask = np.asarray(labels) == l
            var_temp.append(calculate_within_cluster_variance_for_n_features(X[mask]))
        var_mean = np.sum(var_temp)
        var.append(var_mean)
    i = range(1,max_clusters)
    plt.xlabel('number of clusters')
    plt.ylabel('sum of within cluster variance')
    plt.plot(i, var)
    if return_labels == True:
        return labels_arr

#        
def add_magnitudes_in_vega_system(m1, m2, Filter_id=None, ID=None):
    '''
    returns the added magnitudes of m1 and m2
    
    source:
    http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf
    '''
    #zeropoint_legus_c = {5:22.632, 7:23.484, 9:25.784, 11:25.731, 13:25.530}
    #zeropoint_legus_d = {5:22.632, 7:23.484, 9:25.784, 11:25.816, 13:25.530}
    # Add fluxes and transform them in magnitudes, since the zeropoint of m1 and m2 should be the same it can be reduced
    return -2.5*np.log10(10**(-m1/2.5)+10**(-m2/2.5))

def create_dataset_for_pca(data_sitelle_red, data_legus_red, indx_feature_sitelle, indx_feature_legus, scale_data=True, print_used_features=False, return_pruning_mask=False, return_mask_of_observations=False):
    '''
    returns the dataset X which is pruned for nans or infs where X is build by the appendation of Sitelle and Legus data [[s11,...,sn1,l11,...,lm1],[s12,...,sn2,l12,...lm2],...] where n = len(indx_feature_sitelle) and m = len(indx_feature_legus)
    
    If scale = True the features will be scaled to mean 0 and unit standarddev within the perspective feature space
    '''
    data_legus_red_2 = feature_selection(data_legus_red, indx_feature_legus)
    data_sitelle_red_2 = feature_selection(data_sitelle_red, indx_feature_sitelle)

    X = np.append(data_sitelle_red_2, data_legus_red_2, axis=1)
    # prune data so it does not contain any invalid instances look documentation of return_not_nan_or_inf_n_features for more detail
    Xt, mask1 = return_not_nan_or_inf_n_features(X.T, return_mask=True)
    Xt = np.asarray(Xt)
    X = Xt.T
    X, mask = return_not_nan_or_inf_n_features(X, return_mask=True)
    mask_sitelle = mask[:len(indx_feature_sitelle)]
    mask_legus = mask[len(indx_feature_sitelle):]
    X = np.asarray(X)
    if print_used_features == True:
        print('legus features: ----------------')
        for i in range(len(indx_feature_legus)):
            if mask_legus[i] == True:
                print(dir_legus_featurename_and_id[indx_feature_legus[i]])

        print('sitelle features: ----------------')
        for i in range(len(indx_feature_sitelle)):
            if mask_sitelle[i] == True:
                print(dir_sitelle_featurename_and_id[indx_feature_sitelle[i]])
        print(np.shape(data_legus_red), np.shape(data_sitelle_red), np.shape(X))
    
    if scale_data == True:
        X_temp = []
        for i in range(np.shape(X)[1]): 
            
            #scaler_all = StandardScaler()
            #imp = Imputer(strategy="mean", axis=0)
            #X_temp.append(scaler_all.fit_transform(imp.fit_transform(X[:,i])))
            #X_temp.append(X[:,i])
            X_temp.append(scale(X[:,i]))
        
        X_temp = np.asarray(X_temp)
        #np.testing.assert_array_equal(X.T, X_temp)
        # reconstuct X
        X = np.asarray([[X_temp[j,i] for j in range(len(X_temp))] for i in range(len(X_temp[0]))])
    
    if return_mask_of_observations == True:
        return X, mask_sitelle, mask_legus
    elif return_pruning_mask == True:
        return X, mask1
    
    else:
        return X
    
#############################################################################################
# Plot functions
#############################################################################################

def scale_for_plot(scale, color='black', tick_locs_x = 1, tick_locs_y = -2):
    locs_x, labelsx = plt.xticks()
    locs_y, labelsy = plt.yticks()
    
    s = transform_pc_in_ra_dec(scale, 9.9e6)
    plt.plot([locs_x[tick_locs_x], locs_x[tick_locs_x]+s],[locs_y[tick_locs_y], locs_y[tick_locs_y]], color = color)
    plt.annotate(str(scale)+'pc', xy=(locs_x[tick_locs_x],locs_y[tick_locs_y]*1.0001), fontsize=11)

def Visualize_Nearest_Neighbors(index_list, r_asc_1, decl_1, r_asc_2, decl_2, figsize=7, scatter=False):
    '''
    visualizes the data with connecting lines between the found ('index_list') Nearest Neighbors
    '''
    r_asc_le= r_asc_2
    decl_le = decl_2
    r_asc_st= r_asc_1
    decl_st = decl_1
    #plt.figure(figsize=(figsize,figsize))
    if scatter == True:
        plt.scatter(r_asc_le, decl_le, label='legus', s=0.5)
        plt.scatter(r_asc_st, decl_st, label='sitele', s=0.5)
    for i in range(len(index_list)):
        plt.plot([r_asc_le[index_list[i][1]], r_asc_st[index_list[i][0]]],[decl_le[index_list[i][1]], decl_st[index_list[i][0]]], 'r-')

    #plt.xlabel('right ascesion')
    #plt.ylabel('Declination')
    #plt.ylim(15.73, 15.82)
    #plt.xlim(24.12,24.21)
    #plt.legend()
    #plt.show()
    
def visualization_of_matches_with_HII_boundaries(a, SITELLE, LEGUS, figsize=7, return_limits=False):
    '''
    a is the matching array of the catalogs SITELLE and LEGUS with their ID's like [[S_id1, L_id1],[S_id2, L_id2],...]
    SITELLE data
    LEGUS data
    '''
    
    SITELLE = np.asarray(SITELLE)
    LEGUS = np.asarray(LEGUS)
    a = np.asarray(a)
    
    r_asc_le = -LEGUS[a[:,1],3]
    decl_le = LEGUS[a[:,1],4]
    r_asc_st = -SITELLE[a[:,0],1]
    decl_st = SITELLE[a[:,0],2]
    size = SITELLE[a[:,0],12]
    radius_HII = transform_pc_in_ra_dec(size)
    assert(len(r_asc_st) == len(a))
    
    Circles = []
    for i in range(len(a)):
        Circles.append(plt.Circle((r_asc_st[i], decl_st[i]), radius_HII[i], color='green', alpha=0.2))
    
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    
    for c in Circles:
        ax.add_artist(c)
    
    r_asc_le = -LEGUS[:,3]
    decl_le = LEGUS[:,4]
    r_asc_st = -SITELLE[:,1]
    decl_st = SITELLE[:,2]
    
    ax.scatter(np.asarray(r_asc_st), decl_st, s=0.5, label='SITELLE {} datapoints'.format(len(SITELLE)))
    ax.scatter(np.asarray(r_asc_le), decl_le, s=0.5, label='LEGUS {} datapoints'.format(len(LEGUS)))
    
    if return_limits == True:
        x_lower = np.min(np.append(r_asc_le, r_asc_st))
        x_upper = np.max(np.append(r_asc_le, r_asc_st))
        x_lower_boundary = x_lower-(x_upper-x_lower)*0.01
        x_upper_boundary = x_upper+(x_upper-x_lower)*0.01
        y_lower = np.min(np.append(decl_le, decl_st))
        y_upper = np.max(np.append(decl_le, decl_st))
        y_lower_boundary = y_lower-(y_upper-y_lower)*0.01
        y_upper_boundary = y_upper+(y_upper-y_lower)*0.01
        return [x_lower_boundary, x_upper_boundary], [y_lower_boundary, y_upper_boundary]


# plot correspondance of found neighbors
def plot_features_of_found_pairs(a, f1, f2, distance_matrix=None):
    '''
    a contains the found nearest neighbors [(sitele, legus),(s,l),...]
    f1 is the to observing feature of the first dataset
    f2 is the to observing feature of the second dataset
    '''
    mask_1 = [i[0] for i in a] # sitele
    mask_2 = [i[1] for i in a] # legus

    # mask the features corresponding to their neighbors
    f1 = np.asarray(f1)[mask_1]
    f2 = np.asarray(f2)[mask_2]

    if distance_matrix!=None:
        distance = []
        for elmt in a:
            distance.append(distance_matrix[elmt[1]][elmt[0]])
        
        distance = np.asarray(distance, dtype='float64')[np.nonzero(f2)]
        
        # leave out the data which does not exist i.e if the mass is given as 0   
        f1 = f1[np.nonzero(f2)]
        f2 = f2[np.nonzero(f2)]
        plt.scatter(f1, f2, c=(distance), cmap='cool')
        plt.colorbar()
        
    if distance_matrix==None:
        # leave out the data which does not exist i.e if the mass is given as 0   
        f1 = f1[np.nonzero(f2)]
        f2 = f2[np.nonzero(f2)]
        plt.scatter(f1, f2)
    plt.xlabel('luminosity of HII-Region')
    plt.ylabel('mass of cluster')
    plt.ylim(1e2, 1e7)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e34, 1e40)
    plt.show()

def plot_class_of_cluster(r_asc, dec, class_):
    plt.figure(figsize=(9,7))
    cmap = mpl.colors.ListedColormap(['red','red', 'blue'])
    bounds = [0.5, 2.5, 3.5]
    
    r_asc = np.asarray(r_asc)
    dec = np.asarray(dec)
    class_ = np.asarray(class_)
    r_asc = r_asc[(class_== 1) | (class_==2) | (class_==3)]
    dec = dec[(class_== 1) | (class_==2) | (class_==3)]
    plt.scatter(r_asc, dec, c=(class_[(class_== 1) | (class_==2) | (class_==3)]), cmap=cmap, s=0.5)
    plt.colorbar(boundaries=[0.5] + bounds + [3.5], ticks=[1,2,3], spacing='proportional')
    
    plt.show()
'''
def plot_features_of_found_pairs_using_classes(c, r_asc_st=r_asc_st, decl_st=decl_st, r_asc_le=r_asc_le, decl_le=decl_le, class_legus=class_legus, data_legus=data_legus, idx_f1=4, idx_f2=19):
    features_sitele = np.array([r_asc_st,decl_st])
    features_legus = np.array([r_asc_le,decl_le])

    classes = c
    print(len(class_legus))
    features_legus_red = np.array([[r_asc_le[i] for i in range(len(r_asc_le)) if class_legus[i] in classes], [decl_le[i] for i in range(len(decl_le)) if class_legus[i] in classes]])

    a = Nearest_Neighbor(features_sitele, features_legus_red, 0.0009, double_use=False) # previously 0.0005

    legus_mass_red = [data_legus[i][idx_f2] for i in range(len(data_legus)) if class_legus[i] in classes]

    #print(np.shape(a))

    # Ha total luminosity corrected for extinction
    sitele_luminosity = [i[idx_f1] for i in data_sitele]

    print(np.shape(a))

    dist_mat_red = create_distance_matrix(features_sitele, features_legus_red)
    plot_features_of_found_pairs(a, sitele_luminosity, legus_mass_red, dist_mat_red)
'''

def plot_features_of_found_pairs_using_classes(c, r_asc_st, decl_st, r_asc_le, decl_le, class_legus, data_legus, a_thresh=0.0009, idx_f1=4, idx_f2=19, dist=None):
    features_sitele = np.array([r_asc_st,decl_st])
    features_legus = np.array([r_asc_le,decl_le])

    classes = c
    print(len(class_legus))
    features_legus_red = np.array([[r_asc_le[i] for i in range(len(r_asc_le)) if class_legus[i] in classes], [decl_le[i] for i in range(len(decl_le)) if class_legus[i] in classes]])

    a = Nearest_Neighbor(features_sitele, features_legus_red, a_thresh, double_use=False, actual_dist=dist) # previously 0.0005

    legus_mass_red = [data_legus[i][idx_f2] for i in range(len(data_legus)) if class_legus[i] in classes]

    #print(np.shape(a))

    # Ha total luminosity corrected for extinction
    sitele_luminosity = [i[idx_f1] for i in data_sitele]

    print(np.shape(a))

    dist_mat_red = create_distance_matrix(features_sitele, features_legus_red)
    plot_features_of_found_pairs(a, sitele_luminosity, legus_mass_red, dist_mat_red)
    
def multiplot(legus_mask_for_features, sitelle_mask_for_features, data_legus, data_sitele, a, filename, path=r'D:\Uni\Bachelor_Arbeit\Plot', class_idx = 33, xscales=None, yscales=None, save_XY_data=False, classes_le=None, classes_st=None, frequ=2, set_xlim=None, set_ylim=None):
    # only one plot:
    fig, axes = plt.subplots(len(legus_mask_for_features), len(sitelle_mask_for_features), figsize=(8.0*len(sitelle_mask_for_features), 6.0*len(legus_mask_for_features))) #, subplot_kw=dict(polar=True)
    
    # the indices and the corresponding feature of the legus data 
    dir_legus = {0:"source id", 1:"x corr", 2:"y corr", 3:"RA", 4:"DEC", 5:"final total mag in WFC3/F275W", 6:"final photometric error in WFC3/F275W", 7:"final total mag in WFC3/F336W", 8:"final photometric error in WFC3/F336W", 9:"final total mag in ASC/F435W", 10:"final photometric error in ASC/F435W", 11:"final total mag in ASC/F555W", 12:"final photometric error in ASC/F555W", 13:"final total mag in ASC/F814W", 14:"final photometric error in ASC/F814W", 15:"CI=mag(1px)-mag(3px)", 16:"best age in yr", 17:"max age in yr (within 68% confidence level)", 18:"min age in yr (within 68% confidence level)", 19:"best mass in solar masses", 20:"max mass in solar masses (within 68% confidence level)", 21:"min mass in solar masses (within 68% confidence level)", 22:"best E(B-V)", 23:"max E(B-V) (within 68% confidence level)", 24:"min E(B-V) (within 68% confidence level)", 25:"chi2 fit residual in F275W", 26:"chi2 fit residual in F336W", 27:"chi2 fit residual in F435W", 28:"chi2 fit residual in F555W", 29:"chi2 fit residual in F814W", 30:"reduced chi2", 31:"Q probability", 32:"Number of filter", 33:"Final assigned class", 34:"Final assigned class after visual inspection"}
    # the indices and the corresponding feature of the sitelle data 
    dir_sitelle= {0:"ID of Region", 1:"RA", 2:"DEC", 3:"Galactocentric radius", 4:r"$H \alpha$ total luminosity corrected for extinction $[erg\,s^{-1} cm^{-2}]$", 5:"Ha mean diffuse ionized gaz backgound level", 6:"Region category", 7:"I0", 8:"Amp", 9:"sig", 10:"alpha", 11:"R2", 12:" size; pc size of the regions", 13:"EBV; extinction; E(B-V)", 14:"EBV_err; extinction error; E(B-V error)", 15:"log [NII]6583/Ha; log line ratio", 16:"error on log [NII]6583/Ha; log line ratio error", 17:"SNR_cross [NII]6583/Ha; line ratio best SNR", 18:"log [SII]6716+6731/Ha; log line ratio", 19:"error on log [SII]6716+6731/Ha; log line ratio error", 20:"SNR_cross [SII]6716+6731/Ha; line ratio best SNR", 21:"log log [SII}6716+6731/[NII]6583; log line ratio", 22:"error on log [SII}6716+6731/[NII]6583; log line ratio error", 23:"SNR_cross on [SII}6716+6731/[NII]6583; line ratio best SNR", 24:"log [OIII]5007/Hb ; log line ratio", 25:"error on log [OIII]5007/Hb ; log line ratio error", 26:"SNR_cross on [OIII]5007/Hb; line ratio best SNR", 27:"log [OII]3727/Hb ; log line ratio", 28:"error on log [OII]3727/Hb ; log line ratio error", 29:"SNR_cross on [OII]3727/Hb; line ratio best SNR", 30:"log ([OII]3727+[OIII]5007)/Hb ; log line ratio", 31:"error on log ([OII]3727+[OIII]5007)/Hb ; log line ratio error", 32:"SNR_cross on ([OII]3727+[OIII]5007)/Hb; line ratio best SNR", 33:"log [OIII]5007/[OII]3727 ; log line ratio", 34:"error on log [OIII]5007/[OII]3727 ; log line ratio error", 35:"SNR_cross on [OIII]5007/[OII]3727; line ratio best SNR", 36:"log [OIII]5007/[NII}6583 ; log line ratio", 37:"error on log [OIII]5007/[[NII}6583 ; log line ratio error", 38:"SNR_cross on [OIII]5007/[[NII}6583; line ratio best SNR", 39:"log [OII]3727/[NII}6583 ; log line ratio", 40:"error on log [OII]3727/[[NII}6583 ; log line ratio error", 41:"SNR_cross on [OII]3727/[[NII}6583; line ratio best SNR", 42:"[SII]6716/[SII]6731; line ratio", 43:"error on [SII]6716/[SII]6731; line ratio error", 44:"SNR_cross on [SII]6716/[SII]6731; line ratio best SNR"}
    if save_XY_data == True:
        XY = [[None for l in range(len(legus_mask_for_features))] for s in range(len(sitelle_mask_for_features))]
    for l in range(len(legus_mask_for_features)):
        for s in range(len(sitelle_mask_for_features)):
                
            X = np.asarray([data_legus[i,legus_mask_for_features[l]] for i in np.asarray(a)[:,1]]).astype(float)
            Y = np.asarray([data_sitele[i,sitelle_mask_for_features[s]] for i in np.asarray(a)[:,0]]).astype(float)
            
            if save_XY_data == True:
                XY[s][l] = [X,Y]
            
            # plot for every feature pair
            #plt.scatter(X,Y)
            #plt.xlabel(dir_legus[legus_mask_for_features[l]])
            #plt.ylabel(dir_sitelle[sitelle_mask_for_features[s]])
            #plt.savefig(r'D:\Uni\Bachelor_Arbeit\Plot\correlation_{}_{}.png'.format(dir_legus[legus_mask_for_features[l]], dir_sitelle[sitelle_mask_for_features[s]]))
            #plt.close()
            
            # dict for the markers and colors indicating the different classes of the clusters and HII-regions
            dict_cluster = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
            dict_HII = {1:'#d62728', 2:'b', 3:'#17becf', 4:'m'}
            
            cmap = mpl.cm.get_cmap('tab20b')
            dict_cluster = {1:cmap(0), 2:cmap(4), 3:cmap(8), 4:cmap(12), 0:cmap(16)}
            dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
            
            # only one plot:
            for i in range(len(X)):
                axes[l, s].scatter(X[i],Y[i], marker=dict_HII[data_sitele[np.asarray(a)[i,0],6]], alpha=0.7, c=dict_cluster[data_legus[np.asarray(a)[i,1],class_idx]])
            axes[l, s].set(xlabel=dir_legus[legus_mask_for_features[l]], ylabel=dir_sitelle[sitelle_mask_for_features[s]])
            if xscales != None:
                axes[l, s].set_xscale(xscales[l])
            if yscales != None:
                axes[l, s].set_yscale(yscales[s])
            #axes[l, s].ylabel(dir_sitelle[sitelle_mask_for_features[s]])
            
            if l%frequ == 0 and s%frequ == 0:
                for c in classes_le:
                    axes[l,s].scatter(np.inf,0, alpha=0.7, c=dict_cluster[c], marker='o', s=100, label='Clustertype {}'.format(c))
                    
                for c in classes_st:
                    axes[l,s].scatter(np.inf,0, alpha=0.7, marker=dict_HII[c], c='black', s=12, label='HII-type {}'.format(c))
                axes[l,s].legend(markerscale=6., loc='best')
            
            if set_xlim != None:
                if set_xlim[l] != None:
                    axes[l,s].set_xlim(set_xlim[l])
            if set_ylim != None:
                if set_ylim[s] != None:
                    axes[l,s].set_ylim(set_ylim[s])
                
    # only one plot:
    fig.savefig(path+'\{}'.format(filename), dpi=100, bbox_inches='tight')
    if save_XY_data == True:
        return XY

def multiplot_consider_multiple_clusters_in_HII(legus_mask_for_features, sitelle_mask_for_features, data_legus, data_sitele, a, dire, filename, path=r'D:\Uni\Bachelor_Arbeit\Plot', class_idx = 33, xscales=None, yscales=None, save_XY_data=False, classes_le=None, classes_st=None, frequ=2, set_xlim=None, set_ylim=None, agebins=None, maxaged_cluster = False, print_corr_coef=False,legend_fontsize=11):
    # only one plot:
    fig, axes = plt.subplots(len(legus_mask_for_features), len(sitelle_mask_for_features), figsize=(8.0*len(sitelle_mask_for_features), 6.0*len(legus_mask_for_features))) #, subplot_kw=dict(polar=True)
    magnitude_features = [5,7,9,11,13]
    # the indices and the corresponding feature of the legus data 
    dir_legus = {0:"source id", 1:"x corr", 2:"y corr", 3:"RA", 4:"DEC", 5:"final total mag in WFC3/F275W", 6:"final photometric error in WFC3/F275W", 7:"final total mag in WFC3/F336W", 8:"final photometric error in WFC3/F336W", 9:"final total mag in ASC/F435W", 10:"final photometric error in ASC/F435W", 11:"final total mag in ASC/F555W", 12:"final photometric error in ASC/F555W", 13:"final total mag in ASC/F814W", 14:"final photometric error in ASC/F814W", 15:"CI=mag(1px)-mag(3px)", 16:"best age in yr", 17:"max age in yr (within 68% confidence level)", 18:"min age in yr (within 68% confidence level)", 19:"best mass in solar masses", 20:"max mass in solar masses (within 68% confidence level)", 21:"min mass in solar masses (within 68% confidence level)", 22:"best E(B-V)", 23:"max E(B-V) (within 68% confidence level)", 24:"min E(B-V) (within 68% confidence level)", 25:"chi2 fit residual in F275W", 26:"chi2 fit residual in F336W", 27:"chi2 fit residual in F435W", 28:"chi2 fit residual in F555W", 29:"chi2 fit residual in F814W", 30:"reduced chi2", 31:"Q probability", 32:"Number of filter", 33:"Final assigned class", 34:"Final assigned class after visual inspection"}
    # the indices and the corresponding feature of the sitelle data 
    dir_sitelle= {0:"ID of Region", 1:"RA", 2:"DEC", 3:"Galactocentric radius", 4:r"$H \alpha$-luminosity corrected for extinction $[erg\,s^{-1}]$", 5:"Ha mean diffuse ionized gaz backgound level", 6:"Region category", 7:"I0", 8:"Amp", 9:"sig", 10:"alpha", 11:"R2", 12:"mean radius/size of HII-region [pc]", 13:"EBV; extinction; E(B-V)", 14:"EBV_err; extinction error; E(B-V error)", 15:"log [NII]6583/Ha; log line ratio", 16:"error on log [NII]6583/Ha; log line ratio error", 17:"SNR_cross [NII]6583/Ha; line ratio best SNR", 18:"log [SII]6716+6731/Ha; log line ratio", 19:"error on log [SII]6716+6731/Ha; log line ratio error", 20:"SNR_cross [SII]6716+6731/Ha; line ratio best SNR", 21:"log log [SII}6716+6731/[NII]6583; log line ratio", 22:"error on log [SII}6716+6731/[NII]6583; log line ratio error", 23:"SNR_cross on [SII}6716+6731/[NII]6583; line ratio best SNR", 24:"log [OIII]5007/Hb ; log line ratio", 25:"error on log [OIII]5007/Hb ; log line ratio error", 26:"SNR_cross on [OIII]5007/Hb; line ratio best SNR", 27:"log [OII]3727/Hb ; log line ratio", 28:"error on log [OII]3727/Hb ; log line ratio error", 29:"SNR_cross on [OII]3727/Hb; line ratio best SNR", 30:"log ([OII]3727+[OIII]5007)/Hb ; log line ratio", 31:"error on log ([OII]3727+[OIII]5007)/Hb ; log line ratio error", 32:"SNR_cross on ([OII]3727+[OIII]5007)/Hb; line ratio best SNR", 33:"log [OIII]5007/[OII]3727 ; log line ratio", 34:"error on log [OIII]5007/[OII]3727 ; log line ratio error", 35:"SNR_cross on [OIII]5007/[OII]3727; line ratio best SNR", 36:"log [OIII]5007/[NII}6583 ; log line ratio", 37:"error on log [OIII]5007/[[NII}6583 ; log line ratio error", 38:"SNR_cross on [OIII]5007/[[NII}6583; line ratio best SNR", 39:"log [OII]3727/[NII}6583 ; log line ratio", 40:"error on log [OII]3727/[[NII}6583 ; log line ratio error", 41:"SNR_cross on [OII]3727/[[NII}6583; line ratio best SNR", 42:"[SII]6716/[SII]6731; line ratio", 43:"error on [SII]6716/[SII]6731; line ratio error", 44:"SNR_cross on [SII]6716/[SII]6731; line ratio best SNR"}
    # No agebins
    if agebins == None:
        if save_XY_data == True:
            XY = [[None for l in range(len(legus_mask_for_features))] for s in range(len(sitelle_mask_for_features))]
        for l in range(len(legus_mask_for_features)):
            for s in range(len(sitelle_mask_for_features)):
                if (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) > 1):
                    curr_axes = axes[l, s]
                elif (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) == 1):
                    curr_axes = axes[l]
                elif (len(legus_mask_for_features) == 1 and len(sitelle_mask_for_features) > 1):
                    curr_axes = axes[s]
                else:
                    curr_axes = axes
                already_used_HII_region = []
                # create datasets Y are the HII-regions and X the clusters
                # but this time if there are multiple clusters in one HII-region the features of the clusters will be summed up

                X = []
                Y = []
                for i in a[:,0]:
                    if i not in already_used_HII_region:
                        Y.append(data_sitele[i,sitelle_mask_for_features[s]])
                        if legus_mask_for_features[l] in magnitude_features:
                            values = dire[i]
                            magsum = data_legus[values[0], legus_mask_for_features[l]]
                            for j in range(1,len(values)):
                                magsum = add_magnitudes_in_vega_system(magsum, data_legus[values[j], legus_mask_for_features[l]])
                            X.append(magsum)
                        elif legus_mask_for_features[l] == 16:
                            values = dire[i]
                            temp = 0
                            for j in range(len(values)):
                                curr_age = data_legus[values[j], 16]
                                if temp < curr_age:
                                    temp = curr_age
                            X.append(temp)
                        else:
                            # here for other additive features
                            values = dire[i]
                            temp = 0
                            for j in range(len(values)):
                                temp += data_legus[values[j], legus_mask_for_features[l]]
                            X.append(temp)
                        already_used_HII_region.append(i)
                    else:
                        continue
                X = np.asarray(X).astype(float)
                Y = np.asarray(Y).astype(float)

                if save_XY_data == True:
                    XY[s][l] = [X,Y]

                # dict for the markers and colors indicating the different classes of the clusters and HII-regions
                #dict_cluster = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
                #dict_HII = {1:'#d62728', 2:'b', 3:'#17becf', 4:'m'}
                
                cmap = mpl.cm.get_cmap('tab20c')
                dict_cluster = {1:cmap(0), 2:cmap(4), 3:cmap(8), 4:cmap(12), 0:cmap(16)}
                dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}

                # only one plot:
                for i in range(len(X)):
                    curr_axes.scatter(X[i],Y[i], marker=dict_HII[data_sitele[np.asarray(a)[i,0],6]], alpha=0.7, c=dict_cluster[data_legus[np.asarray(a)[i,1],class_idx]])
                curr_axes.set(xlabel=dir_legus[legus_mask_for_features[l]], ylabel=dir_sitelle[sitelle_mask_for_features[s]])
                if xscales != None:
                    curr_axes.set_xscale(xscales[l])
                if yscales != None:
                    curr_axes.set_yscale(yscales[s])
                #curr_axes.ylabel(dir_sitelle[sitelle_mask_for_features[s]])

                if l%frequ == 0 and s%frequ == 0:
                    for c in classes_st:
                        curr_axes.scatter(np.inf,0, alpha=0.7, marker=dict_HII[c], c='black', s=12, label='HII-type {}'.format(c))
                    for c in classes_le:
                        curr_axes.scatter(np.inf,0, alpha=0.7, c=dict_cluster[c], marker='o', s=100, label='Clustertype {}'.format(c))

                    factor_for_legend = 1
                    if len(sitelle_mask_for_features) > 1:
                        factor_for_legend = 1.1
                    curr_axes.legend(markerscale=6., bbox_to_anchor=(0., 1.02, 1.*len(sitelle_mask_for_features)*factor_for_legend,20), loc=3, ncol=2, mode='expand', borderaxespad=0., fontsize=9)
                    #curr_axes.legend(markerscale=6., loc='best')

                if set_xlim != None:
                    if set_xlim[l] != None:
                        curr_axes.set_xlim(set_xlim[l])
                if set_ylim != None:
                    if set_ylim[s] != None:
                        curr_axes.set_ylim(set_ylim[s])

        # only one plot:
        fig.savefig(path+'\{}'.format(filename), dpi=100, bbox_inches='tight')
        if save_XY_data == True:
            return XY
    # use agebins
    # actually use a pie chart like scatter if there are multiple agebins vertretten 
    # https://matplotlib.org/examples/api/scatter_piecharts.html
    elif maxaged_cluster == True:
        cmap = mpl.cm.get_cmap('tab20b')
        
        # intervall from prune_to_relevant_data for ages is (min_age, max_age] so including max_age and excluding min_age
        if save_XY_data == True:
            XY = [[None for l in range(len(legus_mask_for_features))] for s in range(len(sitelle_mask_for_features))]
        for l in range(len(legus_mask_for_features)):
            for s in range(len(sitelle_mask_for_features)):
                if (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) > 1):
                    curr_axes = axes[l, s]
                elif (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) == 1):
                    curr_axes = axes[l]
                elif (len(legus_mask_for_features) == 1 and len(sitelle_mask_for_features) > 1):
                    curr_axes = axes[s]
                else:
                    curr_axes = axes
                already_used_HII_region = []
                # create datasets Y are the HII-regions and X the clusters
                # but this time if there are multiple clusters in one HII-region the features of the clusters will be summed up

                X = []
                X_ages = []
                Y = []
                for i in a[:,0]:
                    if i not in already_used_HII_region:
                        Y.append(data_sitele[i,sitelle_mask_for_features[s]])
                        if legus_mask_for_features[l] in magnitude_features:
                            values = dire[i]
                            magsum = None
                            max_age = None
                            for n in range(0,len(values)):
                                max_age = data_legus[values[n], 16]
                                magsum = data_legus[values[n], legus_mask_for_features[l]]
                                for j in range(n,len(values)):
                                    if data_legus[values[j], 16] > max_age:
                                        max_age = data_legus[values[j], 16]
                                    magsum = add_magnitudes_in_vega_system(magsum, data_legus[values[j], legus_mask_for_features[l]])
                                else:
                                    continue
                            if magsum == None:
                                magsum = -np.inf
                            X.append(magsum)
                            X_ages.append(max_age)
                        #elif legus_mask_for_features[l] == 16:
                        #    values = dire[i]
                        #    temp = 0
                        #    for j in range(len(values)):
                        #        curr_age = data_legus[values[j], 16]
                        #        if temp < curr_age:
                        #            temp = curr_age
                        #    X.append(temp)
                        else:
                            # here for other additive features
                            values = dire[i]
                            temp = 0
                            max_age = None
                            for j in range(len(values)):
                                max_age = data_legus[values[j], 16]
                                if data_legus[values[j], 16] > max_age:
                                    max_age = data_legus[values[j], 16]
                                temp += data_legus[values[j], legus_mask_for_features[l]]
                            if temp == 0:
                                temp = -np.inf
                            X.append(temp)
                            X_ages.append(max_age)
                        already_used_HII_region.append(i)
                    else:
                        continue
                X = np.asarray(X).astype(float)
                Y = np.asarray(Y).astype(float)
                X_ages = np.asarray(X_ages).astype(float)

                if save_XY_data == True:
                    XY[s][l] = [X,Y]

                # dict for the markers and colors indicating the different classes of the clusters and HII-regions
                dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
                #dict_agebin = {1:cmap(4*k), 2:cmap(4*k+1), 3:cmap(4*k+2), 4:cmap(4*k+3)}
                
                def get_color(agebins, age, cl):
                    agebins = np.asarray(agebins)
                    for k in range(len(agebins)):
                        max_age = agebins[k,1]
                        if age <= max_age:
                            dict_agebin = {1:cmap(4*k), 2:cmap(4*k+1), 3:cmap(4*k+2), 4:cmap(4*k+3)}
                            return dict_agebin[cl]
                
                # only one plot:
                for i in range(len(X)):
                    # cl is the legus class
                    cl = data_legus[np.asarray(a)[i,1],class_idx]
                    # get the right color
                    c = get_color(agebins, X_ages[i], cl)
                    curr_axes.scatter(X[i],Y[i], c=c, alpha=0.7, marker=dict_HII[data_sitele[np.asarray(a)[i,0],6]])
                curr_axes.set(xlabel=dir_legus[legus_mask_for_features[l]], ylabel=dir_sitelle[sitelle_mask_for_features[s]])
                if xscales != None:
                    curr_axes.set_xscale(xscales[l])
                if yscales != None:
                    curr_axes.set_yscale(yscales[s])
                #curr_axes.ylabel(dir_sitelle[sitelle_mask_for_features[s]])
                for k in range(len(agebins)):
                    agebins = np.asarray(agebins)
                    max_age = agebins[k,1]
                    min_age = agebins[k,0]
                    dict_agebin = {1:cmap(4*k), 2:cmap(4*k+1), 3:cmap(4*k+2), 4:cmap(4*k+3)}
                    if l%frequ == 0 and s%frequ == 0:
                        if k == 0:
                            for c in classes_st:
                                curr_axes.scatter(np.inf,0, alpha=0.7, c='black',marker=dict_HII[c], s=100, label='HII-type {}'.format(c))
    
                        for i,c in enumerate(classes_le):
                            if i == 0:
                                curr_axes.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_agebin[c], s=12, label='Age: {}-{}Myr\nCluster-type 1'.format(int(min_age/1e6), int(max_age/1e6)))
                            else:
                                
                                curr_axes.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_agebin[c], s=12, label='Cluster-type {}'.format(c))
                    factor_for_legend = 1
                    if len(sitelle_mask_for_features) > 1:
                        factor_for_legend = 1.1
                    curr_axes.legend(markerscale=6., bbox_to_anchor=(0., 1.02, 1.*len(sitelle_mask_for_features)*factor_for_legend,20), loc=3, ncol=1+len(agebins), mode='expand', borderaxespad=0., fontsize=legend_fontsize, handletextpad=0.1)
                    #curr_axes.legend(markerscale=6., loc='best')

                if set_xlim != None:
                    if set_xlim[l] != None:
                        curr_axes.set_xlim(set_xlim[l])
                if set_ylim != None:
                    if set_ylim[s] != None:
                        curr_axes.set_ylim(set_ylim[s])
                
                if print_corr_coef == True:
                    X_calc = X
                    Y_calc = Y
                    if xscales != None:
                        if xscales[l] == 'log':
                            X_calc = np.log10(X)
                    if yscales != None:
                        if yscales[s] == 'log':
                            Y_calc = np.log10(Y)
                    
                    X_calc, Y_calc = prune_machine_learning_data_set_for_nan_and_inf(X_calc, Y_calc, prune_zeros=True)
                    r2 = calculate_correlation(X_calc, Y_calc)
            
                    xticks = curr_axes.get_xticks()
                    yticks = curr_axes.get_yticks()
                    curr_axes.annotate(r'$r={:.2f}$'.format(r2), xy=(0.1,0.1), xycoords='axes fraction', size=20, color='red')
                    
        # only one plot:
        fig.savefig(path+'\{}'.format(filename), dpi=100, bbox_inches='tight')
        if save_XY_data == True:
            return XY
    else:
        cmap = mpl.cm.get_cmap('tab20b')
        for k,agebin in enumerate(agebins):
            # get the data in the current agebin
            max_age = agebin[1]
            min_age = agebin[0]
            # intervall from prune_to_relevant_data for ages is (min_age, max_age] so including max_age and excluding min_age
            if save_XY_data == True:
                XY = [[None for l in range(len(legus_mask_for_features))] for s in range(len(sitelle_mask_for_features))]
            for l in range(len(legus_mask_for_features)):
                for s in range(len(sitelle_mask_for_features)):
                    if (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) > 1):
                        curr_axes = axes[l, s]
                    elif (len(legus_mask_for_features) > 1 and len(sitelle_mask_for_features) == 1):
                        curr_axes = axes[l]
                    elif (len(legus_mask_for_features) == 1 and len(sitelle_mask_for_features) > 1):
                        curr_axes = axes[s]
                    else:
                        curr_axes = axes
                    already_used_HII_region = []
                    # create datasets Y are the HII-regions and X the clusters
                    # but this time if there are multiple clusters in one HII-region the features of the clusters will be summed up

                    X = []
                    Y = []
                    for i in a[:,0]:
                        if i not in already_used_HII_region:
                            Y.append(data_sitele[i,sitelle_mask_for_features[s]])
                            if legus_mask_for_features[l] in magnitude_features:
                                values = dire[i]
                                magsum = None
                                for n in range(0,len(values)):
                                    if (data_legus[values[n], 16] > min_age and data_legus[values[n], 16] <= max_age):
                                        magsum = data_legus[values[n], legus_mask_for_features[l]]
                                        for j in range(n,len(values)):
                                            if (data_legus[values[j], 16] > min_age and data_legus[values[j], 16] <= max_age):
                                                magsum = add_magnitudes_in_vega_system(magsum, data_legus[values[j], legus_mask_for_features[l]])
                                        break
                                    else:
                                        continue
                                if magsum == None:
                                    magsum = -np.inf
                                X.append(magsum)
                            else:
                                # here for other additive features
                                values = dire[i]
                                temp = 0
                                for j in range(len(values)):
                                    if (data_legus[values[j], 16] > min_age and data_legus[values[j], 16] <= max_age):
                                        temp += data_legus[values[j], legus_mask_for_features[l]]
                                if temp == 0:
                                    temp = -np.inf
                                X.append(temp)
                            already_used_HII_region.append(i)
                        else:
                            continue
                    X = np.asarray(X).astype(float)
                    Y = np.asarray(Y).astype(float)

                    if save_XY_data == True:
                        XY[s][l] = [X,Y]

                    # dict for the markers and colors indicating the different classes of the clusters and HII-regions
                    dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
                    dict_agebin = {1:cmap(4*k), 2:cmap(4*k+1), 3:cmap(4*k+2), 4:cmap(4*k+3)}

                    # only one plot:
                    for i in range(len(X)):
                        #curr_axes.scatter(X[i],Y[i], c=dict_agebin[data_sitele[np.asarray(a)[i,0],6]], alpha=0.7, marker=dict_HII[data_legus[np.asarray(a)[i,1],class_idx]])
                        curr_axes.scatter(X[i],Y[i], c=dict_agebin[data_legus[np.asarray(a)[i,1],class_idx]], alpha=0.7, marker=dict_HII[data_sitele[np.asarray(a)[i,0],6]])
                    curr_axes.set(xlabel=dir_legus[legus_mask_for_features[l]], ylabel=dir_sitelle[sitelle_mask_for_features[s]])
                    if xscales != None:
                        curr_axes.set_xscale(xscales[l])
                    if yscales != None:
                        curr_axes.set_yscale(yscales[s])
                    #curr_axes.ylabel(dir_sitelle[sitelle_mask_for_features[s]])

                    if l%frequ == 0 and s%frequ == 0:
                        if k == 0:
                            for c in classes_st:
                                curr_axes.scatter(np.inf,0, alpha=0.7, c='black',marker=dict_HII[c], s=100, label='HII-type {}'.format(c))

                        for i,c in enumerate(classes_le):
                            if i == 0:
                                curr_axes.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_agebin[c], s=12, label='Age: {}-{}Myr\nCluster-type 1'.format(int(min_age/1e6), int(max_age/1e6)))
                            else:
                                
                                curr_axes.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_agebin[c], s=12, label='Cluster-type {}'.format(c))
                        factor_for_legend = 1
                        if len(sitelle_mask_for_features) > 1:
                            factor_for_legend = 1.1
                        curr_axes.legend(markerscale=6., bbox_to_anchor=(0., 1.02, 1.*len(sitelle_mask_for_features)*factor_for_legend,20), loc=3, ncol=1+len(agebins), mode='expand', borderaxespad=0., fontsize=11, handletextpad=0.1)
                        #curr_axes.legend(markerscale=6., loc='best')

                    if set_xlim != None:
                        if set_xlim[l] != None:
                            curr_axes.set_xlim(set_xlim[l])
                    if set_ylim != None:
                        if set_ylim[s] != None:
                            curr_axes.set_ylim(set_ylim[s])
                    if print_corr_coef == True:
                        X_calc = X
                        Y_calc = Y
                        if xscales != None:
                            if xscales[l] == 'log':
                                X_calc = np.log10(X)
                        if yscales != None:
                            if yscales[s] == 'log':
                                Y_calc = np.log10(Y)
                        
                        X_calc, Y_calc = prune_machine_learning_data_set_for_nan_and_inf(X_calc, Y_calc, prune_zeros=True)
                        r2 = calculate_correlation(X_calc, Y_calc)
                
                        xticks = curr_axes.get_xticks()
                        yticks = curr_axes.get_yticks()
                        curr_axes.annotate(r'$r={:.2f}$'.format(r2), xy=(0.1,0.1), xycoords='axes fraction', size=20, color='red')

            
            # only one plot:
        fig.savefig(path+'\{}'.format(filename), dpi=100, bbox_inches='tight')
        if save_XY_data == True:
            return XY
        
# older version of function not necessarily compatible with a if a is created via create_matching_for_embedded
#def plot_extinction_correlation(a, sitelle_data, legus_data, used_sitelle_classes, used_legus_classes, sep = None, max_age=None, fit_line=True, path=r'D:\Uni\Bachelor_Arbeit\Plot', class_idx=33):
#    '''
#    a is the array of the matching containing the sitelle and legus array position resulting from sitelle_data and legus_data
#    
#    sitelle_data is the HII-Region data
#    legus_data is the Clusters coordinates
#    
#    used_legus_classes is an array containing the classes of the clusters which are used
#    used_sitelle_classes is an array containing the classes of the HII-Regions which are used
#    
#    sep is the same as used in a
#    max_age is the maximal considered age of the clusters
#    fit_line: if true it fits a lineare fit to the data with least-squares and ODR
#    '''
#    
#    data_sitele_red = sitelle_data
#    data_legus_red_class = legus_data
#    classes = used_legus_classes
#    classes_st = used_sitelle_classes
#    
#    print('number of matches',np.shape(data_legus_red_class[np.asarray(a)[:,1]][:,22])) #[22])
#
#    def linear(x, m, c):
#        return m*x+c
#
#    def linear_odr(beta, x):
#        return beta[0]*x+beta[1]
#
#    x = data_sitele_red[np.asarray(a)[:,0]][:,13]
#    y = data_legus_red_class[np.asarray(a)[:,1]][:,22]
#    x_err = data_sitele_red[np.asarray(a)[:,0]][:,14]
#    y_err = (data_legus_red_class[np.asarray(a)[:,1]][:,23] - data_legus_red_class[np.asarray(a)[:,1]][:,24])
#    plt.figure(figsize=(8,8))
#    plt.axis('equal')
#    plt.axis([-0.1, 1, -0.1, 1])
#    #plt.xlim(-0.1, 1)
#    #plt.ylim(-0.1, 1)
#    if fit_line == True:
#        x_indx = x.argsort()
#
#        x_fit = x[x_indx[::]]
#        y_fit = y[x_indx[::]]
#
#        mask_for_invalid = np.ones(len(x_fit))
#        # for line fit ignore 0,inf or nan
#        check_for = [x_fit,y_fit]
#        for i_fit in check_for:
#            idx_inf_nan = [i for i, arr in enumerate(i_fit) if not np.isfinite(arr).all()]
#            idx_0 = np.where(i_fit == 0)
#            
#            mask_for_invalid[idx_0] = 0
#            mask_for_invalid[idx_inf_nan] = 0
#
#        mask_for_invalid = mask_for_invalid.astype(bool)
#
#        x_fit = x_fit[mask_for_invalid]
#        x_err_fit = x_err[x_indx[::]]
#        x_err_fit = x_err_fit[mask_for_invalid]
#        y_fit = y_fit[mask_for_invalid]
#        y_err_fit = y_err[x_indx[::]]
#        y_err_fit = y_err_fit[mask_for_invalid]
#
#
#        # use least squares to fit fct
#        popt, pcov = curve_fit(linear, x_fit, y_fit, sigma=y_err_fit, absolute_sigma=True)
#
#        # use orthogonal distance regression 
#        data = RealData(x_fit, y_fit, x_err_fit, y_err_fit)
#        model = Model(linear_odr)
#        odr = ODR(data, model, [1,0])
#        odr.set_job(fit_type=0) # fit_type=0 corresponds to full ODR
#        output = odr.run()
#        print('output: ',output.beta)
#
#        # for least squares
#        p = np.linspace(-0.5,1.5,5)
#        m = popt[0]
#        dm = np.sqrt(pcov[0][0])
#        c = popt[1]
#        dc = np.sqrt(pcov[1][1])
#        plt.plot(p, linear(p, *popt), label='Linear fit with Least squares: y=mx+c\n$m={:.2f}\pm{:.2f}$; $c={:.2f}\pm{:.2f}$'.format(m, dm, c, dc))
#
#        # for ODR
#        m = output.beta[0]
#        dm = output.sd_beta[0]
#        c = output.beta[1]
#        dc = output.sd_beta[1]
#        
#        plt.plot(p, linear(p, *output.beta), label='Linear fit with ODR: y=mx+c\n$m={:.2f}\pm{:.2f}$; $c={:.2f}\pm{:.2f}$'.format(m, dm, c, dc))
#
#    dict_cluster = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
#    dict_cluster_2 = {1:'.', 2:'D', 3:'^', 4:'*'}
#    dict_HII = {1:'#d62728', 2:'b', 3:'#17becf', 4:'m'}
#
#    for c in classes:
#        plt.scatter(np.inf,0, alpha=0.7, c='black',marker=dict_cluster[c], s=100, label='Clustertype {}'.format(c))
#        
#    for c in classes_st:
#        plt.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_HII[c], s=12, label='HII-type {}'.format(c))
#
#    for i in range(len(x)):
#        plt.scatter(x[i],y[i], c=dict_HII[data_sitele_red[np.asarray(a)[i,0],6]], alpha=0.7, marker=dict_cluster[data_legus_red_class[np.asarray(a)[i,1],class_idx]])
#
#    plt.ylabel('E(B-V) LEGUS')
#    plt.xlabel('E(B-V) SITELLE')
#    plt.errorbar(x, y, xerr = x_err, yerr=y_err, linestyle='None', c='gray', alpha=0.7, elinewidth=0.5)
#    plt.title('Quality check with extinction')
#    plt.annotate('Datapruning:\nmax age: {:.2e} \ncluster classes taken into consideration: {}\nHII-Region classes taken into consideration: {}\nseperation boundary: {} a.u.'.format(max_age, ','.join(str(x) for x in classes), ','.join(str(x) for x in classes_st), sep), (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
#    plt.grid()
#    plt.legend(markerscale=6.)
#    filename = 'Extinction_correlation_sep_{}_maxage_{}_classes_{}_HIIclasses_{}_ODR'.format(sep, max_age, '_'.join(str(x) for x in classes), '_'.join(str(x) for x in classes_st))
#    plt.savefig(path+r'\{}.png'.format(filename), dpi=100, bbox_inches='tight')
#    return filename
    
def plot_extinction_correlation(a, sitelle_data, legus_data, used_sitelle_classes, used_legus_classes, sep = None, max_age=None, fit_line=True, path=r'D:\Uni\Bachelor_Arbeit\Plot', class_idx=33, fig_size=6, verbose=None):
    '''
    a is the array of the matching containing the sitelle and legus array position resulting from sitelle_data and legus_data
    
    sitelle_data is the HII-Region data
    legus_data is the Clusters coordinates
    
    used_legus_classes is an array containing the classes of the clusters which are used
    used_sitelle_classes is an array containing the classes of the HII-Regions which are used
    
    sep is the same as used in a
    max_age is the maximal considered age of the clusters
    fit_line: if true it fits a lineare fit to the data with least-squares and ODR
    '''
    
    data_sitele_red = sitelle_data
    data_legus_red_class = legus_data
    classes = used_legus_classes
    classes_st = used_sitelle_classes
    
    print('number of matches',np.shape(data_sitele_red[np.asarray(a)[:,0]][:,13])) #[22])

    def linear(x, m, c):
        return m*x+c

    def linear_odr(beta, x):
        return beta[0]*x+beta[1]

    x = data_sitele_red[np.asarray(a)[:,0]][:,13]
    
    y = data_legus_red_class[np.asarray(a)[:,1]][:,22]
    x_err = data_sitele_red[np.asarray(a)[:,0]][:,14]
    y_err = 0.5*(data_legus_red_class[np.asarray(a)[:,1]][:,23] - data_legus_red_class[np.asarray(a)[:,1]][:,24])
    
    # more intuitive way, but might be slower
    #y2 = np.asarray([data_legus_red_class[i,22] for i in np.asarray(a)[:,1]])
    #y_err2 = np.asarray([data_legus_red_class[i,23] - data_legus_red_class[i,24] for i in np.asarray(a)[:,1]])
    # uncomment above and below to test equality
    #np.testing.assert_array_equal(y,y2)
    #np.testing.assert_array_equal(y_err,y_err2)
    
    plt.figure(figsize=(fig_size, fig_size))
    plt.axis('equal')
    plt.axis([-0.1, 1, -0.1, 1])
    #plt.xlim(-0.1, 1)
    #plt.ylim(-0.1, 1)
    if fit_line == True:
        x_indx = x.argsort()

        x_fit = x[x_indx[::]]
        y_fit = y[x_indx[::]]

        mask_for_invalid = np.ones(len(x_fit))
        # for line fit ignore 0,inf or nan
        check_for = [x_fit,y_fit]
        for i_fit in check_for:
            idx_inf_nan = [i for i, arr in enumerate(i_fit) if not np.isfinite(arr).all()]
            idx_0 = np.where(i_fit == 0)
            
            mask_for_invalid[idx_0] = 0
            mask_for_invalid[idx_inf_nan] = 0

        mask_for_invalid = mask_for_invalid.astype(bool)

        x_fit = x_fit[mask_for_invalid]
        x_err_fit = x_err[x_indx[::]]
        x_err_fit = x_err_fit[mask_for_invalid]
        y_fit = y_fit[mask_for_invalid]
        y_err_fit = y_err[x_indx[::]]
        y_err_fit = y_err_fit[mask_for_invalid]


        # use least squares to fit fct
        popt, pcov = curve_fit(linear, x_fit, y_fit, sigma=y_err_fit, absolute_sigma=True)

        # use orthogonal distance regression 
        data = RealData(x_fit, y_fit, x_err_fit, y_err_fit)
        model = Model(linear_odr)
        odr = ODR(data, model, [1,0])
        odr.set_job(fit_type=0) # fit_type=0 corresponds to full ODR
        output = odr.run()
        print('output: ',output.beta)

        # for least squares
        p = np.linspace(-0.5,1.5,5)
        m = popt[0]
        dm = np.sqrt(pcov[0][0])
        c = popt[1]
        dc = np.sqrt(pcov[1][1])
        plt.plot(p, linear(p, *popt), label='Linear fit with Least squares: y=mx+c\n$m={:.2f}\pm{:.2f}$; $c={:.2f}\pm{:.2f}$'.format(m, dm, c, dc))

        # for ODR
        m = output.beta[0]
        dm = output.sd_beta[0]
        c = output.beta[1]
        dc = output.sd_beta[1]
        
        plt.plot(p, linear(p, *output.beta), label='Linear fit with ODR: y=mx+c\n$m={:.2f}\pm{:.2f}$; $c={:.2f}\pm{:.2f}$'.format(m, dm, c, dc))

    #dict_cluster = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
    #dict_cluster_2 = {1:'.', 2:'D', 3:'^', 4:'*'}
    #dict_HII = {1:'#d62728', 2:'b', 3:'#17becf', 4:'m'}
    # change the dicts so that the clusters are colorlabeld and the HII-Region are labbeld according to the markers
    cmap = mpl.cm.get_cmap('tab20b')
    dict_cluster = {1:cmap(0), 2:cmap(4), 3:cmap(8), 4:cmap(12), 0:cmap(16)}
    dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}

    for c in classes:
        plt.scatter(np.inf,0, alpha=0.7, c=dict_cluster[c], marker='.', s=100, label='Clustertype {}'.format(c))
        
    for c in classes_st:
        plt.scatter(np.inf,0, alpha=0.7, marker=dict_HII[c], c='black', s=12, label='HII-type {}'.format(c))

    for i in range(len(x)):
        plt.scatter(x[i],y[i], marker=dict_HII[data_sitele_red[np.asarray(a)[i,0],6]], alpha=0.7, c=dict_cluster[data_legus_red_class[np.asarray(a)[i,1],class_idx]])
        #plt.scatter(x[i],y[i], c=dict_HII[data_sitele_red[np.asarray(a)[i,0],6]], alpha=0.7, marker=dict_cluster[data_legus_red_class[np.asarray(a)[i,1],class_idx]])

    plt.ylabel('E(B-V) LEGUS')
    plt.xlabel('E(B-V) SITELLE')
    plt.errorbar(x, y, xerr = x_err, yerr=y_err, linestyle='None', c='black', alpha=0.8, elinewidth=0.8)
    #plt.title('Quality check with extinction')
    if verbose != None:
        plt.annotate('Datapruning:\nmax age: {:.2e} \ncluster classes taken into consideration: {}\nHII-Region classes taken into consideration: {}\nseperation boundary: {} a.u.'.format(max_age, ','.join(str(x) for x in classes), ','.join(str(x) for x in classes_st), sep), (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.grid()
    plt.legend(markerscale=6.)
    filename = 'Extinction_correlation_sep_{}_maxage_{}_classes_{}_HIIclasses_{}_ODR'.format(sep, max_age, '_'.join(str(x) for x in classes), '_'.join(str(x) for x in classes_st))
    plt.savefig(path+r'\{}.png'.format(filename), dpi=100, bbox_inches='tight')
    return filename
    
def make_frequency_plot_for_num_of_clusters_in_HII_regions(dic, path, name=r'\frequency_of_number_of_clusters_in_HII_regions.png', plot=False):
    '''
    creates a frequency plot of the number of clusters within the HII-regions
    
    dic is the dictionary as created by clusters_embedded_in_HII_region
    path is the path were the figure should be saved
    name is the file name
    if plot is True the plot will be shown else it will be saved in directory indecated with path
    '''
    
    clusters = [len(v) for v in dic.values()]
    bins = np.bincount(np.trim_zeros(np.sort(clusters)))
    width = 1
    x = np.arange(len(bins))
    plt.bar(x,bins, width, edgecolor='black', linewidth=0.2)
    plt.ylabel('frequency')
    plt.xlabel('number of clusters within one HII-region')
    
    if plot == True:
        plt.show()
    else:
        plt.savefig(path+name, dpi=200)
    

def mean_features_of_SITELLE_with_num_of_clusters(data_sitele_FOV, data_legus, feature, save=False, plot=False, path=None, save_positional_index=False, linear_fit=False, max_num_of_clusters_in_HII=24):
    '''
    returns a plot of the given SITELLE mean feature against the number of clusters
    
    '''
    dic = clusters_embedded_in_HII_region(data_sitele_FOV, data_legus, save_positional_index=save_positional_index)
    
    features = []
    features_error = []
    for i in range(max_num_of_clusters_in_HII):
        data_sitele_0, data_legus_0 = reduce_dic_to_n_embedded(i, dic, data_sitele_FOV, np.asarray(data_legus), save_positional_index=save_positional_index)
        #print('len data sitele {} cluster'.format(i), len(data_sitele_0))
        idx_inf_nan = [i for i, arr in enumerate(data_sitele_0[:,feature]) if not np.isfinite(arr).all()]
        mask_for_invalid = np.ones(len(data_sitele_0[:,feature]))
        mask_for_invalid[idx_inf_nan] = 0
        mask_for_invalid = mask_for_invalid.astype(bool)
        
        features.append(np.nanmean(data_sitele_0[:,feature][mask_for_invalid]))
        features_error.append(np.nanstd(data_sitele_0[:,feature][mask_for_invalid]))
        #print('mean {} sitelle {} cluster: '.format(dir_sitelle_featurename_and_id[feature],i), np.nanmean(data_sitele_0[:,feature][mask_for_invalid]), 'stddev: ', np.nanstd(data_sitele_0[:,feature][mask_for_invalid]))
    if linear_fit == True:
        def linear(x, m, c):
            return m*x+c
        x = np.argwhere(~np.isnan(features)).flatten()
        y = np.asarray(features)[x]
        dy = np.asarray(features_error)[x]
        nonzeros = np.nonzero(dy)
        popt, pcov = curve_fit(linear, x[nonzeros], y[nonzeros], sigma=dy[nonzeros], absolute_sigma=True)
        plt.plot(x, linear(x, *popt), label='$linear fit: y=mx+c$\n $m={:.1f}\pm{:.1f}$ $c={:.1f}\pm{:.1f}$'.format(popt[0], np.sqrt(pcov[0][0]), popt[1], np.sqrt(pcov[1][1])))
        plt.legend()
    dict_HII = {1:'#d62728', 2:'b', 3:'#17becf', 4:'m'}
    
    plt.scatter(np.arange(0,max_num_of_clusters_in_HII),features, marker='o')
    plt.errorbar(np.arange(0,max_num_of_clusters_in_HII),features, yerr= features_error, linestyle='None', c='gray', alpha=0.7, elinewidth=0.5)
    #plt.title('Extinction of HII-regions with increasing number of clusters')
    plt.xlabel('Number of clusters within HII-region')
    plt.ylabel('Mean {}'.format(dir_sitelle_featurename_and_id[feature]))
    if save == True:
        if path == None:
            path = r'D:/Uni/Bachelor_Arbeit/Plot/Bachelor_thesis_plots'
        name = dir_sitelle_featurename_and_id[feature]
        name = name.replace('/', '-')
        plt.savefig(path + r'/mean_{}_against_num_of_clusters.png'.format(name), dpi=200)
        plt.close()
    if plot == True:
        plt.show()
    
        
#################################################
# function not yet finished

def set_ticks_for_right_ascesion(locs, num_of_ticks='auto', plus_or_minus=-1):
    '''
    creates the labels and location for a figure which uses right ascesion as unit
    
    returns labels, locs
    the locs should be handed over as returned by plt.xticks()/plt.yticks()
    '''
    locs = np.asarray(locs)*plus_or_minus
    if num_of_ticks == 'auto':
        
        labels = []
        h_prev = None
        labels.append("")
        for i in range(1, len(locs)):
            # the hours
            h = int(locs[i]/15)
            # the rest r is given in degrees
            r = locs[i]-h*15
            # convert degrees into minutes 4m  = 1°
            minu = int(r*4)
            # the rest2 is given in degrees
            r2 = r - minu/4
            # calculate the seconds
            s = int(r2 * 60 * 4)
            #minu = int(np.around(minu*60, decimals=0))
            locs[i] = plus_or_minus*(h*15+minu/4+s/(60*4))
            # only print the hour when a new one is starting
            if h_prev != h:
                labels.append('{}'.format(h*-plus_or_minus)+'h'+'{}'.format(minu*-plus_or_minus)+'min'+'{}'.format(s*-plus_or_minus)+'s')
                h_prev = h
            else:
                labels.append('{}'.format(minu*-plus_or_minus)+'min'+'{}'.format(s*-plus_or_minus)+'s')
        labels.append("")
        return labels, locs
    else:
        num_of_ticks += 1
        labels = []
        locs_return = []
        loc_prev = None
        h_prev = None
        num_of_ticks = int(num_of_ticks)
        lower_boundary = locs[0]
        upper_boundary = locs[-1]
        diff = upper_boundary-lower_boundary
        seperation = diff/num_of_ticks
        for i in range(1,num_of_ticks):
            # the hours
            h = int((locs[0]+i*seperation)/15)
            # the rest r is given in degrees
            r = (locs[0]+i*seperation)-h*15
            # convert degrees into minutes 4m  = 1°
            minu = int(r*4)
            # the rest2 is given in degrees
            r2 = r - minu/4
            # calculate the seconds
            s = int(r2 * 60 * 4)
            #minu = int(np.around(minu*60, decimals=0))
            loc_curr = plus_or_minus*(h*15+minu/4+s/(60*4))
            
            if loc_prev != loc_curr:
                locs_return.append(loc_curr)
                loc_prev = loc_curr
                # only print the hour when a new one is starting
                if h_prev != h:
                    labels.append('{}'.format(h*-plus_or_minus)+'h'+'{}'.format(minu*-plus_or_minus)+'min'+'{}'.format(s*-plus_or_minus)+'s')
                    h_prev = h
                else:
                    labels.append('{}'.format(minu*-plus_or_minus)+'min'+'{}'.format(s*-plus_or_minus)+'s')
            else:
                continue
        return labels, locs_return
                
        
def set_ticks_for_declination(locs, num_of_ticks='auto'):
    '''
    creates the labels and location for a figure which uses declination as unit
    
    returns labels, locs
    the locs should be handed over as returned by plt.xticks()/plt.yticks()
    '''
    if num_of_ticks == 'auto':
        # hier noch bug dass manche labels doppelt eingezeichnet werden und diese dann 'dick' erscheinen
        labels = []
        locs_return = []
        loc_prev = None
        for i in range(len(locs)):
            # calculate degrees
            deg = int(locs[i])
            # with the rest calculate the degree minutes
            minu = locs[i]-deg
            minu = int(np.around(minu*60, decimals=0))
            loc_curr = deg+minu/60
            if loc_prev != loc_curr:
                locs_return.append(loc_curr)
                loc_prev = loc_curr
                labels.append('{}'.format(deg)+'°'+'{}'.format(minu))
            else:
                continue
        return labels, locs_return
    else:
        labels = []
        locs_return = []
        loc_prev = None
        num_of_ticks = int(num_of_ticks)
        lower_boundary = locs[0]
        upper_boundary = locs[-1]
        diff = upper_boundary-lower_boundary
        seperation = diff/num_of_ticks
        for i in range(num_of_ticks):
            # calculate degrees
            deg = int(locs[0]+i*seperation)
            # with the rest calculate the degree minutes
            minu = (locs[0]+i*seperation-deg)
            minu = int(np.around(minu*60, decimals=0))
            loc_curr = deg+minu/60
            if loc_prev != loc_curr:
                locs_return.append(loc_curr)
                loc_prev = loc_curr
                labels.append('{}'.format(deg)+'°'+'{}'.format(minu))
            else:
                continue
        return labels, locs_return

###########################################
# Test functions 
###########################################

def check_if_clusters_are_embedded(SITELLE, LEGUS, ab):
    r_asc_st = np.asarray(SITELLE)[:,1] # right ascesion of SITELLE data 
    decl_st = np.asarray(SITELLE)[:,2] # declination of SITELLE data
    size_st = np.asarray(SITELLE)[:,12] # this is given in pc
    size_st_deg = transform_pc_in_ra_dec(size_st) # transform the distance from pc into deg
    
    r_asc_le = np.asarray(LEGUS)[:,3] # right ascesion of LEGUS data
    decl_le = np.asarray(LEGUS)[:,4] # declination of LEGUS data
    
    features_sitele = np.array([r_asc_st,decl_st])
    features_legus = np.array([r_asc_le,decl_le])
    
    # as values in the dictionary the positional index will be saved of the legus data
    a = np.asarray(ab)
    for i in range(len(a)):
        if size_st_deg[a[i,0]] < calculate_angle(features_sitele[0,a[i,0]],features_sitele[1,a[i,0]],features_legus[0,a[i,1]],features_legus[1,a[i,1]]):
            print('Function Error: Legus data {} is not embedded in Sitelle data {}'.format(a[i,1],a[i,0]))
            
            
            
            
##############################
# latest added functions

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    source: https://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    '''
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def return_only_valid_numbers(a,b, return_zero = True):
    if return_zero == False:
        idx = [i for i, arr in enumerate(a) if arr==0]
        mask = np.ones(len(a))
        for i in idx:
            mask[i] = 0
        mask = mask.astype(bool)
        a = a[mask]
        b = b[mask]
        idx = [i for i, arr in enumerate(b) if arr==0]
        mask = np.ones(len(a))
        for i in idx:
            mask[i] = 0
        mask = mask.astype(bool)
        a = a[mask]
        b = b[mask]
    idx = [i for i, arr in enumerate(a) if not np.isfinite(arr)]
    mask = np.ones(len(a))
    for i in idx:
        mask[i] = 0
    mask = mask.astype(bool)
    a = a[mask]
    b = b[mask]
    idx = [i for i, arr in enumerate(b) if not np.isfinite(arr)]
    mask = np.ones(len(a))
    for i in idx:
        mask[i] = 0
    mask = mask.astype(bool)
    a = a[mask]
    b = b[mask]
    return a,b
    
    
def check_if_old_cluster_in_same_HII_as_young(dic, LEGUS, max_age=20e6, age_idx = 16):
    Return_LEGUS = deepcopy(LEGUS)
    delete_idx = []
    for v in dic.values():
        if len(v) > 1:
            cluster_below_max_age = False
            for cluster_id in v:
                # previously age 0 was allowed but age 0 corresponds mostly to not classifiable
                if Return_LEGUS[cluster_id, age_idx] <= max_age and Return_LEGUS[cluster_id, age_idx] > 0:
                    cluster_below_max_age = True
                    break
            
            if cluster_below_max_age == True:
                for cluster_id in v:
                    # remember the indices of the old clusters which are probably only by chance in the HII-region since there is a younger
                    if Return_LEGUS[cluster_id, age_idx] > max_age:
                        delete_idx.append(cluster_id)
                    else:
                        continue
            # no cluster below max_age
            else:
                continue
        # only cluster in HII-region -> age is irrelevant
        else:
            continue
            
    # just account every cluster once
    delete_idx = np.unique(delete_idx)
    # sort them in descending order
    delete_idx = np.sort(delete_idx)[::-1]
    for i in delete_idx:
        Return_LEGUS = np.delete(Return_LEGUS, i, 0)
    
    return Return_LEGUS
    
    
def creating_biplot(X, eigvec, data_sitelle_pruned, data_legus_pruned, pca_comp_1 = 0, pca_comp_2 = 1, scale=True, scale_feature_vec=1, indx_feature_sitelle=None, indx_feature_legus=None, dir_sitelle_featurename_and_id=dir_sitelle_featurename_and_id, dir_legus_featurename_and_id=dir_legus_featurename_and_id, figsize=10, fontsize=10, plot_legend=False):
    v = eigvec
    n = eigvec.shape[0]
    plt.figure(figsize=(figsize,figsize))
    # plot the first two PCA axis against each other
    # therefore multiply each instance of the dataset with the first and second pca axis (eigenvector from PCA)
    x, y = X.dot(v[pca_comp_1]), X.dot(v[pca_comp_2])
    cmap = mpl.cm.get_cmap('tab20c')
    dict_cluster = {1:cmap(0), 2:cmap(4), 3:cmap(8), 4:cmap(12), 0:cmap(16)}
    dict_HII = {0:'x', 1:'.', 2:'D', 3:'^', 4:'*'}
    
    if scale == True:
        scale_x = 1.0/(x.max()-x.min())
        scale_y = 1.0/(y.max()-y.min())
    else:
        scale_x, scale_y = 1, 1
    
    for i in range(len(X)):
        plt.scatter(x[i]*scale_x, y[i]*scale_y, c=dict_cluster[data_legus_pruned[i,33]], marker=dict_HII[data_sitelle_pruned[i,6]])
        # old style guide (HII-region's were delared with colors and star clusters with markers)
        #plt.scatter(x[i]*scale_x, y[i]*scale_y, marker=dict_cluster[data_legus_pruned[i,33]], c=dict_HII[data_sitelle_pruned[i,6]])
    v = np.transpose(v[:,:])
    for j in range(n):
        plt.arrow(0, 0, scale_feature_vec*v[j, pca_comp_1], scale_feature_vec*v[j, pca_comp_2], color='r', alpha=0.5)
        # write the next columns (of SITELLE features)
        if j < len(indx_feature_sitelle):
            plt.text(scale_feature_vec*v[j, pca_comp_1]*1.15, scale_feature_vec*v[j, pca_comp_2]*1.15, dir_sitelle_featurename_and_id[indx_feature_sitelle[j]], color='g', ha = 'center', va = 'center', fontsize=fontsize)
        
        # write the next coumns (of LEGUS features)
        else:
            k = j-len(indx_feature_sitelle)
            plt.text(scale_feature_vec*v[j, pca_comp_1]*1.15, scale_feature_vec*v[j, pca_comp_2]*1.15, dir_legus_featurename_and_id[indx_feature_legus[k]], color='g', ha = 'center', va = 'center', fontsize=fontsize)
    
    d = {0:'st', 1:'nd', 2:'rd'}
    if pca_comp_1 < 4:
        plt.xlabel('{}{} pca component [a.U.]'.format(pca_comp_1+1, d[pca_comp_1]))
    else:
        plt.xlabel('{}th pca component [a.U.]'.format(pca_comp_1+1))
    if pca_comp_2 < 4:
        plt.ylabel('{}{} pca component [a.U.]'.format(pca_comp_2+1, d[pca_comp_2]))
    else:
        plt.ylabel('{}th pca component [a.U.]'.format(pca_comp_2+1))
    if plot_legend == True:
        for c in range(4):
            plt.scatter(np.inf,0, alpha=0.7, c='black',marker=dict_HII[c], s=12, label='HII-type {}'.format(c))

        for c in range(3):
            plt.scatter(np.inf,0, alpha=0.7, marker='o', c=dict_cluster[c], s=12, label='Cluster-type {}'.format(c))

            

def create_super_feature_set(data_sitelle_red, data_legus_red, data_sitelle_for_super_features=None, data_legus_for_super_features=None, scale_data=True, explainable_variance=0.95, indx_feature_sitelle=[3, 4, 6, 12, 13, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42] , indx_feature_legus=[5, 7, 9, 11, 13,16, 19, 22, 33]):
    '''
    creates a SITELLE and LEGUS dataset with centralized and normalized data, whos features are a linear combination of the original features according to the pca
    the number of the output features is given through the explained_variance threshold
    
    data_sitelle_red is the input of the SITELLE data which the pca should be applied and therefore is responsible for the outcome of the used features
    data_legus_red is the input of the LEGUS data which the pca should be applied and therefore is responsible for the outcome of the used features
    data_sitelle_for_super_features is the data which will be returned with the applied feature transformation
    data_legus_for_super_features   -----------""-------------------------------------------------------------
    scale_data if the data should be centralized and normalized default: True
    explained_variance is a threshold which represents how much variance of the original data should be explained with the new super features
    indx_feature explains a mask which features should be used for the pca
    '''
    X = create_dataset_for_pca(data_sitelle_red, data_legus_red, indx_feature_sitelle, indx_feature_legus, scale_data=scale_data, print_used_features=False)
    X_sitelle = X[:,:len(indx_feature_sitelle)]
    X_legus = X[:,len(indx_feature_sitelle):]
    #X_sitelle = create_dataset_for_pca(data_sitelle_red, data_legus_red, indx_feature_sitelle, [], scale_data=scale_data, print_used_features=False)
    #X_legus = create_dataset_for_pca(data_sitelle_red, data_legus_red, [], indx_feature_legus, scale_data=scale_data, print_used_features=False)
    vecs = []
    # contains the number of principal components which are necessary to explain explainable_variance of the data variance
    idx_num_var = []
    Xs = [X_sitelle, X_legus]
    for X in Xs:
        components = np.shape(X)[1]
        # set up the PCA and give the number of components
        pca = PCA(n_components=components, svd_solver='full')
        # fit the PCA
        pca_sitelle_legus = pca.fit(X)
        
        # get the covariance matrix
        cov = pca.get_covariance()
                
        # calculate the eigenvalues w and eigenvectors v of the covariance matrix cov
        w, v = np.linalg.eig(cov)
        
        # np.linalg.eig returns the eigenvectors of eigenvalue i in coulumn i thats why we need to transpose the eigenvectors in order to get the eigenvectors in each row to sort them in the next step
        v = v.T
        
        # sort the eigenvalues w decreasingly and the corresponding eigenvectors
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[idx]
        
        # sometimes the np.linalg.eig() returns numbers with small complex values
        w = w.real
        v = v.real
        # calculate the percentages of variance
        p_of_var = [w[i]/np.sum(w) for i in range(len(w))]
        # calculate accumulative total variance
        acc_tot_var = np.asarray([np.sum(p_of_var[:i+1]) for i in range(len(p_of_var))])
        idx = np.argwhere(acc_tot_var >= explainable_variance)
        idx_num_var.append(idx[0][0]+1)
        vecs.append(v[:idx[0][0]+1])
    # check if the right number of eigenvectors was chosen
    assert(len(vecs[0])==idx_num_var[0])
    assert(len(vecs[1])==idx_num_var[1])
    
    if data_sitelle_for_super_features == None and data_legus_for_super_features == None:
        X_sitelle = X_sitelle.dot(vecs[0].T)
        X_legus = X_legus.dot(vecs[1].T)
        return X_sitelle, X_legus
    else:
        X = create_dataset_for_pca(data_sitelle_for_super_features, data_legus_for_super_features, indx_feature_sitelle, indx_feature_legus, scale_data=scale_data, print_used_features=False)
        X_sitelle = X[:,:len(indx_feature_sitelle)]
        X_legus = X[:,len(indx_feature_sitelle):]
                
        X_sitelle = X_sitelle.dot(vecs[0])
        X_legus = X_legus.dot(vecs[1])
        return X_sitelle, X_legus
        
        
def multiplot_for_super_features(data_sitelle, data_legus, dir_sitelle, dir_legus, filename=None, display=False ,path=r'D:\Uni\Bachelor_Arbeit\Plot', xscales=None, yscales=None, save_XY_data=False, classes_le=None, classes_st=None, frequ=2, set_xlim=None, set_ylim=None):
    # only one plot:
    fig, axes = plt.subplots(np.shape(data_legus)[1], np.shape(data_sitelle)[1], figsize=(8.0*np.shape(data_sitelle)[1], 6.0*np.shape(data_legus)[1])) #, subplot_kw=dict(polar=True)
    
    if save_XY_data == True:
        XY = [[None for l in range(len(legus_mask_for_features))] for s in range(len(sitelle_mask_for_features))]
    for l in range(np.shape(data_legus)[1]):
        for s in range(np.shape(data_sitelle)[1]):
                
            X = data_legus[:,l]
            Y = data_sitelle[:,s]
            
            if save_XY_data == True:
                XY[s][l] = [X,Y]
            
            # only one plot:
            for i in range(len(X)):
                axes[l, s].scatter(X[i],Y[i], alpha=0.7)
            axes[l, s].set(xlabel=dir_legus[l], ylabel=dir_sitelle[s])
            if xscales != None:
                axes[l, s].set_xscale(xscales[l])
            if yscales != None:
                axes[l, s].set_yscale(yscales[s])
            
            r2 = calculate_correlation(X, Y)
            
            xticks = axes[l, s].get_xticks()
            yticks = axes[l, s].get_yticks()
            axes[l, s].annotate(r'$r={:.2f}$'.format(r2), (np.min(X), np.min(Y)), size=20, color='red')
            '''
            if l%frequ == 0 and s%frequ == 0:
                for c in classes_le:
                    axes[l,s].scatter(np.inf,0, alpha=0.7, c='black',marker=dict_cluster[c], s=100, label='Clustertype {}'.format(c))
                    
                for c in classes_st:
                    axes[l,s].scatter(np.inf,0, alpha=0.7, marker='o', c=dict_HII[c], s=12, label='HII-type {}'.format(c))
                axes[l,s].legend(markerscale=6., loc='best')
            '''
            if set_xlim != None:
                if set_xlim[l] != None:
                    axes[l,s].set_xlim(set_xlim[l])
            if set_ylim != None:
                if set_ylim[s] != None:
                    axes[l,s].set_ylim(set_ylim[s])
                
    if filename != None:
        fig.savefig(path+'\{}'.format(filename), dpi=220, bbox_inches='tight')
    if display == True:
        fig.show()
    if save_XY_data == True:
        return XY