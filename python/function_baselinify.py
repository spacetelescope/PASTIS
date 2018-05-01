"""
Translation of Lucie's IDL code function_baselinify_ll.pro

Computes the number of unique pairs in your pupil.
No inputs.

nb_seg is the number of segments WITHOUT the central obscuration.
nb_seg+1 is the number of segemts WITH the central obscurtiton.

Outputs:
    Projection_Matrix:  [nb_seg+1, nb_seg+1, 3] array
                        First plane: Projection_Matrix[i,j,0] = n means that the segment pair formed by the segments #i
                        and #j is equivalent to the pair #n of the non-redundant-pair basis.
                        Second and third planes: Projection_Matrix[i,j,1] = m and Projection_Matrix[i,j,2] = n means
                        that the segment pair formed by the segments #i and #j is equivalent to the non-redundant pair
                        formed by the segments #m and #n.
                        Only the FIRST plane is useful for later (Projection_Matrix[*,*,0]).
    vec_list:           [nb_seg, nb_seg, 2] array
                        Relative positions of the centers between all pairs of segments.
                        vec_list[i,j,*] = [x,y] means that the vector between the centers of the segments #i and #j has
                        the values x and y in pixels as coordinates.
    NR_pairs_list_int:  for JWST: [30, 2] array
                        List of non-redundant pairs.
                        NR_pairs_list_int[n,*] = [i,j] means that the pair formed by the segments #i and #j is the nth
                        non-redundant pair of the non-redundant-pair basis.
    Baseline_vec:       Restructured version of NR_pairs_list_int.
"""

import os
import time
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import poppy
import webbpsf


if __name__ == "__main__":

    # Some parameters
    NA = 18   # Number of apertures, without central obscuration

    #-# Generate the pupil with segments and spiders or read in from fits file

    # For now, use poppy directly
    jwst_pup = poppy.MultiHexagonAperture(rings=2)   # Create JWST pupil without spiders
    #jwst_pup.display()   # Show pupil
    #plt.show()

    #-# Get the coordinates of the central pixel of each segment
    ### In IDL done by 'eroding' the pupil - fill each center pixel with 1 and the rest with 0.
    ### Then use 'where' to find the 1s.
    ### seg_position is a [2, nb_seg] array that holds x and y position of each central pixel

    #test = jwst_pup.to_fits(npix=512)   # Create fits file from pupil file
    #pup_im = test[0].data   # Extract the data from the first ([0]) fits entry, which is the pupil image

    #one = poppy.MultiHexagonAperture(rings=1, center=True, segmentlist=[0])   # Create single segment for erosion
    #test2 = one.to_fits(npix=512)   # Convert to fits
    #single_seg = test2[0].data   # Convert to proper array

    # Cut out one single segment
    #cop = np.copy(pup_im)   # because for some reason pup_im gets overwritten somehow
    #mini_seg = cop[:103, 197:314]
    # Create meshgrid to be able to do grid operations
    #x, y = np.meshgrid(np.arange(np.size(mini_seg, 1)), np.arange(np.size(mini_seg, 0)))
    # Create blank data canvas
    #data = np.zeros([np.size(mini_seg,0), np.size(mini_seg,1)])
    #data1 = np.copy(data)
    # Make a line to cover segment overlap on bottom right
    #bottom_left = np.where(1.74*x+y > 256)
    #data[bottom_left] = 1
    #soso = np.copy(mini_seg)
    #soso[bottom_left] = 0
    #mini_seg[bottom_left] = 0

    #bottom_right = np.where(-1.74*x+y > 52)
    #mini_seg[bottom_right] = 0

    #plt.imshow(pup_im)
    #plt.show()

    # Scraping the stuff above. New try with poppy function.

    seg_position = np.zeros((NA, 2))
    for i in range(NA):
        seg_position[i, 1], seg_position[i, 0] = jwst_pup._hex_center(i)   # y, x = center position
        # The units of seg_posiiton are currently physical meters. I don't thing I need to do it in pixels as long
        # as it stays consistent.

    #-# Make distance list with distances between all of the central pixels among each other
    ### vec_list is a [nb_seg, nb_seg, 2] array

    vec_list = np.zeros((NA, NA, 2))
    for i in range(NA):
        for j in range(NA):
            vec_list[i,j,:] = seg_position[i,:] - seg_position[j,:]

    #-# Nulling redundant vectors = setting redundant vectors in vec_list equal to zero

    # Reshape vec_list array to one dimension so that we can implement the loop below
    longshape = vec_list.shape[0] * vec_list.shape[1]
    vec_flat = np.reshape(vec_list, (longshape, 2))

    # Create array that will hold the nulled coordinates
    vec_null = np.copy(vec_flat)

    ap = 0
    rp = 0
    for i in range(np.square(NA)):
        for j in range(i):

            # Some print statements for testing
            #print('i, j', i, j)
            #print('vec_flat[i,:]: ', vec_flat[i,:])
            #print('vec_flat[j,:]: ', vec_flat[j,:])
            #print('norm diff: ', np.abs(np.linalg.norm(vec_flat[i,:]) - np.linalg.norm(vec_flat[j,:])))
            #print('dir diff: ', np.linalg.norm(np.cross(vec_flat[i,:], vec_flat[j,:])))
            ap += 1

            # Check if length of two vectors is the same (within certain limits)
            if np.abs(np.linalg.norm(vec_flat[i,:]) - np.linalg.norm(vec_flat[j,:])) <= 1.e-10:

                # Check if direction of two vectors is the same (within certain limits)
                if np.linalg.norm(np.cross(vec_flat[i,:], vec_flat[j,:])) <= 1.e-10:

                    # Some print statements for testing
                    #print('i, j', i, j)
                    #print('vec_flat[i,:]: ', vec_flat[i, :])
                    #print('vec_flat[j,:]: ', vec_flat[j, :])
                    #print('norm diff: ', np.abs(np.linalg.norm(vec_flat[i, :]) - np.linalg.norm(vec_flat[j, :])))
                    #print('dir diff: ', np.linalg.norm(np.cross(vec_flat[i, :], vec_flat[j, :])))
                    rp += 1

                    vec_null[j,:] = [0, 0]
                    print(vec_null)

    # Reshape nulled array back into proper shape of vec_list
    vec_list_nulled = np.reshape(vec_null, (vec_list.shape[0], vec_list.shape[1], 2))

    #-# Extract the (number of) non redundant vectors: NR_distance_list

    # Create vector that holds distances between segments (instead of distance COORDINATES like in vec_list)
    distance_list = np.square(vec_list_nulled[:,:,0]) + np.square(vec_list_nulled[:,:,1])   # We use square distances so that we don't miss out on negative values
    nonzero = np.nonzero(distance_list)
    NR_distance_list = distance_list[nonzero]
    NR_pairs_nb = np.count_nonzero(distance_list)   # How many non-redundant (NR) pairs do we have?

    #-# Select non redundant vectors
    ### NR_pairs_list is [NRP number, seg1, seg2] vector to hold non redundant vector information

    #-# Create NR_pairs_list_int and baseline_vec

    #-# Generate projection matrix

    #-# Get baseline_vec and Projection_Matrix