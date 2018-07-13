"""
Translation of Lucie's IDL code function_baselinify_ll.pro

Computes the number of unique pairs in your pupil, currently configured for JWST.
No inputs.

nb_seg is the number of segments WITHOUT the central obscuration.
nb_seg+1 is the number of segemts WITH the central obscuration, but it shouldn't be needed anywhere.
Segments are numbered from 0 to nb_seg at its creation with Poppy, but the central segments gets discarded once we
are getting the coordinates of the segment centers.
NRPs are numbered from 1 to NR_pairs_nb (total number of NRPs).

Outputs:
    Projection_Matrix:  [nb_seg, nb_seg, 3] array
                        First plane: Projection_Matrix[i,j,0] = n means that the segment pair formed by the segments #i
                        and #j is equivalent to the pair #n of the non-redundant-pair basis.
                        Second and third planes: Projection_Matrix[i,j,1] = m and Projection_Matrix[i,j,2] = n means
                        that the segment pair formed by the segments #i and #j is equivalent to the non-redundant pair
                        formed by the segments #m and #n.
                        Only the FIRST plane is useful for later (Projection_Matrix[:,:,0]).
    vec_list:           [nb_seg, nb_seg, 2] array
                        Relative positions of the centers between all pairs of segments.
                        vec_list[i,j,:] = [x,y] means that the vector between the centers of the segments #i and #j has
                        the values x and y in pixels as coordinates.
    NR_pairs_list_int:  for JWST: [30, 2] array
                        List of non-redundant pairs.
                        NR_pairs_list_int[n,:] = [i,j] means that the pair formed by the segments #i and #j is the n-th
                        non-redundant pair of the non-redundant-pair basis. Numbered from 0 to NR_pairs-1
    Baseline_vec:       Restructured version of NR_pairs_list_int.
    JWST_aperture.pdf:  PDF display of the telescope pupil
    pupil.fits:         fits file of the telescope pupil, stored for later use
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import poppy

import python.util_pastis as util
from python.config import CONFIG_INI


if __name__ == "__main__":

    # Keep track of time
    start_time = time.time()   # runtime currently is around 2 seconds

    # Parameters
    outDir = os.path.join(CONFIG_INI.get('local', 'local_data_path'), 'segmentation')
    nb_seg = CONFIG_INI.getint('telescope', 'nb_subapertures')   # Number of apertures, without central obscuration
    flat_to_flat = CONFIG_INI.getfloat('telescope', 'flat_to_flat')
    wvl = CONFIG_INI.getfloat('filter', 'lambda')/1e9   # convert from nm to m
    flat_diam = CONFIG_INI.getfloat('telescope', 'flat_diameter')
    total_diam = CONFIG_INI.getfloat('telescope', 'diameter')

    # If subfolder "segmentation" doesn't exist yet, create it.
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    #-# Generate the pupil with segments and spiders or read in from fits file

    # Use poppy to create JWST aperture without spiders
    print('Creating and saving aperture')
    jwst_pup = poppy.MultiHexagonAperture(rings=2, flattoflat=flat_to_flat)   # Create JWST pupil without spiders
    jwst_pup.display(colorbar=False)   # Show pupil
    plt.title('JWST telescope pupil')
    # Number the segments
    for i in range(nb_seg+1):
        ycen, xcen = jwst_pup._hex_center(i)
        plt.annotate(str(i), size='x-large', xy=(xcen-0.1, ycen-0.1))   # -0.1 is for shifting the numbers closer to the segment centers
    # Save a PDF version of the pupil
    plt.savefig(os.path.join(outDir, 'JWST_aperture.pdf'))

    # Get pupil as fits image
    pupil_dir = jwst_pup.sample(wavelength=wvl, grid_size=flat_diam, return_scale=True)
    # If the image size is equivalent to the total diameter of the telescope, we don't have to worry about sampling later
    # But for the JWST case with poppy it makes such a small difference that I am skipping it for now
    util.write_fits(pupil_dir[0], os.path.join(outDir, 'pupil.fits'))

    #-# Get the coordinates of the central pixel of each segment
    seg_position = np.zeros((nb_seg, 2))   # holds x and y position of each central pixel
    for i in range(nb_seg+1):   # our pupil is still counting the central segment as seg 0, so we need to include it
                                # in the loop, however, we will just discard the values for the center
        if i == 0:     # Segment 0 is the central segment, which we want to skip and not put into seg_position
            continue   # Continues with the next iteration of the loop
        else:
            seg_position[i-1, 1], seg_position[i-1, 0] = jwst_pup._hex_center(i)   # y, x = center position
            # The units of seg_position are currently physical meters. I don't think I need to do it in pixels here, as
            # long as it stays consistent.

    #-# Make distance list with distances between all of the central pixels among each other
    vec_list = np.zeros((nb_seg, nb_seg, 2))
    for i in range(nb_seg):
        for j in range(nb_seg):
            vec_list[i,j,:] = seg_position[i,:] - seg_position[j,:]

    #-# Nulling redundant vectors = setting redundant vectors in vec_list equal to zero
    # This was really hard to figure out, so I simply went with exactly the same way like in IDL.

    # Reshape vec_list array to one dimension so that we can implement the loop below
    longshape = vec_list.shape[0] * vec_list.shape[1]
    vec_flat = np.reshape(vec_list, (longshape, 2))

    # Create array that will hold the nulled coordinates
    vec_null = np.copy(vec_flat)

    ap = 0
    rp = 0

    print('Nulling redundant segment pairs')
    for i in range(np.square(nb_seg)):
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

                    vec_null[i,:] = [0, 0]

    # Reshape nulled array back into proper shape of vec_list
    vec_list_nulled = np.reshape(vec_null, (vec_list.shape[0], vec_list.shape[1], 2))

    #-# Extract the (number of) non redundant vectors: NR_distance_list

    # Create vector that holds distances between segments (instead of distance COORDINATES like in vec_list)
    distance_list = np.square(vec_list_nulled[:,:,0]) + np.square(vec_list_nulled[:,:,1])   # We use square distances so that we don't miss out on negative values
    nonzero = np.nonzero(distance_list)
    NR_distance_list = distance_list[nonzero]
    NR_pairs_nb = np.count_nonzero(distance_list)   # Counting many non-redundant (NR) pairs we have

    #-# Select non redundant vectors
    # NR_pairs_list_int is a [NRP number, seg1, seg2] vector to hold non-redundant vector information.
    # NRPs are numbered from 0 to NR_pairs_nb-1, 0 designating a pair of a segment with itself.

    # Create the array of NRPs that will be the output
    NR_pairs_list_int = np.zeros((NR_pairs_nb, 2))   # not numbering center segment - this is an output

    # Loop over number of NRPs
    for i in range(NR_pairs_nb):
        NR_pairs_list_int[i,0] = nonzero[0][i]
        NR_pairs_list_int[i,1] = nonzero[1][i]


    # Create baseline_vec
    baseline_vec = np.copy(NR_pairs_list_int)
    baseline_vec[:,1] = NR_pairs_list_int[:,0]
    baseline_vec[:,0] = NR_pairs_list_int[:,1]

    NR_pairs_list_int = NR_pairs_list_int.astype(int)

    #-# Generate projection matrix

    # Set diagonal to zero (distance between a segment and itself will always be zero)
    # Although I am pretty sure they already are.
    vec_list2 = np.copy(vec_list)
    for i in range(nb_seg):
        for j in range(nb_seg):
            if i ==j:
                vec_list2[i,j,:] = [0,0]

    # Initialize the projection matrix
    Projection_Matrix_int = np.zeros((nb_seg, nb_seg, 3))

    # Reshape needed arrays so that we can loop over them easier
    vec2_long = vec_list2.shape[0] * vec_list2.shape[1]
    vec2_flat = np.reshape(vec_list2, (vec2_long, 2))

    matrix_long = Projection_Matrix_int.shape[0] * Projection_Matrix_int.shape[1]
    matrix_flat = np.reshape(Projection_Matrix_int, (matrix_long, 3))

    print('Creating projection matrix')
    for i in range(np.square(nb_seg)):
        for k in range(NR_pairs_nb):

            if np.abs(np.linalg.norm(vec2_flat[i, :]) - np.linalg.norm(vec_list[NR_pairs_list_int[k,0], NR_pairs_list_int[k,1], :])) <= 1.e-10:

                if np.linalg.norm(np.cross(vec2_flat[i, :], vec_list[NR_pairs_list_int[k,0], NR_pairs_list_int[k,1], :])) <= 1.e-10:

                    matrix_flat[i, 0] = k + 1
                    matrix_flat[i, 1] = NR_pairs_list_int[k,1]
                    matrix_flat[i, 2] = NR_pairs_list_int[k,0]

    # Reshape matrix back to normal form
    Projection_Matrix = np.reshape(matrix_flat, (Projection_Matrix_int.shape[0], Projection_Matrix_int.shape[1], 3))

    #-# Save the arrays: baseline_vec, vec_list, NR_pairs_list_int, Projection_Matrix
    util.write_fits(baseline_vec, os.path.join(outDir, 'baseline_vec.fits'), header=None, metadata=None)
    util.write_fits(vec_list, os.path.join(outDir, 'vec_list.fits'), header=None, metadata=None)
    util.write_fits(NR_pairs_list_int, os.path.join(outDir, 'NR_pairs_list_int.fits'), header=None, metadata=None)
    util.write_fits(Projection_Matrix, os.path.join(outDir, 'Projection_Matrix.fits'), header=None, metadata=None)

    print('All outputs saved')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for aperture_definition.py:', end_time - start_time, 'sec =', (end_time - start_time)/60, 'min')
