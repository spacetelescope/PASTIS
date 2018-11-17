"""
Translation of Lucie's IDL code function_baselinify_ll.pro

Computes the number of unique pairs in your pupil, currently configured for JWST.
No inputs.

nb_seg is the number of segments WITHOUT the central obscuration.
nb_seg+1 is the number of segments WITH the central obscuration, but it shouldn't be needed anywhere.
Segments are numbered from 0 to nb_seg at its creation with Poppy, but the central segment, labelled with 0, gets
discarded once we are getting the coordinates of the segment centers.
NRPs are numbered from 0 to NR_pairs_nb-1 (NR_pairs_nb = total number of NRPs).

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
    NR_pairs_list:  for JWST: [30, 2] array
                        List of non-redundant pairs.
                        NR_pairs_list[n,:] = [i,j] means that the pair formed by the segments #i and #j is the n-th
                        non-redundant pair of the non-redundant-pair basis. Numbered from n = 1 to NR_pairs, BUT when
                        they're indexed, they start at 0 since that's how Python does indexing. Careful about this.
    Baseline_vec:       Restructured version of NR_pairs_list.
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
    im_size = CONFIG_INI.getint('numerical', 'im_size_px')
    m_to_px = im_size/flat_diam      # for conversion from meters to pixels: 3 [m] = 3 * m_to_px [px]

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
    pupil_dir = jwst_pup.sample(wavelength=wvl, npix=1024, grid_size=flat_diam, return_scale=True)
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
            # Units are meters!!!

    # Save the segment center positions just in case we want to check them without running the code
    np.savetxt(os.path.join(outDir, 'seg_position.txt'), seg_position, fmt='%2.2f')
    # 18 segments, central segment (0) not included

    #-# Make distance list with distances between all of the segment centers among each other - in meters
    vec_list = np.zeros((nb_seg, nb_seg, 2))
    for i in range(nb_seg):
        for j in range(nb_seg):
            vec_list[i,j,:] = seg_position[i,:] - seg_position[j,:]
    # Save, but gotta save x and y coordinate separately because of the function I use for saving
    np.savetxt(os.path.join(outDir, 'vec_list_x.txt'), vec_list[:,:,0], fmt='%2.2f')   # x distance; units: meters
    np.savetxt(os.path.join(outDir, 'vec_list_y.txt'), vec_list[:,:,1], fmt='%2.2f')   # y distance; units: meters

    #-# Nulling redundant vectors = setting redundant vectors in vec_list equal to zero
    # This was really hard to figure out, so I simply went with exactly the same way like in IDL.

    # Reshape vec_list array to one dimension so that we can implement the loop below
    longshape = vec_list.shape[0] * vec_list.shape[1]
    vec_flat = np.reshape(vec_list, (longshape, 2))
    # Save for testing
    np.savetxt(os.path.join(outDir, 'vec_flat.txt'), vec_flat)

    # Create array that will hold the nulled coordinates
    vec_null = np.copy(vec_flat)

    ap = 0
    rp = 0

    print('Nulling redundant segment pairs')
    for i in range(longshape):
        for j in range(i):   # Since i starts at 0, the case with i=0 & j=0 never happens, we start at i=1 & j=0
                             # With this loop setup, in all cases we have i != k, which is the baseline between a
                             # segment with itself - which is not a valid baseline, so these vectors are already set
                             # to 0 in vec_null (they're already 0 in vec_flat).

            # Some print statements for testing
            #print('i, j', i, j)
            #print('vec_flat[i,:]: ', vec_flat[i,:])
            #print('vec_flat[j,:]: ', vec_flat[j,:])
            #print('norm diff: ', np.abs(np.linalg.norm(vec_flat[i,:]) - np.linalg.norm(vec_flat[j,:])))
            #print('dir diff: ', np.linalg.norm(np.cross(vec_flat[i,:], vec_flat[j,:])))
            ap += 1

            # Check if length of two vectors is the same (within numerical limits)
            if np.abs(np.linalg.norm(vec_flat[i,:]) - np.linalg.norm(vec_flat[j,:])) <= 1.e-10:

                # Check if direction of two vectors is the same (within numerical limits)
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
    # Save for testing
    np.savetxt(os.path.join(outDir, 'vec_list_nulled_x.txt'), vec_list_nulled[:, :, 0], fmt='%2.2f')
    np.savetxt(os.path.join(outDir, 'vec_list_nulled_y.txt'), vec_list_nulled[:, :, 1], fmt='%2.2f')

    #-# Extract the (number of) non redundant vectors: NR_distance_list

    # Create vector that holds distances between segments (instead of distance COORDINATES like in vec_list)
    distance_list = np.square(vec_list_nulled[:,:,0]) + np.square(vec_list_nulled[:,:,1])   # We use square distances so that we don't miss out on negative values
    nonzero = np.nonzero(distance_list)             # get indices of non-redundant segment pairs
    NR_distance_list = distance_list[nonzero]       # extract the list of distances between segments of NR pairs
    NR_pairs_nb = np.count_nonzero(distance_list)   # Counting how many non-redundant (NR) pairs we have
    # Save for testing
    np.savetxt(os.path.join(outDir, 'NR_distance_list.txt'), NR_distance_list, fmt='%2.2f')
    print('Number of non-redundant pairs: ' + str(NR_pairs_nb))

    #-# Select non redundant vectors
    # NR_pairs_list is a [NRP number, seg1, seg2] vector to hold non-redundant vector information.
    # NRPs are numbered from 1 to NR_pairs_nb, but Python indexing starts at 0!

    # Create the array of NRPs that will be the output
    NR_pairs_list = np.zeros((NR_pairs_nb, 2))   # NRP are numbered from 1 to NR_pairs_nb, as are the segments!

    # Loop over number of NRPs
    for i in range(NR_pairs_nb):
        # Since 'nonzero' holds the indices of segments, and Python indices start at 0, we have to add 1 to all the
        # 'segment names' in the array that tells us which NRP they form.
        NR_pairs_list[i,0] = nonzero[0][i] + 1
        NR_pairs_list[i,1] = nonzero[1][i] + 1
        # Again, NRP are numbered from 1 to NR_pairs_nb, and the segments are too!

    NR_pairs_list = NR_pairs_list.astype(int)
    # Save for testing
    np.savetxt(os.path.join(outDir, 'NR_pairs_list.txt'), NR_pairs_list, fmt='%i')

    #-# Generate projection matrix

    # Set diagonal to zero (distance between a segment and itself will always be zero)
    # Although I am pretty sure they already are. - yeah they are, vec_list is per definition a vector of distances
    # between all segments between each other, and the distance of a segment with itself is always zero.
    vec_list2 = np.copy(vec_list)
    for i in range(nb_seg):
        for j in range(nb_seg):
            if i ==j:
                vec_list2[i,j,:] = [0,0]
    # Save for testing
    np.savetxt(os.path.join(outDir, 'vec_list2_x.txt'), vec_list2[:, :, 0], fmt='%2.2f')
    np.savetxt(os.path.join(outDir, 'vec_list2_y.txt'), vec_list2[:, :, 1], fmt='%2.2f')

    # Initialize the projection matrix
    Projection_Matrix_int = np.zeros((nb_seg, nb_seg, 3))

    # Reshape arrays so that we can loop over them easier
    vec2_long = vec_list2.shape[0] * vec_list2.shape[1]
    vec2_flat = np.reshape(vec_list2, (vec2_long, 2))

    matrix_long = Projection_Matrix_int.shape[0] * Projection_Matrix_int.shape[1]
    matrix_flat = np.reshape(Projection_Matrix_int, (matrix_long, 3))

    print('Creating projection matrix')
    for i in range(np.square(nb_seg)):
        # Compare segment pair in i against all available NRPs.
        # Where it matches, record the NRP number in the matrix entry that corresponds to segments in i.

        for k in range(NR_pairs_nb):

            # Since the segment names (numbers) in NR_pairs_list assume we start numbering the segments at 1, we have to
            # subtract 1 every time when we need to convert a segment number into an index.
            # This means we write NR_pairs_list[k,0]-1 and NR_pairs_list[k,1]-1 .

            # Figure out which NRP a segment distance vector corresponds to - first by length.
            if np.abs(np.linalg.norm(vec2_flat[i, :]) - np.linalg.norm(vec_list[NR_pairs_list[k,0]-1, NR_pairs_list[k,1]-1, :])) <= 1.e-10:

                # Figure out which NRP a segment distance vector corresponds to - now by direction.
                if np.linalg.norm(np.cross(vec2_flat[i, :], vec_list[NR_pairs_list[k,0]-1, NR_pairs_list[k,1]-1, :])) <= 1.e-10:

                    matrix_flat[i, 0] = k + 1                       # Again: NRP start their numbering at 1
                    matrix_flat[i, 1] = NR_pairs_list[k,1] + 1      # and segments start their numbering at 1 too
                    matrix_flat[i, 2] = NR_pairs_list[k,0] + 1      # (see pupil image!).

    # Reshape matrix back to normal form
    Projection_Matrix = np.reshape(matrix_flat, (Projection_Matrix_int.shape[0], Projection_Matrix_int.shape[1], 3))

    # Convert the segment positions in vec_list from meters to pixels
    vec_list_px = vec_list * m_to_px

    #-# Save the arrays: vec_list, NR_pairs_list, Projection_Matrix
    util.write_fits(vec_list_px, os.path.join(outDir, 'vec_list.fits'), header=None, metadata=None)
    util.write_fits(NR_pairs_list, os.path.join(outDir, 'NR_pairs_list_int.fits'), header=None, metadata=None)
    util.write_fits(Projection_Matrix, os.path.join(outDir, 'Projection_Matrix.fits'), header=None, metadata=None)

    print('All outputs saved')

    # Tell us how long it took to finish.
    end_time = time.time()
    print('Runtime for aperture_definition.py:', end_time - start_time, 'sec =', (end_time - start_time)/60, 'min')
