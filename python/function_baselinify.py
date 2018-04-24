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
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Generate the pupil with segments and spiders or read in from fits file

    # Get the coordinates of the central pixel of each segment
    ### In IDL done by 'eroding' the pupil - fill each center pixel with 1 and the rest with 0.
    ### Then use 'where' to find the 1s.
    ### seg_position is a [2, nb_seg] array that holds x and y position of each central pixel

    # Make distance list with distnaces between all of the central pixels among each other
    ### vec_list is a [nb_seg, nb_seg, 2] array

    # Nulling redundant vectors = setting redundant vectors in vec_list equal to zero

    # Extract the (number of) non redundant vectors: NR_distance_list

    # Select non redundant vectors
    ### NR_pairs_list is [NRP number, seg1, seg2] vector to hold non redundant vector information

    # Create NR_pairs_list_int and baseline_vec

    # Generate projection matrix

    # Get baseline_vec and Projection_Matrix