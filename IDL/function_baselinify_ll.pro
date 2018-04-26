function function_baselinify_ll, Projection_Matrix=Projection_Matrix, vec_list=vec_list, NR_pairs_list_int=NR_pairs_list_int

  ; computes the number of uniques pairs of segments in the ATLAST pupil
  ; input : none
  ; output :
  ; baseline_vec : number of unique baselines x 2 matrix, since each baseline has two coordinates.
  
  ;;;;;;;;;;;;;; make pupil without spider ;;;;;;;;;;;;;
  nb_seg = 7      ; number of segments *in diameter*
  size_seg = 100  ; size of array encircling one segment [px]
  size_gap = 1    ; size of gap between segments [px]
  
  ; Create full, circular pupil with hexagonal elements - ATLAST configuration
  pup_int = genere_segmented_pupil_ll(nb_seg=nb_seg, size_seg=size_seg, size_gap=size_gap)
  sz = (size(pup_int))[1]
  mini_seg = genere_hexa_pupil(NP=size_seg, Phi0=0)   ; returns a hexagon filled with ones embedded in an array filled by zeros

; full sized pupil with only one segment present in each of the 37 images (37 segments in ATLAST) 
tab_seg = make_array(sz, sz, 37, value=0.)

; Bottom line with four segments
for i=1,4 do begin &$
  tab_seg[*,*,i-1] = isolate_segment_ll2(pup=pup_int, hex=mini_seg, num_seg=i) &$
endfor

; Second line from bottom with five segments
for i=5,9 do begin &$
  tab_seg[*,*,i-1] = isolate_segment_ll2(pup=pup_int, hex=mini_seg, num_seg=i+1) &$
endfor

; Three middle lines with 18 segments in total; central obscuration is left black
for i=10,28 do begin &$
  tab_seg[*,*,i-1] = isolate_segment_ll2(pup=pup_int, hex=mini_seg, num_seg=i+2) &$
endfor

; Second line from top with five segments
for i=29,33 do begin &$
  tab_seg[*,*,i-1] = isolate_segment_ll2(pup=pup_int, hex=mini_seg, num_seg=i+3) &$
endfor

; Top line with four segments
for i=34,37 do begin &$
  tab_seg[*,*,i-1] = isolate_segment_ll2(pup=pup_int, hex=mini_seg, num_seg=i+4) &$
endfor

; Define the final pupil as combination of all the individual segments you picked
pup_final = total(tab_seg,3) - tab_seg(*,*,18)   ; sum of all the planes minus central obscuration - the IDL function total is the equivalent to np.sum()

;;;;;;;;;;;;;; make center list seg_position ;;;;;;;;;;;;;

center_label = erode(pup_final, mini_seg)   ; Get the central pixels of the full pupil as a map, central obscuration excluded
index = where(center_label)
nb_seg = (size(index))[1]   ; nubmer of segments, central obscuration excluded
seg_position = make_array(nb_seg, 2, value=0.)   ; This will hold the x and y pixel posiiton of each segment
sz = (size(pup_final))[1]   ; Size of final pupil

; Fill with x and y pixel positions for each segment; it looks this complicated because IDL is not great at it
for i=0, nb_seg-1 do begin &$
  seg_position[i,0] = index[i] MOD sz &$
  seg_position[i,1] = (index[i] - seg_position[i,0]) / sz &$
endfor

;;;;;;;;;;;;;; make distance list vec_list ;;;;;;;;;;;;;

vec_list = make_array(nb_seg, nb_seg, 2., value=0.)   ; Will hold relative positions of the centers between all pairs of segments

for i=0,nb_seg-1 do begin &$
  for j=0,nb_seg-1 do begin &$
    vec_list[i,j,*] = seg_position[i,*] - seg_position[j,*] &$
  endfor
endfor

;;;;;;;;;;;;;; nulling redundant vectors ;;;;;;;;;;;;;

vec_list2 = vec_list
vec_list_x = vec_list[*,*,0]
vec_list_y = vec_list[*,*,1]
vec_list_z = 0. * vec_list[*,*,1] ; useless, just makes the function "crossp" further down work, which is a crossproduct and needs three dimensions in order to work

; Loop over all pairs
for i=1,nb_seg*nb_seg-1 do begin &$
  for k=0,i-1 do begin &$
      if abs(norm([vec_list_x[i], vec_list_y[i], vec_list_z[i]]) - norm([vec_list_x[k], vec_list_y[k], vec_list_z[k]])) LT 4. then begin  &$  ; check length with norm, offset/margin of four pixels
        if norm(crossp([vec_list_x[i], vec_list_y[i], vec_list_z[i]], [vec_list_x[k],vec_list_y[k],vec_list_z[k]])) LT 1000. then begin  &$   ; check directions with crossproduct, offset/margin of a 1000 becaue vectors are huge
            
          ; All redundant distance pairs are set to zero
          vec_list[i MOD nb_seg, (i-(i MOD nb_seg))/nb_seg, *] = [0.,0.]      ; going back from total index to x and y index (complicated IDL way like for seg_position above)
         
        endif   
      endif
  endfor
endfor

;;;;;;;;;;;;;; number of non-redundant vectors ;;;;;;;;;;;;;

distance_list = vec_list[*,*,0]^2. + vec_list[*,*,1]^2.   ; Square the components to account for negative coordinates
index = where(distance_list NE 0.)                        ; Find indices where both coordinates are non-zero meaning it is a non-redundant pair (NRP)
NR_distance_list = distance_list[index]                   ; Put distances between NRPs into new (1D-)array
NR_pairs_nb = (size(NR_distance_list))[1]                 ; number of NRPs; ATLAST has 63 NRPs

;;;;;;;;;;;;;; selecting non-redundant vectors ;;;;;;;;;;;;;
; Assign each NRP its according segment pair (one segment pair that forms that NRP)

NR_pairs_list = make_array(NR_pairs_nb, 2, value=0.)   ; Create empty array to hold NRPs [NRP number, seg1, seg2]

; Loop over number of NRPs, taking into account the shift due to the central obscuration
for i=0,NR_pairs_nb-1 do begin
  NR_pairs_list[i,0] = index[i] MOD nb_seg                                              ; Assign first segment of the pair
  NR_pairs_list[i,1] = (index[i] - NR_pairs_list[i,0]) / nb_seg                         ; Assign second segment of the pair
  if NR_pairs_list[i,0] GE nb_seg/2 then NR_pairs_list[i,0] = NR_pairs_list[i,0] + 1    ; These two lines account for the offset of one segment because 
  if NR_pairs_list[i,1] GE nb_seg/2 then NR_pairs_list[i,1] = NR_pairs_list[i,1] + 1    ;   we numbered the segments including the central obscuration, but it doesn't really exist
endfor

; Create the actual array of NRPs [NRP number, seg1, seg2] that will be the output
NR_pairs_list_int = make_array(NR_pairs_nb, 2, value=0.)

; Loop over number of NRPs, now not taking into account the shift due to the central obscuration
for i=0,NR_pairs_nb-1 do begin
  NR_pairs_list_int[i,0] = index[i] MOD nb_seg
  NR_pairs_list_int[i,1] = (index[i] - NR_pairs_list_int[i,0]) / nb_seg
endfor

; Creating new vector baseline_vec with swapped segment assignments for NRPs
NR_pairs_list = round(NR_pairs_list)
NR_pairs_list_int = round(NR_pairs_list_int)
baseline_vec = 0. * NR_pairs_list
baseline_vec[*,1] = NR_pairs_list[*,0]
baseline_vec[*,0] = NR_pairs_list[*,1]

;;;;;;;;;;;;;;;;;; Generate projection matrix ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

vec_list3 = vec_list2   ; Relative positions of the centers between all pairs of segments

; Set relative posiitons between a segment and itself to zero
for i=0,nb_seg-1 do begin
  for j=0,nb_seg-1 do begin
    if i GE j then vec_list2[i,j,*] = [0.,0.]
  endfor
endfor

; Extract individual coordinates
vec_list_x = vec_list2[*,*,0]
vec_list_y = vec_list2[*,*,1]
vec_list_z = 0. * vec_list[*,*,1]   ; Holds non information, is needed to enable the function crossp below, the cross product

; Initialize projection matrix
Projection_Matrix_int = make_array(nb_seg,nb_seg, 3, value = 0.)   ; [NRP #, seg1, seg2]

; Loop through redundant (=all) segment pairs
for i=0,nb_seg*nb_seg-1 do begin
  ; Loop thourgh non-redundant segment pairs
  for k=0,NR_pairs_nb-1 do begin
    if abs(norm([vec_list_x[i], vec_list_y[i], vec_list_z[i]]) - norm([vec_list3[NR_pairs_list_int[k,0], NR_pairs_list_int[k,1],0], vec_list3[NR_pairs_list_int[k,0], NR_pairs_list_int[k,1],1], 0.*vec_list3[NR_pairs_list_int[k,0], NR_pairs_list_int[k,1],0]])) LT 4. then begin  ; check lengths with norm
      if norm(crossp([vec_list_x[i], vec_list_y[i], vec_list_z[i]], [vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0],vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],1],0.*vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0]])) LT 1000. then begin      ; check directions with cross product
        Projection_Matrix_int[i MOD nb_seg, (i-(i MOD nb_seg))/nb_seg, 0] = k+1                     ; NRP #
        Projection_Matrix_int[i MOD nb_seg, (i-(i MOD nb_seg))/nb_seg, 1] = NR_pairs_list[k,1]      ; segmet 1 of NRP
        Projection_Matrix_int[i MOD nb_seg, (i-(i MOD nb_seg))/nb_seg, 2] = NR_pairs_list[k,0]      ; segment 2 of NRP
      endif
    endif
  endfor
endfor

Projection_Matrix = make_array(nb_seg+1, nb_seg+1, 3, value = 0.)     ; New array for projection matrix

; Renumbering because of central obscuration
Projection_Matrix[0:17, 0:17, *] = Projection_Matrix_int[0:17, 0:17, *]
Projection_Matrix[19:36, 19:36, *] = Projection_Matrix_int[18:35, 18:35, *]
Projection_Matrix[0:17, 19:36, *] = Projection_Matrix_int[0:17, 18:35, *]
Projection_Matrix[19:36, 0:17, *] = Projection_Matrix_int[18:35, 0:17, *]

vec_list = vec_list3

return, baseline_vec

; REMEMBER TO SAVE THE OUTPUTS TO FITS FILES!!! (4 files)
; (baseline_vec)
; vec_list
; NR_pairs_list_int
; Projection_Matrix

;writefits, 'vec_list' + '.fits', vec_list
;writefits, 'NR_pairs_list_int' + '.fits', NR_pairs_list_int
;writefits, 'Projection_Matrix' + '.fits', Projection_Matrix


end
