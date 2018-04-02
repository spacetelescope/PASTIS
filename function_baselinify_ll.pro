function function_baselinify_ll, Projection_Matrix = Projection_Matrix, vec_list = vec_list, NR_pairs_list_int = NR_pairs_list_int

  ; computes the number of uniques pairs of segments in the ATLAST pupil
  ; input : none
  ; output :
  ; baseline_vec : number of unique baselines x 2 matrix, since each baseline has two coordinates.
  
  ;;;;;;;;;;;;;; make pupil without spider ;;;;;;;;;;;;;
  nb_seg = 7
  size_seg = 100
  size_gap = 1 
  
  pup_int=genere_segmented_pupil_ll(nb_seg = nb_seg, size_seg = size_seg, size_gap = size_gap)
  sz=(size(pup_int))[1]
  mini_seg=genere_hexa_pupil(NP=size_seg,Phi0=0)

tab_seg=make_array(sz,sz,37,value=0.)
for i= 1,4 do begin &$
  tab_seg[*,*,i-1]=isolate_segment_ll2(pup = pup_int, hex = mini_seg, num_seg=i)
endfor
for i= 5,9 do begin &$
  tab_seg[*,*,i-1]=isolate_segment_ll2(pup = pup_int, hex = mini_seg, num_seg=i+1)
endfor
for i= 10,28 do begin &$
  tab_seg[*,*,i-1]=isolate_segment_ll2(pup = pup_int, hex = mini_seg, num_seg=i+2)
endfor
for i= 29,33 do begin &$
  tab_seg[*,*,i-1]=isolate_segment_ll2(pup = pup_int, hex = mini_seg, num_seg=i+3)
endfor
for i= 34,37 do begin &$
  tab_seg[*,*,i-1]=isolate_segment_ll2(pup = pup_int, hex = mini_seg, num_seg=i+4)
endfor

pup_final=total(tab_seg,3)-tab_seg(*,*,18)

;;;;;;;;;;;;;; make center list ;;;;;;;;;;;;;

center_label=erode(pup_final,mini_seg)
index = where(center_label)
nb_seg=(size(index))[1]
seg_position=make_array(nb_seg,2,value=0.)
sz=(size(pup_final))[1]

for i=0,nb_seg-1 do begin
  seg_position[i,0]=index[i] MOD sz
  seg_position[i,1]=(index[i]-seg_position[i,0])/sz
endfor

;;;;;;;;;;;;;; make distance list ;;;;;;;;;;;;;

vec_list=make_array(nb_seg,nb_seg,2.,value=0.)

for i=0,nb_seg-1 do begin
  for j=0,nb_seg-1 do begin
    vec_list[i,j,*]=seg_position[i,*]-seg_position[j,*]
  endfor
endfor

;;;;;;;;;;;;;; nulling redundant vectors ;;;;;;;;;;;;;

vec_list2=vec_list
vec_list_x=vec_list[*,*,0]
vec_list_y=vec_list[*,*,1]
vec_list_z=0.*vec_list[*,*,1]

for i=1,nb_seg*nb_seg-1 do begin
  for k=0,i-1 do begin
      if abs(norm([vec_list_x[i],vec_list_y[i],vec_list_z[i]])-norm([vec_list_x[k],vec_list_y[k],vec_list_z[k]])) LT 4. then begin
        if norm(crossp([vec_list_x[i],vec_list_y[i],vec_list_z[i]],[vec_list_x[k],vec_list_y[k],vec_list_z[k]])) LT 1000. then vec_list[i MOD nb_seg,(i-(i MOD nb_seg))/nb_seg,*]=[0.,0.]
      endif
  endfor
endfor

;;;;;;;;;;;;;; number of non-redundant vectors ;;;;;;;;;;;;;

distance_list=vec_list[*,*,0]^2.+vec_list[*,*,1]^2.
index=where(distance_list NE 0.)
NR_distance_list=distance_list[index]
NR_pairs_nb=(size(NR_distance_list))[1]

;;;;;;;;;;;;;; selecting non-redundant vectors ;;;;;;;;;;;;;

NR_pairs_list=make_array(NR_pairs_nb,2,value=0.)

for i=0,NR_pairs_nb-1 do begin
  NR_pairs_list[i,0]=index[i] MOD nb_seg
  NR_pairs_list[i,1]=(index[i]-NR_pairs_list[i,0])/nb_seg
  if NR_pairs_list[i,0] GE nb_seg/2 then NR_pairs_list[i,0]=NR_pairs_list[i,0]+1
  if NR_pairs_list[i,1] GE nb_seg/2 then NR_pairs_list[i,1]=NR_pairs_list[i,1]+1
endfor

NR_pairs_list_int=make_array(NR_pairs_nb,2,value=0.)
for i=0,NR_pairs_nb-1 do begin
  NR_pairs_list_int[i,0]=index[i] MOD nb_seg
  NR_pairs_list_int[i,1]=(index[i]-NR_pairs_list_int[i,0])/nb_seg
endfor

NR_pairs_list=round(NR_pairs_list)
NR_pairs_list_int=round(NR_pairs_list_int)
baseline_vec = 0.*NR_pairs_list
baseline_vec[*,1] = NR_pairs_list[*,0]
baseline_vec[*,0] = NR_pairs_list[*,1]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

vec_list3=vec_list2

for i=0,nb_seg-1 do begin
  for j=0,nb_seg-1 do begin
    if i GE j then vec_list2[i,j,*]=[0.,0.]
  endfor
endfor

vec_list_x=vec_list2[*,*,0]
vec_list_y=vec_list2[*,*,1]
vec_list_z=0.*vec_list[*,*,1]

Projection_Matrix_int = make_array(nb_seg,nb_seg,3, value = 0.)

for i=0,nb_seg*nb_seg-1 do begin
  for k=0,NR_pairs_nb-1 do begin
    if abs(norm([vec_list_x[i],vec_list_y[i],vec_list_z[i]])-norm([vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0],vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],1],0.*vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0]])) LT 4. then begin
      if norm(crossp([vec_list_x[i],vec_list_y[i],vec_list_z[i]],[vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0],vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],1],0.*vec_list3[NR_pairs_list_int[k,0],NR_pairs_list_int[k,1],0]])) LT 1000. then begin
        Projection_Matrix_int[i MOD nb_seg,(i-(i MOD nb_seg))/nb_seg,0]=k+1
        Projection_Matrix_int[i MOD nb_seg,(i-(i MOD nb_seg))/nb_seg,1]=NR_pairs_list[k,1]
        Projection_Matrix_int[i MOD nb_seg,(i-(i MOD nb_seg))/nb_seg,2]=NR_pairs_list[k,0]
      endif
    endif
  endfor
endfor

Projection_Matrix = make_array(nb_seg+1,nb_seg+1,3, value = 0.)
Projection_Matrix[0:17,0:17,*]=Projection_Matrix_int[0:17,0:17,*]
Projection_Matrix[19:36,19:36,*]=Projection_Matrix_int[18:35,18:35,*]
Projection_Matrix[0:17,19:36,*]=Projection_Matrix_int[0:17,18:35,*]
Projection_Matrix[19:36,0:17,*]=Projection_Matrix_int[18:35,0:17,*]

vec_list = vec_list3

return, baseline_vec

end
