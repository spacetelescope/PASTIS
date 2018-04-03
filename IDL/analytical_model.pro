function analytical_model, zernike_pol=zernike_pol, coef=coef

; This function applies the analytical model in case of one single Zernike polynomial on the segments.
; OUTPUT: Image in the dark hole only
; INPUTS: zernike: scalar referencing the Zernike: 0 for piston, 1 for tip, 2 for tilt, 3 for defocus, 4 for 45°-astigmatism, 5 for 0°-astigmatism.
;         coef: 37 x 1 vector of Zernike coefficients applied on all the segments of the atlast pupil. The 18-coef must be left blank, because of central obstruction.

;;;;;;;;;;;;;;;;;;; PARAMETERS ;;;;;;;;;;;;;;;;;;;
lambda = 640
s_path = 'IMGS_BLACK_HOLE-test002/'
flux = 1e9
ech = 2D
n_seg = 37
nb_seg = 7
size_seg = 100
size_gap = 1
marge = 2.0
alpha = nb_seg * marge
y_max = alpha/2-1
x_max_init = alpha/2-1
np = size_seg
gap = size_gap
np_tot = (NP+gap) * (alpha / marge)
pupdiam = np_tot
largeur = 614.*ech
D_telescope = 16.8e9 ; en nm
Size_Telescope = D_telescope/614. ; in nm per pixel, in pupil plane (size of one pixel in pupil in nm)
Size_Pixel = 18.e3 ; in nm per pix, in detector plane (focal plane)
Focal_length = 2.*Size_Pixel*D_Telescope/lambda ; since at Shannon sampling, size_pixel = lambda*f/2D
Wave_number = 2.*!Pi/lambda ; nombre d'onde
PixelSquare2Rad = (Size_Telescope*Size_Pixel*Wave_number/Focal_Length) ; 528 ; (495./614.)*
;PixelSquare2Rad = (614./785)*(Size_Telescope*Size_Pixel*Wave_number/Focal_Length)

;;;;;;;;;;;;;;;; DARK HOLE SHAPE ;;;;;;;;;;;;;;;;;
polaire2, rt=10.*ech*614./708., largeur=largeur, /entre4, masque=masko, /double
polaire2, rt=4.*ech*614./708., largeur=largeur, /entre4, masque=maski, /double
dh_area = abs(masko-maski)

;;;;;;;;;;;;;;;; MEAN SUBTRACTION ;;;;;;;;;;;;;;;;
coef = coef*2.*!PI/lambda

if zernike_pol EQ 0 then begin
  mean_coef = mean(coef)
  coef = coef - mean_coef
endif

;;;;;;;;;;;;;;;;;;;; GENERIC SEGMENT SHAPE ;;;;;;;;;;;;;;;;;;;;
atlast = make_pup_atlast_ll(tab_seg = tab_seg, mini_seg = mini_seg)
mini_seg = mini_seg/sqrt(36.*total(mini_seg))

;;;;;;;;;;;;;;;;;;; PROJECTION MATRICES ;;;;;;;;;;;;;;;;;;;
;Baseline_vec = function_baselinify_ll(Projection_Matrix = Projection_Matrix, vec_list = vec_list, NR_pairs_list_int = NR_pairs_list_int)
;writefits, 'Baseline_vec.fits', Baseline_vec
;writefits, 'Projection_Matrix.fits', Projection_Matrix
;writefits, 'vec_list.fits', vec_list
;writefits, 'NR_pairs_list_int.fits', NR_pairs_list_int
;;;;;;;;;;;;;; Instead, we just import them: ;;;;;;;;;;;;;;;;
cd, '/astro/opticslab1/Users/Lucie/2017/Inputs'
Baseline_vec = readfits('Baseline_vec.fits')
Projection_Matrix = readfits('Projection_Matrix.fits')
vec_list = readfits('vec_list.fits')
NR_pairs_list_int = readfits('NR_pairs_list_int.fits')
NR_pairs_nb = (size(Baseline_vec))[1]

;;;;;;;;;;;;;;;;;;; CALIBRATION ;;;;;;;;;;;;;;;;;;;
cd, 'C:/Users/lleboulleux/Desktop'

if zernike_pol EQ 0 then begin
  APLC_normalisation = readfits('APLC_normalisation.fits')
  Model_normalisation = readfits('Model_normalisation.fits')
  ck = sqrt(APLC_normalisation/Model_normalisation)
endif
if zernike_pol EQ 1 then ck=sqrt(readfits('Calibration_Tip.fits'))
if zernike_pol EQ 2 then ck=sqrt(readfits('Calibration_Tilt.fits'))
if zernike_pol EQ 3 then ck=sqrt(readfits('Calibration_Focus.fits'))
if zernike_pol EQ 4 then ck=sqrt(readfits('Calibration_Astig45.fits'))
if zernike_pol EQ 5 then ck=sqrt(readfits('Calibration_Astig0.fits'))

coef = coef * ck

;;;;;;;;;;;;;;;;;;; GENERIC COEFFICIENTS Aq ;;;;;;;;;;;;;;;;;;;
Generic_Coef = make_array(NR_pairs_nb,value = 0.)
for q=0,NR_pairs_nb-1 do begin
  for i=0,n_seg-1 do begin
    for j=i+1,n_seg-1 do begin
      if Projection_Matrix[i,j,0] EQ q+1 then Generic_Coef[q] = Generic_Coef[q]+(coef[i]*coef[j])
    endfor
  endfor
endfor

;;;;;;;;;;;;;;;;;;; CONSTANT SUM AND COSINE SUM ;;;;;;;;;;;;;;;;;;;
tab_i = (dindgen(largeur, largeur) mod largeur) - largeur/2. + 0.5
tab_j = transpose(tab_i)
cos_u_mat = dblarr(largeur, largeur, 63)
PixelSquare2Rad = double(PixelSquare2Rad)
for q=0,NR_pairs_nb-1 do begin
  cos_u_mat[*,*,q] = cos(PixelSquare2Rad*(vec_list[NR_pairs_list_int[q,0], NR_pairs_list_int[q,1],0]*tab_i) + PixelSquare2Rad*(vec_list[NR_pairs_list_int[q,0], NR_pairs_list_int[q,1],1]*tab_j))
endfor

Somme1 = total(coef^2)
Somme2 = make_array(largeur, largeur, value=0.)

for q=0,NR_pairs_nb-1 do begin
  Somme2 = Somme2 + Generic_Coef[q] * Cos_U_mat[*,*,q]
endfor

;;;;;;;;;;;;;;;;;;; LOCAL ZERNIKE ;;;;;;;;;;;;;;;;;;;
size_seg = 100
sz=(size(mini_seg))[1]
isolated_zern = calc_mode_zernike(nbmodes=6, pupdiam=size_seg, largeur=sz)
if zernike_pol EQ 0 then Zer = mini_seg
if zernike_pol EQ 1 then Zer = mini_seg*carre(isolated_zern[0,*])
if zernike_pol EQ 2 then Zer = mini_seg*carre(isolated_zern[1,*])
if zernike_pol EQ 3 then Zer = mini_seg*carre(isolated_zern[2,*])
if zernike_pol EQ 4 then Zer = mini_seg*carre(isolated_zern[3,*])
if zernike_pol EQ 5 then Zer = mini_seg*carre(isolated_zern[4,*])

;;;;;;;;;;;;;;;;;;; FINAL IMAGE ;;;;;;;;;;;;;;;;;;;
TF_seg = (abs(mft(Zer, param=100, dim_tf=largeur, double=double))^2.) * (Somme1+2.*Somme2)
TF_seg_zoom = crop(TF_seg,/m,nc=40)
dh_area_zoom = crop(dh_area,/m,nc=40)
DH_PSF = dh_area_zoom * TF_seg_zoom

return, DH_PSF

end