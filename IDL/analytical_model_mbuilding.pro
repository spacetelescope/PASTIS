PRO analytical_model_Mbuilding

; INPUT:
; Piston_vec: 37 x 1 Vector of piston coefficients applied on all the segments of the atlast pupil. The 18-coef must be left blank, because of central obstruction.

;;;;;;;;;;;;;;;;;;;;;;; variables (input) ;;;;;;;;;;;;;;;

;Model_Mean_DH=make_array(37, value=0.)

;For inc=0,36 do begin

; Iva: just a test 
Piston_vec = make_array(37, value=0.)
;Piston_vec[15] = 1. ; nm
;Piston_vec[10] = 1. ; nm
Piston_vec = randomu(seed, 37, 1)

;;;;;;;;;;;;;;;;;;;;;;; parameters (do not change them) ;;;;;;;;;;;;;;;;;;;;;;;;;

device, retain=2, decomposed=0
!p.color=0
!p.background = 255
!x.style = 1
!y.style = 1
plotsym, 0, /fill
defsysv, '!i', complex(0,1), 1
defsysv, '!di', dcomplex(0,1), 1
defsysv, '!rad2arcsec', 180./!dpi * 3600., 1
defsysv, '!shack', makeshack(240, 40, ob=.14)
tek_color
!p.charthick = 2.0
!p.charsize = 1.4
!p.thick = 1.0
!x.thick = 1.0
!y.thick = 1.0

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; Basis of baselines bq ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

tic = systime(1)
DH_PSF = analytical_model(zernike_pol=1, coef=Piston_vec)
toc = systime(1)
print, toc-tic

; Create dark hole; 708... total size of image (D), 614... 
ech = 2D
largeur = 614.*ech
polaire2, rt=10.*ech*614./708., largeur=largeur, /entre4, masque=masko, /double
polaire2, rt=4.*ech*614./708., largeur=largeur, /entre4, masque=maski, /double
dh_area = abs(masko-maski)
dh_area_zoom = crop(dh_area, /m, nc=40)

loadct, 3
aff, alog(DH_PSF>1e-10), 0, z=4

print, 'MOYENNE DANS DH AVEC MODELE ANALYTIQUE'
print, mean(DH_PSF[where(dh_area_zoom)])

; Model_Mean_DH[inc]=mean(TF_seg_zoom[where(dh_area_zoom)])

; Endfor

;cd,'C:/Users/lleboulleux/Desktop'
;writefits, 'Model_test_focus.fits', Model_Mean_DH

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;; PARTIE MATRICIELLE ;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

A = Piston_vec
lambda = 640.
A = A*2.*!PI/lambda
mean_A = mean(A)
A = A - mean_A   ; gettin rid of general offset, global PSF wouldn't affect PSF at all; here it effects the pupil - so we have to get rid of global piston






M = make_array(37, 37, value=0.)

;;;;;;;;;;;;; MATRIX M BUILDING ;;;;;;;;;;;;;;;;
 
n_seg = 37

for i=0,n_seg-1 do begin
  for j=0,n_seg-1 do begin
    print, 'STEP'
    print, i
    print, j
    ; Putting 1nm only on segments i and j
    tempA = 0.*A
    tempA[i] = 1.
    tempA[j] = 1.
    tempIm = analytical_model(zernike_pol=2, coef=tempA)   ; returns the dark hole PSF
    M[i,j] = mean(tempIm[where(dh_area_zoom)])
  endfor
endfor

N = M

; Filling the off-axis elements
n_seg = 37
for i=0,n_seg-1 do begin
  for j=0,n_seg-1 do begin
    if i NE j then M[i,j] = (N[i,j] - N[i,i] - N[j,j]) / 2.
  endfor
endfor

; WRITE MATRIX TO FILE
;writefits, 'Moyennes_Matrix_Tilt.fits', M

stop

print, 'MOYENNE DANS DH AVEC MODELE ANALYTIQUE'
print, mean(DH_PSF[where(dh_area_zoom)])
print, 'MOYENNE DANS DH AVEC MATRICES'
print, A##M##transpose(A)   ; ## is matrix multiplication
print, mean(DH_PSF[where(dh_area_zoom)])/(A##M##transpose(A))

stop

end
