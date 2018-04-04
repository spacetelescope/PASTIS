PRO atlast_calibration

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; APLC ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
wfe_up_nm = 0.   ; wavefront error upstream [nm]
ae_nm = 0.
wfe_down_nm = 0. ; wavefront error downstream [nm]
lambda = 640; 1589. LL   ; wavelength [nm]

nm2rad = 2.*!dpi/lambda    ; conversion factor
wfe_up_rad = wfe_up_nm * nm2rad   ; wavefront error upstream [red]
ae_rad = ae_nm*nm2rad
wfe_down_rad = wfe_down_nm * nm2rad   ; wavefront error downstream [rad]

lyot = 0.9   ; size of Lyot stop, 0.9 times the entrance pupil (it is a bit smaller than entrance pupil)
ech = 2D
throughput = 0.44583701

save = 0
s_path = 'IMGS_BLACK_HOLE-test002/'

corono={type:'lyot', size:4.5D*2.*614./708., epsilon:1D}

apod = 0

meth = 'SA'   ; "semi-analytic"

; Plotting parameters
device, retain=2, decomposed=0
!p.color=0
!p.background = 255
!x.style = 1
!y.style = 1
plotsym, 0, /fill
defsysv, '!i', complex(0,1), 1
defsysv, '!di', dcomplex(0,1), 1
defsysv, '!rad2arcsec', 180./!dpi * 3600., 1
defsysv, '!shack', makeshack(240, 40, ob = .14)
tek_color
!p.charthick = 2.0
!p.charsize = 1.4
!p.thick = 1.0
!x.thick = 1.0
!y.thick = 1.0


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
largeur = fix(pupdiam*ech)

print, largeur


pup_meth = largeur
larg_meth = largeur
tab_np = [largeur, largeur]
diam_rec = float(ceil(largeur/ech))
IF diam_rec MOD 2 NE 0 THEN diam_rec = diam_rec + 1
larg_rec = diam_rec

; Load APLC
pup_up = readfits('/Users/ilaginja/Documents/Git/PASTIS/data/ApodSol_APLC_quart_atlastX025cobs1gap1_N0354r_FPM450M060_LSann20D70clear_Img097C_40DA100_BW10Nlam04fpres2_linbarhompre1.fits')

; Load Lyot stop
pup_do = readfits('/Users/ilaginja/Documents/Git/PASTIS/data/LS_full_atlast.fits')

pup_up = crop(pup_up, /m, nc=614)
pup_do = crop(pup_do, /m, nc=614)

pupdiam = (size(pup_up))[1]
spec_pup = [[[pup_up]], [[pup_do]]]   ; creating an array with the APLC in one plane and Lyot stop in the other
power_sp = 2.

phi_up = simu_phi_fourier(pup=pup_up, wfe_rad=wfe_up_rad, seed=5678, power_sp=power_sp)   ; Creating the aberration that can be taken as input in end-to-end simulator
ampl = simu_phi_fourier(pup= pup_up, wfe_rad=ae_rad, seed=1234, power_sp=power_sp)

phi_up = phi_up - complex(0.,1.) * ampl

phi_do = simu_phi_fourier(pup=pup_do, wfe_rad=wfe_down_rad, seed=5678, power_sp=power_sp)

psf_ref = 1
h_npc = 1

; End-to-end simulator of ONERA - here perfect case, no aberrations, for calibration
calc_psf_coro_phase, PHI_up = phi_up, $              ; input: upstream phase map
                     PHI_DO = phi_do, $              ; input: downstream map
                     spec_pup = spec_pup, $          ; optional input: binary transmission map
                     PUPDIAM_up = pupdiam, $         ; input: upstream pupil diameter
                     PUPDIAM_DO = pupdiam*lyot, $    ; input: downstream pupil diameter
                     ECH = ECH, $                    ; input: sampling
                     tab_np = tab_np, $
                     h_sc = psf_ref, $               ; optional outup: PSF without coronagraph
                     h_npc = h_npc, $                ; output: PSF with coronagraph
                     CORONO = corono,$               ; input: coronagraph to use
                     APOD = apod, $                  ; optional input: apodizer to use
                     METH = meth                     ; input: calculation method
; Creates a coronagraphic PSF from a downstream and an upstream phase map
           
p_ref = circminmaxmoy(psf_ref)
normp = max(p_ref)
p_ref = p_ref/normp
N = n_elements(p_ref)
taille_image = (size(h_npc))[1]

; debut LL DARK HOLE - define dark hole by subtracting two circles of different radius
polaire2, rt = 10.*ech*614./708., largeur=taille_image, masque=masko, /double
polaire2, rt = 4.*ech*614./708., largeur=taille_image, masque=maski, /double
dh_area = abs(masko-maski)

norm = max(psf_ref)   ; normalization factoro for later

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; ERROR BUDGET ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

tic = systime(1)    ; keep track of time

; construction des matrices de coefficients de Zernike
nb_zer = 6
nb_seg = 37

;APLC_Mean_DH = make_array(37, value=0.)

nb_iterations = 1.
nb_steps = 37.
contrastAPLC_vec = make_array(nb_steps, value=0.)   ; for contrast of APLC end-to-end simulations
contrastAM_vec = make_array(nb_steps, value=0.)     ; for contrast of AM = Analytical Model = PASTIS

;cd,'C:/Users/lleboulleux/Desktop'
;M = readfits('Moyennes_Matrix.fits')
nb_seg = 37

contrastAPLC_vec_int = make_array(37, value=0.)
contrastAM_vec_int = make_array(37, value=0.)
  
  for inc = 0,36 do begin
    
    print, 'for inc = ' + inc
    
    ; Put aberration on only one segment
    A = 0.*randomu(seed, nb_seg, 1)
    A[inc] = 1.
    A[18] = 0.
    
    ; Define phase mask of single aberrated segment on total pupil
    isolated_coef_zern = make_array(37, nb_zer, value=0.)   ; local Zernikes
    isolated_coef_zern[*,2] = 2.*!PI*A/lambda               ; [*,0] is piston, [*,1] is tip, [*,2] is tilt and so on - you pick here what kind of aberration you want on your one segment
    global_coef_zern = make_array(1, 5, value=0.)   ; global Zernikes over totla pupil; for us always zero, but it's needed as input for funcitno below
    phi_ab = make_phi_atlast_ll(isolated_coef_zern=isolated_coef_zern, global_coef_zern=global_coef_zern)   ; make phase mask of total pupil where only one segment is aberrated
    phi_ab = crop(phi_ab, /m, nc=614)
    
    pup = make_pup_atlast_ll(tab_seg=tab_seg, mini_seg=mini_seg)   ; create ATLAST pupil
    pup = crop(pup, /m, nc=614)
    
    ; End-to-end ismulator of ONERA - here with aberrated segment
    calc_psf_coro_phase, PHI_up = phi_ab, $
      PHI_DO = phi_do, $
      spec_pup = spec_pup, $
      PUPDIAM_up = pupdiam, $
      PUPDIAM_DO = pupdiam*lyot, $
      ECH = ECH, $
      tab_np = tab_np, $
      h_sc = psf_ref, $
      h_npc = h_npc, $
      CORONO = corono,$
      APOD = apod, $
      METH = meth
      
    h_npc = h_npc/normp   ; h_nps is image at the end of sim with coronagraph
    h_npc_zoom3 = crop(h_npc, /m, nc=40)
    dh_area_zoom = crop(dh_area, /m, nc=40)
    ImAPLC = h_npc_zoom3 * dh_area_zoom
    contrastAPLC_vec_int[inc] = mean(h_npc_zoom3[where(dh_area_zoom)])   ; get sim contrast in DH
    
    ImAM = analytical_model(zernike_pol=2, coef=A)   ; image in dark hole generated by AM (=PASTIS)
    contrastAM_vec_int[inc] = mean(ImAM[where(dh_area_zoom)])   ; get PASTIS contrast in DH

  endfor

loadct, 3, /silent
aff, alog(ImAPLC >1e-12), 0, z=8

curve = plot(findgen(37), contrastAPLC_vec_int, xrange=[0.,36.], yrange=[1.e-12,1.e-7], color='red', thick=2, xthick=2, ythick=2, /current, /ylog)
curve = plot(findgen(37), contrastAM_vec_int,xrange=[0.,36.], yrange=[1.e-12,1.e-7], color='green', thick=2, xthick=2, ythick=2, /current, /ylog)
;curve = plot(var_vec, max_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='blue', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, contrastAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='red', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, minAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='green', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, maxAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='blue', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)

;cd, 'C:/Users/lleboulleux/Desktop'
;curve.save, 'Comparison_APLC_AM_piston1nm_Random2.png'

; Generating the calibration vector
Calibration_Astig0 = contrastAPLC_vec_int
Calibration_Astig0[18] = 0.   ; central obscuration
Calibration_Astig0 = Calibration_Astig0/contrastAM_vec_int    ; C_0 is so small here that Lucie removed it, but for JWSt it is actually big, so I will need to take it into account
Calibration_Astig0[18] = 0.

; REMEMBER TO SAVE OUTPUTS!!! - one calibration vector
;writefits, 'Calibration_Tilt.fits',Calibration_Astig0

stop

end
