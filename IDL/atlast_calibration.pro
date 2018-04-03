PRO atlast_calibration

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; APLC ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
wfe_up_nm = 0.
ae_nm = 0.
wfe_down_nm = 0.
lambda = 640; 1589. LL

nm2rad = 2.*!dpi/lambda
wfe_up_rad = wfe_up_nm*nm2rad
ae_rad = ae_nm*nm2rad
wfe_down_rad = wfe_down_nm*nm2rad

lyot = 0.9   ;size of Lyot stop, 0.9 tmies the entrance pupil (it is a bit smaller)
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

pup_up = readfits('C:/Users/lleboulleux/Desktop/Budget d erreur/APLC designs/ATLAST_4à10/ApodSol_APLC_quart_atlastX025cobs1gap1_N0354r_FPM450M060_LSann20D70clear_Img097C_40DA100_BW10Nlam04fpres2_linbarhompre1.fits')
pup_do = readfits('C:/Users/lleboulleux/Desktop/Budget d erreur/APLC designs/ATLAST_4à10/LS_full_atlast.fits')

pup_up = crop(pup_up, /m, nc=614)
pup_do = crop(pup_do, /m, nc=614)

pupdiam = (size(pup_up))[1]

spec_pup = [[[pup_up]], [[pup_DO]]]

power_sp = 2.

phi_up = simu_phi_fourier(pup=pup_up, wfe_rad=wfe_up_rad, seed=5678, power_sp=power_sp)

ampl = simu_phi_fourier(pup= pup_up, wfe_rad=ae_rad, seed=1234, power_sp=power_sp)

phi_up = phi_up - complex(0.,1.) * ampl

phi_do = simu_phi_fourier(pup=pup_do, wfe_rad=wfe_down_rad, seed=5678, power_sp=power_sp)

psf_ref = 1
h_npc = 1

calc_psf_coro_phase, PHI_up = phi_up, $
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
           
p_ref = circminmaxmoy(psf_ref)
normp = max(p_ref)
p_ref = p_ref/normp
N = n_elements(p_ref)
taille_image = (size(h_npc))[1]
; debut LL DARK HOLE
polaire2, rt = 10.*ech*614./708., largeur=taille_image, masque=masko, /double
polaire2, rt = 4.*ech*614./708., largeur=taille_image, masque=maski, /double
dh_area = abs(masko-maski)

norm = max(psf_ref)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; ERROR BUDGET ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

tic = systime(1)

; construction des matrices de coefficients de Zernike
nb_zer = 6
nb_seg = 37

;APLC_Mean_DH = make_array(37, value=0.)

nb_iterations = 1.
nb_steps = 37.
contrastAPLC_vec = make_array(nb_steps, value=0.)
contrastAM_vec = make_array(nb_steps, value=0.)

cd,'C:/Users/lleboulleux/Desktop'
;M = readfits('Moyennes_Matrix.fits')
nb_seg = 37

contrastAPLC_vec_int = make_array(37, value=0.)
contrastAM_vec_int = make_array(37, value=0.)
  
  for inc = 0,36 do begin
    
    A = 0.*randomu(seed, nb_seg, 1)
    A[inc] = 1.
    A[18] = 0.
    
    isolated_coef_zern = make_array(37, nb_zer, value=0.)
    isolated_coef_zern[*,2] = 2.*!PI*A/lambda
    global_coef_zern = make_array(1, 5, value=0.)
    phi_ab = make_phi_atlast_ll(isolated_coef_zern=isolated_coef_zern, global_coef_zern=global_coef_zern)
    phi_ab = crop(phi_ab, /m, nc=614)
    pup = make_pup_atlast_ll(tab_seg=tab_seg, mini_seg=mini_seg)
    pup = crop(pup, /m, nc=614)
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
      
    h_npc = h_npc/normp
    h_npc_zoom3 = crop(h_npc, /m, nc=40)
    dh_area_zoom = crop(dh_area, /m, nc=40)
    ImAPLC = h_npc_zoom3 * dh_area_zoom
    contrastAPLC_vec_int[inc] = mean(h_npc_zoom3[where(dh_area_zoom)])
    
    ImAM = analytical_model(zernike_pol=2, coef=A)
    contrastAM_vec_int[inc] = mean(ImAM[where(dh_area_zoom)])

  endfor

loadct, 3, /silent
aff, alog(ImAPLC >1e-12), 0, z=8

curve = plot(findgen(37), contrastAPLC_vec_int, xrange=[0.,36.], yrange=[1.e-12,1.e-7], color='red', thick=2, xthick=2, ythick=2, /current, /ylog)
curve = plot(findgen(37), contrastAM_vec_int,xrange=[0.,36.], yrange=[1.e-12,1.e-7], color='green', thick=2, xthick=2, ythick=2, /current, /ylog)
;curve = plot(var_vec, max_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='blue', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, contrastAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='red', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, minAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='green', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)
;curve = plot(var_vec, maxAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='blue', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog)

cd, 'C:/Users/lleboulleux/Desktop'
;curve.save, 'Comparison_APLC_AM_piston1nm_Random2.png'

Calibration_Astig0 = contrastAPLC_vec_int
Calibration_Astig0[18] = 0.
Calibration_Astig0 = Calibration_Astig0/contrastAM_vec_int
Calibration_Astig0[18] = 0.

;writefits, 'Calibration_Tilt.fits',Calibration_Astig0

stop

end
