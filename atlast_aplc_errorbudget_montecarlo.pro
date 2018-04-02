PRO atlast_aplc_errorbudget_montecarlo

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; APLC ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
wfe_up_nm = 0.
ae_nm = 0.
wfe_down_nm = 0.
lambda = 640; 1589. LL

nm2rad = 2.*!dpi/lambda
wfe_up_rad = wfe_up_nm*nm2rad
ae_rad = ae_nm*nm2rad
wfe_down_rad = wfe_down_nm*nm2rad

;flux = 1e9
lyot = 0.9 ;0.96D LL
;largeur = 120.
ech = 2D
;pupdiam = largeur 
throughput = 0.44583701
;throughput = 0.004689451

save = 0
s_path = 'IMGS_BLACK_HOLE-test002/'

;corono = {type:'4QPM', deph:!dpi}
;corono={type:'RRPM', deph:0., size:1.06}
;corono={type:'lyot', size:0D, epsilon:1D}
corono={type:'lyot', size:4.5D*2.*614./708., epsilon:1D}
;corono={type:'parfait'}

apod = 0

meth = 'SA'

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
y_max      = alpha/2-1
x_max_init = alpha/2-1
np     = size_seg
gap    = size_gap
np_tot = (NP+gap) * (alpha / marge)
pupdiam=np_tot
largeur=fix(pupdiam*ech)

print,largeur


pup_meth = largeur
larg_meth = largeur
tab_np = [largeur, largeur]
diam_rec = float(ceil(largeur/ech))
IF diam_rec MOD 2 NE 0 THEN diam_rec = diam_rec + 1
larg_rec = diam_rec

pup_up=readfits('C:/Users/lleboulleux/Desktop/Budget d erreur/APLC designs/ATLAST_4à10/ApodSol_APLC_quart_atlastX025cobs1gap1_N0354r_FPM450M060_LSann20D70clear_Img097C_40DA100_BW10Nlam04fpres2_linbarhompre1.fits')
pup_do=readfits('C:/Users/lleboulleux/Desktop/Budget d erreur/APLC designs/ATLAST_4à10/LS_full_atlast.fits')

pup_up=crop(pup_up,/m,nc=614)
pup_do=crop(pup_do,/m,nc=614)

pupdiam=(size(pup_up))[1]

spec_pup = [[[pup_up]], [[pup_DO]]]

power_sp = 2.

phi_up = simu_phi_fourier(pup = pup_up, $
                          wfe_rad = wfe_up_rad, $
                          seed = 5678, $
                          power_sp = power_sp)

ampl = simu_phi_fourier(pup = pup_up, $
                        wfe_rad = ae_rad, $
                        seed = 1234, $
                        power_sp = power_sp)

phi_up = phi_up-complex(0.,1.)*ampl

phi_do = simu_phi_fourier(pup = pup_do, $
                          wfe_rad = wfe_down_rad, $
                          seed = 5678, $
                          power_sp = power_sp)

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
polaire2, rt = 10.*ech*614./708., largeur = taille_image, masque = masko, /double ;/entre4, 
polaire2, rt = 4.*ech*614./708., largeur = taille_image, masque = maski, /double
dh_area = abs(masko-maski)
;dh_area[0:taille_image/2.-1, *] = 0.

norm = max(psf_ref)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; ERROR BUDGET ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

tic=systime(1)

; construction des matrices de coefficients de Zernike
nb_zer = 6
nb_seg = 37

APLC_Mean_DH=make_array(37, value=0.)

nb_iterations = 250.;1.
nb_steps = 1.;28.;50.
contrastAPLC_vec = make_array(nb_steps, value=0.)
minAPLC_vec = make_array(nb_steps, value=0.)
maxAPLC_vec = make_array(nb_steps, value=0.)
contrast_vec = make_array(nb_steps, value=0.)
min_vec = make_array(nb_steps, value=0.)
max_vec = make_array(nb_steps, value=0.)
var_vec = make_array(nb_steps, value=0.)
contrastAM_vec = make_array(nb_steps, value=0.)

cd,'C:/Users/lleboulleux/Desktop'
M = readfits('Moyennes_Matrix_Piston.fits')
nb_seg = 37

for inc_step = 0,nb_steps-1 do begin

  print, 'ITERATION'
  print, inc_step
  
  if inc_step EQ 27 then var_pm = 1000.
  if inc_step LT 27 then var_pm = (inc_step-17)*100.
  if inc_step LT 18 then var_pm = (inc_step-8)*10.
  if inc_step LT 9 then var_pm = 1+inc_step

  var_pm = 1000.
  var_vec[inc_step] = var_pm
  
  var_nm = var_pm/1000.
  contrast_vec_int = make_array(nb_iterations, value=0.)
  contrastAPLC_vec_int = make_array(nb_iterations, value=0.)
  contrastAM_vec_int = make_array(nb_iterations, value=0.)

  for inc_iteration = 0,nb_iterations-1 do begin
    
    print, 'ITERATION'
    print, inc_step
    print, 'ITERATION'
    print, inc_iteration
    
    A = randomu(seed,nb_seg,1)
    A = A-mean(A)
    lala = pv(A,rms)
    A[18] = 0.
    A = A*var_nm/rms
    contrast_vec_int[inc_iteration] = A##M##transpose(A)  
    
    isolated_coef_zern = make_array(37,nb_zer,value=0.)
    isolated_coef_zern[*,0] = 2.*!PI*A/lambda
    global_coef_zern = make_array(1,5,value=0.)
    phi_ab=make_phi_atlast_ll(isolated_coef_zern = isolated_coef_zern, global_coef_zern = global_coef_zern)
    phi_ab=crop(phi_ab,/m,nc=614)
    pup=make_pup_atlast_ll(tab_seg = tab_seg, mini_seg = mini_seg)
    pup=crop(pup,/m,nc=614)
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
    h_npc = h_npc/normp;(normp*throughput)
    h_npc_zoom3=crop(h_npc,/m,nc=40)
    dh_area_zoom=crop(dh_area,/m,nc=40)
    contrastAPLC_vec_int[inc_iteration]=mean(h_npc_zoom3[where(dh_area_zoom)])
    
    ;tempIm = analytical_model(zernike_pol=0, coef = A)
    ;ImAM = mean(tempIm[where(dh_area_zoom)])
    ;contrastAM_vec_int[inc_iteration]=mean(ImAM[where(dh_area_zoom)])
  endfor

  contrast_vec[inc_step] = mean(contrast_vec_int)
  min_vec[inc_step] = min(contrast_vec_int)
  max_vec[inc_step] = max(contrast_vec_int)
  
  contrastAPLC_vec[inc_step] = mean(contrastAPLC_vec_int)
  minAPLC_vec[inc_step] = min(contrastAPLC_vec_int)
  maxAPLC_vec[inc_step] = max(contrastAPLC_vec_int)
  
  ;contrastAM_vec[inc_step] = mean(contrastAM_vec_int)
endfor

curve = plot(var_vec, contrast_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='red', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Mean contrast for AM')
curve2 = plot(var_vec, min_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='green', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Min contrast for AM')
curve3 = plot(var_vec, max_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='blue', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Max contrast for AM')
curve4 = plot(var_vec, contrastAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='red', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Mean contrast for APLC')
curve5 = plot(var_vec, minAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='green', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Min contrast for APLC')
curve6 = plot(var_vec, maxAPLC_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='blue', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Max contrast for APLC')
;curve = plot(findgen(nb_steps), contrastAM_vec, xtitle='Iteration', ytitle = 'Contrast in DH',title='CONTRAST',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-15,1.e-7],color='blue', linestyle = 3, thick=2, xthick=2,ythick=2,/current)
;leg = legend(target=[curve,curve2,curve3, curve4, curve5, curve6],position=[30.,5e-8], /DATA)
;curve = plot(findgen(50), contrast_vec_int, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',color='red', thick=2, xthick=2,ythick=2,/current,/ylog,name='Mean contrast for AM')
;curve2 = plot(var_vec, min_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='green', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Min contrast for AM')
;curve3 = plot(var_vec, max_vec, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='blue', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Max contrast for AM')
;curve4 = plot(findgen(50), contrastAPLC_vec_int, xtitle='Piston rms amplitude [pm]', ytitle = 'Contrast in DH',title='Contrast sensitivity to piston on segments',color='red', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/ylog,name='Mean contrast for APLC')

error = 100.*(contrast_vec_int-contrastAPLC_vec_int)/contrastAPLC_vec_int
print,pv(error,rms),rms

cd,'C:/Users/lleboulleux/Desktop'
;curve.save,'EB_contrast_AMvsE2E.png'

;writefits,'var_vec_EB_contrast_AMvsE2E.fits',var_vec
;writefits,'contrast_vec_EB_contrast_AMvsE2E.fits',contrast_vec
;writefits,'min_vec_EB_contrast_AMvsE2E.fits',min_vec
;writefits,'max_vec_EB_contrast_AMvsE2E.fits',max_vec
;writefits,'contrastAPLC_vec_EB_contrast_AMvsE2E.fits',contrastAPLC_vec
;writefits,'minAPLC_vec_EB_contrast_AMvsE2E.fits',minAPLC_vec
;writefits,'maxAPLC_vec_EB_contrast_AMvsE2E.fits',maxAPLC_vec

cd,'C:/Users/lleboulleux/Desktop'
var_vec = readfits('var_vec_EB_contrast_AMvsE2E.fits')
contrast_vec = readfits('contrast_vec_EB_contrast_AMvsE2E.fits')
min_vec = readfits('min_vec_EB_contrast_AMvsE2E.fits')
max_vec = readfits('max_vec_EB_contrast_AMvsE2E.fits')
contrastAPLC_vec = readfits('contrastAPLC_vec_EB_contrast_AMvsE2E.fits')
minAPLC_vec = readfits('minAPLC_vec_EB_contrast_AMvsE2E.fits')
maxAPLC_vec = readfits('maxAPLC_vec_EB_contrast_AMvsE2E.fits')
curve = plot(var_vec, contrast_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='red', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Mean contrast for Model')
curve2 = plot(var_vec, min_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='green', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Min contrast for Model')
curve3 = plot(var_vec, max_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='blue', thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Max contrast for Model')
curve4 = plot(var_vec, contrastAPLC_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='red', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Mean contrast for E2E')
curve5 = plot(var_vec, minAPLC_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='green', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Min contrast for E2E')
curve6 = plot(var_vec, maxAPLC_vec, xtitle='Piston rms amplitude $\sqrt<a_{k,1}^2>$ (pm)', ytitle = 'Contrast in DH C',title='Contrast sensitivity to piston on segments',xrange=[min(var_vec),max(var_vec)],yrange=[1.e-12,1.e-7],color='blue', linestyle = 3, thick=2, xthick=2,ythick=2,/current,/xlog,/ylog,name='Max contrast for E2E')
leg = legend(target=[curve,curve2,curve3, curve4, curve5, curve6],position=[30.,5e-8], /DATA)
cd,'C:/Users/lleboulleux/Desktop/FiguresJATIS'
curve.save,'EB_contrast_AMvsE2E.png'

stop

end
