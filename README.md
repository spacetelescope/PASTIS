# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed by Lucie Leboulleux and published in Leboulleux at al. (2018).

------
Quickstart from template:
------
- cd into the directory you want to have your repo in
- `git clone git@github.com:spacetelescope/PASTIS.git`
- `cd <repo>/pastis`
- open `config.ini` with a text editor and change the entry for "local_data_path" to the directory you want to save the data to and "webbpsf_data_path" to the path you have your WebbPSF data saved in. You can figure this out by running `webbpsf.utils.get_webbpsf_data_path()`
- `python aperture_definition.py`
- `python calibration.py`
- run either `python matrix_building_analytical.py` or `python matrix_building_numerical.py`, depending on what matrix you want to use
- open `contrast_calculation.py` and set whether you want to use the numerical or analytical matrix and enter an RMS value you want to use, then run

------
Requirements:
------

numpy
astropy
matplotlib
poppy
webbpsf


------
Setup:
------
- SETUP OF CONFIGFILE AND CHANGING PASTIS PARAMETERS:
The template configfile `config.ini` is supposed to be a static template. It is version controlled and should hence not be changed if you're only using a different parameter setup. You should copy `config.ini` and call the copy `config_local.ini`. This file should be ignored by git (if it is not, add it to your .gitignore file) and you should only change the local configfile to update your parameters.

If you just want to run the code, there is no need for you to open any of the scripts, except in the contrast calculation itself. Just go into your `config_local.ini` and adjust your experimental parameters there, save, and run the code (see below or in the quickstart section). The outputs will be saved in the directory you specified in "local_data_path".

- SETTING UP LOCAL DATA PATHS:
Open up `config_local.ini` and put your global path to where you keep your webbpsf-data in the line [local] --> webbpsf_data_path. you can figure out where this is by printing `webbpsf.utils.get_webbpsf_data_path()` in any python session in which you have imported webbpsf.  Then pick or create a folder in which all the PASTIS data you generate will be saved and copy its global path to [local] --> local_data_path.

-----------------
Running the code:
-----------------

1) adjust the necessary parameters in `config_local.ini`
2) run `aperture_definition.py`, this creates the telescope aperture in the format PASTIS needs it. This is currently (v1.1.0) only implemented for the James Webb Space Telescope.
3) run `calibration.py`, this calibrates the PASTIS model with the E2E simulator (WebbPSF for JWST)
4) run `matrix_building_analytical.py` or `matrix_building_numerical.py` to generate either an analytical or numerical PASTIS matrix

--> Now everything is set up and you can *use* the model to calculate contrasts. Currently only one example script is given: `contrast_calculation.py`. Open it up and adjust:
- keyword `matrix_mode` to pick between using the `numerical` or `analytical` matrix
- keyword `rms_wanted` to set the total RMS value on your telescope for which you want to calculate the contrast

The output will be just a couple of lines, including the calculated contrast values by the E2E model, image PASTIS and matrix PASTIS, e.g. (for `matrix_mode = 'analytical'` and `rms_wanted = 1`):

```
--- CONTRASTS: ---
Mean contrast from WebbPSF: 1.53897501438e-05
Mean contrast with image PASTIS: 2.9384920197e-05
Contrast from matrix PASTIS: 2.9384920197e-05
```

-----------------
Jupyter notebooks:
-----------------

The directory "Jupyter Notebooks" contains a suite of notebooks that test and explain each part of the code step by step. Their numbering refers mostly to the order they were generated in and for easier identification.

-------
Caveats:
-------

PASTIS was developped for high-contrast, low-aberrations regimes and was first validated on an ATLAST geometry in Leboulleux et al. (2018). It's application to JWST is supposed to 1) translate the original IDL code into Python and 2) validate the model on a different telescope setup. This is not fully done yet and we can not guarantee that everything is working correctly.
