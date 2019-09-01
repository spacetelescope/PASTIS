# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed by Lucie Leboulleux and published in Leboulleux at al. (2018) and Laginja et al (2019).

This release was specifically made to accompany the Laginja et al. (2019) paper and this readme provides quick instructions to get PASTIS results for the LUVOIR-A telescope only. For further info, please anticipate the next release or contact the author under `ilaginja@stsci.edu`.

------
Quickstart from template:
------
- cd into the directory you want to have your repo in
- `git clone git@github.com:spacetelescope/PASTIS.git`
- `cd <repo>/pastis`
- open `config.ini` with a text editor and change the entry for "local_data_path" to the directory you want to save the data to and "local_repo_path" to the path where you keep your PASTIS repository.
- run `python matrix_building_numerical.py` - this will take a couple of hours
- run `hockeystick_contrast_curve.py` to get your contrast pucks in
- run `modal_analysis.py` to get modal and segment constraints
- inspect your results in your output directory


------
Requirements:
------

- numpy
- astropy
- matplotlib
- pandas
- hcipy, specifically from commit `980f39c`
- future releases will include an `environement.yml` file to make setup easier

------
Setup:
------
- SETUP OF CONFIGFILE AND CHANGING PASTIS PARAMETERS:
The template configfile `config.ini` is supposed to be a static template. It is version controlled and should hence not be changed if you're only using a different parameter setup. You should copy `config.ini` and call the copy `config_local.ini`. You should only change the local configfile to update your parameters and not push any changes made to `config.ini`.

- SETTING UP LOCAL DATA PATHS:
Pick or create a folder in which all the PASTIS data you generate will be saved and copy its global path to `[local] --> local_data_path`. Specify where you keep your clone of the PASTIS repository and copy its global path (including "/PASTIS" to `[local] --> local_repo_path`.


----------------
Folder structure:
----------------

In the config file, the entry `[local] --> local_data_path` specifies where the output data will be saved. The code will create a subfolder `active` that contains everything from your current code run. It is set up this way so that if you want to save some data, you just rename that directory into something else.

**matrix_numerical**  
+ contrasts.txt: E2E DH average contrast per aberrated segment pair  
+ PASTISmatrix_num_piston_Noll1.fits: numerical PASTIS matrix  
+ darkholes:  
- dh_cube.fits: all E2E DH images in one cube  
- E2E DH images per aberrated segment pair  
+ OTE_images:  
- OTE images per aberrated segment pair  
+ psfs:  
- psf_cube.fits: all E2E full PSFs in one cube  
- E2E full PSFs per aberrated segment pair  

**results**
Whatever results we’re saving

-----------------
Jupyter notebooks:
-----------------

The directory "Jupyter Notebooks" contains a suite of notebooks that test and explain each part of the code step by step. Their numbering refers to the order they were generated in and exist mostly for easier identification. The most up-to-date ones are in the subdirectory "LUVOIR".
