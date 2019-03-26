# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed by Lucie Leboulleux and published in Leboulleux at al. (2018).

------
Setup:
------
- SETUP OF `config_local.ini` AND CHANGING PASTIS PARAMETERS:
Copy `config.ini` and call the copy `config_local.ini`. This file will be ignored by git and you should only change the local configfile to update your parameters.

If you just want to run the code, there is no need for you to open any of the scripts. Just go into your 1config_local.ini1 and adjust your experimental parameters there, save, and run the code (see below). The outputs will be saved in the directory you specified in "local_data_path".

- SETTING UP LOCAL DATA PATHS:
Open up `config_local.ini` and put your global path to where you keep your webbpsf-data in the line [local] --> webbpsf_data_path. Then pick or create a folder in which all the PASTIS data you generate will be saved and copy its global path to [local] --> local_data_path.

-----------------
Running the code:
-----------------

1) adjust the necessary parameters in config_local.ini
2) run `aperture_definition.py`
3) run `calibration.py`
4) run `matrix_building.py`

--> Now everything is set up and you can *use* the model to calculate contrasts. Some example scripts are given, like:

-) `contrast_calculation.py`
