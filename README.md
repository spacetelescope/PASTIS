# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed by Lucie Leboulleux and published in Leboulleux at al. (2018).

------
Setup:
------
- IGNORING config_local.ini:
First, you need to configure your configfile. Don't touch config.ini (I mean you could, it won't do anything though) - this is just a template. Once you have cloned your repo, make sure to make your git "ignore" the config_local.ini, this way you won't be pushing your local setup back to the repo.

- SETTIGN UP LOCAL DATA PATHS:
Open up config_local.ini and put your global path to where you keep your webbpsf-data in the line [local] --> webbpsf_data_path. Then pick or create a folder in which all the PASTIS data you generate will be saved and copy its global path to [local] --> local_data_path.

- CHANGING PASTIS PARAMETERS IN THE CONFIGFILE:
If you just want to run the code, there is no need for you to open any of the scripts. Just go into your congif_local.ini and adjust your experimental parameters there, save, and run the code. The outputs will be saved in the directory you specified in "local_data_path".

-----------------
Running the code:
-----------------

1) adjust the necessary parameters in config_local.ini
2) run aperture_definition.py
3) run calibration.py
4) run matrix_building.py

--> Now everything is set up and you can *use* the model to calculate contrasts. Some example scripts are given, like:

-) contrast_calculation.py
