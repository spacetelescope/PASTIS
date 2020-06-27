# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed and published in Leboulleux at al. (2018) and Laginja et al (2020).

This release was specifically made to accompany the Laginja et al. (2020) paper and this readme provides quick instructions to get PASTIS results for the LUVOIR-A telescope. For further info, contact the author under `iva.laginja@lam.fr`.

## Quickstart from template:

*This section will you give all the necessary terminal commands to go from opening our GitHub page in the browser to having 
reduced results of the template on your local machine.*

We assume that you have `conda` and `git` installed and that you're using `bash`.

### Clone the repo and create conda environment

- Navigate to the directory you want to clone the repository into:  
```bash
$ cd /User/<YourUser>/repos/
```

- Clone the repository:
```bash
 git clone https://github.com/spacetelescope/PASTIS.git
```
or use SSH if that is your preferred way of cloning repositories:
```bash
 git clone git@github.com:spacetelescope/PASTIS.git
```

- Navigate into the cloned `PASTIS` repository:  
```bash
cd PASTIS
```

- Create the `pastis` conda environment:  
```bash
$ conda env create --file environment.yml
```

- Activate the environment:
```bash
$ conda activate pastis
```

- Go back into your repositories directory and clone `hcipy`:
```bash
$ cd ..
$ git clone https://github.com/ehpor/hcipy.git
```
or
```bash
git@github.com:ehpor/hcipy.git
```

- Navigate into the cloned `hcipy` repository:  
```bash
cd hcipy
```

- Check out commit `980f39c`:
```bash
$ git checkout 980f39c
```

- Install `hcipy` from this commit:
```bash
$ python setup.py install
```
*Note: This will create a "build" directory that you can delete.*

### Set up local configfile

- Go into the code directory:
```bash
cd pastis
```

- Copy the file `config.ini` and name the copy `config_local.ini`.

- Open your local configfile `config_local.ini` and edit the entry `[local][local_repo_path]` to point to your local repo clone that you just created, e.g.:
```ini
[local]
...
local_repo_path = /Users/<user-name>/<my-repos>/PASTIS
```

- In the same file, define with `[local][local_data_path]` where your output data should be saved to, e.g.:  
```ini
[local]
...
local_data_path = /Users/<user-name>/<path-to-data>
```

### Create a PASTIS matrix and run the analysis

- If not already activated, activate the `pastis` conda environment:
```bash
$ conda activate pastis
```

- Create a PASTIS matrix and analysis for the narrow-angle LUVOIR-A APLC design:
```bash
python run_cases.py
```
This will run for a couple of hours as the first thing that is generated is the PASTIS matrix.
When it is done, you can inspect your results in the path you  specified under `[local][local_data_path]` in your `config_local.ini`!

### Changing the input parameters

The default out-of-the-box analysis from the Quickstart section runs the following case:  
- LUVOIR-A telescope
- narrow-angle ("small") Apodized Pupil Lyot Coronagraph (APLC)
- wavelength = 500 nm
- local aberration = piston
- calibration aberration per segment to generate the PASTIS matrix with: 1 nm

## Requirements:

Set up a conda environment with the provided `environement.yml`:
1) `$ conda env create --file environment.yml`
2) install `hcipy` specifically from commit `980f39c`

## Setup:

- SETUP OF CONFIGFILE AND CHANGING PASTIS PARAMETERS:
The template configfile `config.ini` is supposed to be a static template. It is version controlled and should hence not 
be changed if you're only using a different parameter setup. You should copy `config.ini` and call the copy 
`config_local.ini`. You should only change the local configfile to update your parameters and not push any changes 
made to `config.ini`.

- SETTING UP LOCAL DATA PATHS:
Pick or create a folder in which all the PASTIS data you generate will be saved and copy its global path 
to `[local] --> local_data_path`. Specify where you keep your clone of the PASTIS repository and copy its global 
path (including "/PASTIS" to `[local] --> local_repo_path`.


## Folder structure:

In the configfile, the entry `[local] --> local_data_path` specifies where the output data will be saved.
Each new run will create a new subdirectory whose name starts with a time stamp.

+ coronagraph_floor.txt: E2E DH average contrast for unaberrated pupil
+ unaberrated_dh.pdf: image of unaberrated DH from E2E simulator

**matrix_numerical**  
+ config_local.ini: a copy of the configfile that the matrix was created with
+ contrasts.txt: E2E DH average contrast per aberrated segment pair  
+ OTE_images: PDF images of the segment pair aberrations in the pupil
+ PASTISmatrix_num_piston_Noll1.fits: semi-analytical PASTIS matrix  
+ psfs: psf_cube.fits: all E2E DH images in one cube   

**results**  
contains all results from the PASTIS analysis


## Jupyter notebooks:

The directory "Jupyter Notebooks" contains a suite of notebooks that were used to develop the code on the repository.
Their numbering refers to the order they were generated in and exist mostly for easier identification. The most 
up-to-date ones are in the subdirectory "LUVOIR", although there is no guarantee the notebooks are correct, as the main 
code is in the scripts within `pastis`.
