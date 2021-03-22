<!-- PROJECT SHIELDS -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4288496.svg)](https://doi.org/10.5281/zenodo.4288496)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Python version][python-version-url]


# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes,
developed and published in [Leboulleux et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018JATIS...4c5002L/abstract) 
and [Laginja et al. (2021, accepted for publication in JATIS)](https://ui.adsabs.harvard.edu/abs/2021arXiv210306288L/abstract).

This release brings significant updates especially in the PASTIS matrix calculations, which is now multiprocessed. We also
take advantage of the fact that the PASTIS matrix is symmetrical, which allows us to calcualte only half of the contrast
matrix, including the diagonal, before calculating the PASTIS matrix.

This readme provides quick instructions to get PASTIS results for the LUVOIR-A telescope, as well as more detailed info
about the code and other telescopes it suppoerts. For further info, contact the author under `iva.laginja@lam.fr`.

## Table of Contents

* [Quickstart from template](#quickstart-from-template)
  * [Clone the repo and create conda environment](#clone-the-repo-and-create-conda-environment)
  * [Set up local configfile](#set-up-local-configfile)
  * [Create a PASTIS matrix and run a full analysis](#create-a-pastis-matrix-and-run-a-full-analysis)
  * [Changing the input parameters](#changing-the-input-parameters)
* [Full Requirements](#full-requirements)
  * [Git](#git)
  * [Conda environment](#conda-environment)
  * [The package `hcipy`](#the-package-hcipy)
  * [Plotting](#plotting)
  * [Known `matplotlib` issues on MacOS](#known-matplotlib-issues-on-macos)
* [Configuration file](#configuration-file)
* [Output directory](#output-directory)
* [PASTIS analysis](#pastis-analysis)
  * [Rerunning just the analysis](#rerunning-just-the-analysis)
* [Supported Simulators](#supported-simulators)
  * [LUVOIR-A](#luvoir-a)
  * [JWST](#jwst)
  * [HiCAT](#hicat)
* [Jupyter notebooks](#jupyter-notebooks)
* [About this repository](#about-this-repository)
  * [Contributing and code of conduct](#contributing-and-code-of-conduct)
  * [Citing](#citing)
  * [License](#license)


## Quickstart from template

*This section will give you all the necessary terminal commands to go from opening this GitHub page in the browser to having 
reduced results of the template on your local machine.*

We assume that you have `conda` and `git` installed and that you're using `bash`.

### Clone the repo and create conda environment

- Navigate to the directory you want to clone the repository into:  
```bash
$ cd /User/<YourUser>/repos/
```

- Clone the repository:
```bash
$ git clone https://github.com/spacetelescope/PASTIS.git
```

- Navigate into the cloned `PASTIS` repository:  
```bash
$ cd PASTIS
```

- Create the `pastis` conda environment:  
```bash
$ conda env create --file environment.yml
```

- Activate the environment:
```bash
$ conda activate pastis
```

- Install the package into this environment in develop mode
```bash
$ python setup.py develop
```

### Set up local configfile

- Go into the code directory:
```bash
$ cd pastis
```

- Copy the file `config_pastis.ini` and name the copy `config_local.ini`.

- Open your local configfile `config_local.ini` and find the section `[local]`. In that section, define where all the 
output data will be saved to by adjusting the key `local_data_path`, e.g. (for more about the configfile, see 
[Configuration file](#configuration-file)):
```ini
[local]
...
local_data_path = /Users/<user-name>/<path-to-data>
```
**Save `config_local.ini` after these edits.**

### Create a PASTIS matrix and run a full analysis

- If not already activated, activate the `pastis` conda environment with `$ conda activate pastis` and get into the 
`PASTIS/pastis/launchers` subfolder.

- Create a PASTIS matrix and run the PASTIS analysis for the narrow-angle LUVOIR-A APLC design from the default template:
```bash
$ python run_luvoir.py
```
**This will run for a couple of hours** as the first thing that is generated is the PASTIS matrix. On a 13-in MacBook 
Pro 2020, the matrix gets calculated in about 80min, and the analysis runs in about 15 minutes.
When it is done, you can inspect your results and log files in the path you specified under `local_data_path` in the section `[local]`
of your `config_local.ini`!

### Changing the input parameters

The default out-of-the-box analysis from the Quickstart section runs the following case:  
- LUVOIR-A telescope
- narrow-angle ("small") Apodized Pupil Lyot Coronagraph (APLC)
- wavelength = 500 nm
- local aberration = piston
- calibration aberration per segment to generate the PASTIS matrix with: 1 nm

If you want to change any of these, please refer to the section about the [configfile](#configuration-file) and [Supported Simulators](#supported-simulators). 

## Full Requirements

### Git

You will need `git` to clone this repository. Already a `git` user? Jump ahead. If not, please don't be *that* person 
who downloads the code and doesn't use version control. If you need a primer on `git`, 
[see here](https://swcarpentry.github.io/git-novice/). For the fastest ways to install `git`:
- To install it on a Mac, type `git` in your terminal and follow the instructions to install the Apple Xcode command tools.
- To make it easy on Windows, [install Git Bash](https://gitforwindows.org/). **Note**: If 
you will use Git Bash with Miniconda (see below), you will have to add Miniconda to your PATH during setup, even if it 
is marked as not recommended. Otherwise Git Bash can't access it.
- For Linux, [follow this link](https://gist.github.com/derhuerst/1b15ff4652a867391f03#file-linux-md).

### Conda environment
We provide an `environment.yml` file that can be taken advantage of with the conda package management system. By creating 
a conda environment from this file, you will be set up to run all the code in this package. This includes the 
 installation of the `hcipy` package from a specific commit of the repository, see below.

If you don't know how to start with conda, you can [download miniconda here](https://docs.conda.io/en/latest/miniconda.html). 
After a successful installation, you can create the `pastis` conda environment by navigating with the terminal into the 
`PASTIS` repository, where the `environment.yml` is located, and run:
```bash
$ conda env create --file environment.yml
```
This will create a conda environment called `pastis` that contains all the needed packages at the correct versions. If 
you want to give the environment a different name while it is getting created, you can run:
```bash
$ conda env create --name <env-name> --file environment.yml
```
where you have to replace `<env-name>` with your desired package name.

If you ever need to update your conda environment from the file `environment.yml`, you can do that with 
(where "pastis" is the environment name):
```bash
conda env update -n pastis -f environment.yml
```

You can also remove a conda environment with:
```bash
conda remove --name pastis --all
```

### Plotting
There are certain things that the code is not controlling that are connected to plotting settings with `matplotlib`. Initially,
the plotting should work as expected but the results might be slightly different from what is presented in the paper 
Laginja et al. (2020, submitted), for example where `matplotlib` puts the image origin. If you want to use the lower left
as your origin, this is adjustable in the plots directly by setting `origin=lower`, although I recommend adjusting your global
plotting parameters in the `matplotlibrc` file so that you don't have to edit each plot manually.

In short, you can find the location of your rc file on disk by running:
```python
>>> import matplotlib
>>> matplotlib.matplotlib_fname()
'/home/<some-path>/.config/matplotlib/matplotlibrc'
```
Opening up this file will show you a template rc file like described in the [matplotlib documentation for these settings](https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html#the-matplotlibrc-file).
To set your image origin permanently to the lower left for whenever you plot anything with `matplotlib`, search for `image.origin`
within that file, delete the `#` which is commenting it out and set it to `lower`:
```bash
image.origin : lower
```
then save and close.

### Known `matplotlib` issues on MacOS

While writing code for the repository, we ran into a couple of other issues with the plotting that were dependent on the
OS and its version that we were using. If you run into the same issues, here is how we solved them:

#### On MacOS Catalina 10.15.5 - PDF font types
It does not support Type 3 PostScript fonts in PDF documents, while `matplotlib` uses Type 3 fonts by default.
We got the error message:
```bash
The PDF backend does not currently support the selected font.
```
To  mitigate this, go to your `matplotlibrc` file and make sure to uncomment and set:
```bash
pdf.fonttype       : 42
```
This will make it use Type 42 fonts instead. Instead of permanently editing you rc file, you can also drop in these two
lines in the scripts concerned:
```python
import matplotlib
matplotlib.rc('pdf', fonttype=42)
```

#### On MacOS Mojave 10.14.6 - backend
The `tkagg` backend makes the machine crash and restart, so don't use that one. The default should run fine, but if you
encounter this issue you can change the default backend in the `matplotlibrc` file under `backend` at almost the very top of the file.


## Configuration file

The main configuration file is `config_pastis.ini`, which holds all of your simulation parameters. This specific file, however, 
is version controlled, and the paths to local directories will get messed up if you push or pull this file; you might 
also lose the changes you made to the parameters. This is why `config_pastis.ini` is initially supposed to be used as a **template**.

In order to make it work for you, copy `config_pastis.ini` and rename the copy to `config_local.ini`. In this **local configfile**, 
you can set all your parameters, and it will override the `config_pastis.ini` at runtime. This means that if there is a `config_local.ini`,
it will be used, if not, the code will fall back on `config_pastis.ini`. A copy of the used configfile is always saved together 
with the PASTIS matrix output when a matrix is generated. In the case you want to version control the configfile you use, 
we recommend that you **fork** the repository and simply use the `config_pastis.ini` file directly.

The first section deals with local paths. Here, you need to point the file to the directory you want to have the output data saved to:
```ini
[local]
...
local_data_path = /Users/<user-name>/data_from_repos/pastis_data
```

In the next section, you make a selection of the telescope you want to run the analysis on. See [Supported Simulators](#supported-simulators) to see
what telescopes `pastis` currently supports.
```ini
[telescope]
name = LUVOIR
```
This name has to equal the section name of the configfile that specifies the telescope parameters. In the LUVOIR case, 
we have some parameters for the telescope itself, and for the coronagraph, as well as the operating wavelength.
```ini
[LUVOIR]
; aberration for matrix calculation, in NANOMETERS
calibration_aberration = 1.
; log10 limits of PASTIS validity in nm WFE
valid_range_lower = -4
valid_range_upper = 4

; telescope
nb_subapertures = 120
diameter = 15.
gaps = 0.02
optics_path_in_repo = LUVOIR_delivery_May2019/
aperture_path_in_optics = inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000.fits
indexed_aperture_path_in_optics = inputs/TelAp_LUVOIR_gap_pad01_bw_ovsamp04_N1000_indexed.fits
lyot_stop_path_in_optics = inputs/LS_LUVOIR_ID0120_OD0982_no_struts_gy_ovsamp4_N1000.fits

; coronagraph
; iwa and owa from dictionaries within files. could move that to util.

; the coro size is not used automatically in the functions, it is always defined (or read from here) manually
coronagraph_size = small
lambda = 500.
```
The number of subapertures will not change, the diameter and gaps are in units of meters. The key `optics_path_in_repo`  specifies
the data location (within the repository) of the files that define the LUVOIR telescope: aperture, Lyot stop and APLC
designs.There are three APLC designs available, with a small, medium and large FPM, and the key `coronagraph_size` lets 
you switch between them. Finally, `lambda` sets the wavelength in nanometers. The three keys below `optics_path_in_repo`
are paths within the optics path that point to different files needed by the LUVOIR simulator.

The `calibration_aberration` is the local aberration coefficient that will be used when calculating the PASTIS matrix, and
the two parameters below set the upper and lower log limits of the total pupil WFE rms value for which the hockey stick curve
will be calcualted.

The following section sets some image parameters:
```ini
[numerical]
...
sampling = 4.
...
im_size_lamD_hcipy = 30

; this is not used automatically in the functions, it is always defined (or read from here) manually
current_analysis = 2020-01-13T21-34-29_luvoir-small
```
The key `sampling` defines the image sampling in units of pixels per lambda/D, `im_size_lamD_hcipy` is the total image size of 
the dark hole images in units of lambda/D. The key `current_analysis` is *not* used in the main launcher scripts (e.g. `run_luvoir.py`),
but lets you define a matrix directory for repeating an analysis with the main function in the modules `hockeystick_contrast_curve.py`, 
`pastis_analysis.py` and `single_mode_error_budget.py`.

Finally, there is a section defining how we count our Zernikes, and the calibration section defines the local aberration 
used on each segment, by the numbering defined in the `[zernikes]` section (not shown in README).
```ini
[calibration]
local_zernike = 1
```


## Output directory

Each time a new PASTIS matrix is generated, this will create a new data folder in the directory you specified in the
section `[local]` with the key `local_data_path`. These data folders will be of the form `2020-01-13T21-34-29_luvoir-small`, capturing 
date and time of the start of the matrix generation, the telescope name, and for LUVOIR the APLC choice.

The code will copy the used configfile into this data folder, together with all results and log files. The data 
directory structure is as follows:

```bash
|-- 2020-11-20T21-34-29_example
|   |-- coronagraph_floor.txt                    # E2E DH average contrast for unaberrated pupil
|   |-- full_report.pdf                          # a PDF file summarizing all results 
|   |-- matrix_numerical
|       |-- config_local.ini                     # copy of the configfile used for matrix generation
|       |-- contrast_matrix.pdf                  # PDF image of the half-filled contrast matrix, before it is transformed into the PASTIS matrix
|       |-- OTE_images
|           |-- opd[...].pdf                     # PDF images of each segment pair aberration in the pupil
|           |-- ...
|      |-- contrast_matrix.fits:                 # contrast matrix - E2E DH average contrasts per aberrated segment pair (only half of it since it is symmetric), contrast floor is NOT subtracted
|      |-- contrast_matrix.pdf:                  # PDF image of contrast matrix
|      |-- pastis_matrix_example.log             # logging output of matrix calculation
|      |-- pastis_matrix.pdf                     # PDF image of the calculated PASTIS matrix 
|      |-- pastis_matrix.fits                    # the PASTIS matrix
|      |-- psfs
|          |-- psf_cube.fits                     # an image cube of the PSF from each segment pair aberration
|   |-- pastis_analysis.log:                     # logging output of the PASTIS analysis; new runs get appended
|   |-- results
|       |-- [...].pdf/.txt                       # all results from the PASTIS analysis, including the modes
|       |-- ...
|   |-- title_page.pdf                           # title page of the full_report PDF file 
|   |-- unaberrated_dh.fits                      # image of unaberrated DH from E2E simulator
|   |-- unaberrated_dh.pdf                       # PDF image of unaberrated DH from E2E simulator
```


## PASTIS analysis

### Rerunning just the analysis
Calculating the PASTIS matrix takes some time, but once this is over the PASTIS analysis can be redone on it without 
having to regenerate the matrix. To do this, open the script `run_luvoir.py` and comment out the line that calls 
the matrix calculation function:
```py
    #dir_small = num_matrix_luvoir(design='small')
```
and instead uncomment the line where you can pre-define the data directory, and drop in the correct folder directory 
within your output destination:
```py
    dir_small = os.path.join(CONFIG_PASTIS.get('local', 'local_data_path'), '<your-data-directory_small>')
```

If you now run `run_luvoir.py`, it will only rerun the analysis.

## Supported Simulators
`pastis` currently supports E2E simulators for three telescopes: LUVOIR-A, JWST and HiCAT. Only LUVOIR comes with a built-in
E2E simulator within `pastis`. The simulator for JWST is `webbpsf` and can be installed additionally, while the HiCAT simulator
is currently private. The analysis for each of them can be started with the respective launcher in `pastis/launchers`.

### LUVOIR-A
There is a built-in LUVOIR-A simulator readily usable within the pastis package, and it supports the three baseline APLC designs
for this observatory. The script `run_luvoir.py` is pre-set to easily run the medium and large design APLCs of LUVOIR-A as well. You just need
to uncomment the according lines and it will generate the matrices, and run the PASTIS analysis for those cases as well.

### JWST
The coronagraphs currently supported on JWST are the NIRCam coronagraphs. You will need to install `webbpsf`
([installation instructions here](https://webbpsf.readthedocs.io/en/latest/installation.html#installing-with-conda-but-not-astroconda))
in order to be able to use it with `pastis`, and don't forget that you also need to install the [required data files](https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files).
Once you are done with the installation, you will also need to drop in your local path to your new webbpsf data files to the
PASTIS configfile. you can figure out the path to your webbpsf by running the following in a python session:
```py
import webbpsf
webbpsf.utils.get_webbpsf_data_path()
```
Then you need to copy the output to the (local) configfile in the following section:
```ini
[local]
; figure out webbpsf-data path with: webbpsf.utils.get_webbpsf_data_path()
webbpsf_data_path = ...
```

The launcher for a JWST analysis is also located in `pastis/launchers`, as `run_jwst.py`.

### HiCAT
The HiCAT simulator is private and its support is only provided for internal use.


## Jupyter notebooks

The directory "Jupyter Notebooks" contains a suite of notebooks that were used to develop the code on the repository.
Their numbering refers to the order they were generated in and exist mostly for easier identification. The most 
up-to-date ones are in the subdirectory "LUVOIR", although there is no guarantee the notebooks are correct, as the main 
code is in the scripts within `pastis`.


## About this repository

### Contributing and code of conduct

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines, and the process for submitting issues and pull requests to us.
Please also see our [CODE OF CONDUCT.md](CODE_OF_CONDUCT.md).

### Citing

If you use this code in your work, please find citation snippets to give us credit with in [CITING.md](CITING.md).

### License

This project is licensed under the BSD-3-Clause-License - see the [LICENSE.md](LICENSE.md) file for details.

### Acknowledgments

Big thanks to Robel Geda ([@robelgeda](https://github.com/robelgeda)) for testing, checking and providing suggestions for the 
repo setup, quickstart and README.


<!-- MARKDOWN LINKS & IMAGES -->
[python-version-url]: https://img.shields.io/badge/Python-3.7-green.svg?style=flat
