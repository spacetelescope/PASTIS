<!-- PROJECT SHIELDS -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3382986.svg)](https://doi.org/10.5281/zenodo.3382986)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Python version][python-version-url]


# PASTIS
Sweet liquor from the south of France.

In this repo though, PASTIS is an algorithm for analytical contrast predictions of coronagraphs on segmented telescopes, developed and published in Leboulleux at al. (2018) and Laginja et al. (2020, in prep).

This release was specifically made to accompany the Laginja et al. (2020, in prep) paper and this readme provides quick instructions to get PASTIS results for the LUVOIR-A telescope. For further info, contact the author under `iva.laginja@lam.fr`.

## Table of Contents

* [Quickstart from template](#quickstart-from-template)
  * [Clone the repo and create conda environment](#clone-the-repo-and-create-conda-environment)
  * [Set up local configfile](#set-up-local-configfile)
  * [Create a PASTIS matrix and run the analysis](#create-a-pastis-matrix-and-run-the-analysis)
  * [Changing the input parameters](#changing-the-input-parameters)
* [Requirements](#requirements)
  * [Conda environment](#conda-environment)
  * [hcipy](#hcipy)
  * [Plotting](#plotting)
* [Configuration file](#configuration-file)
* [Output directory](#output-directory)
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
or use SSH if that is your preferred way of cloning repositories:
```bash
$ git clone git@github.com:spacetelescope/PASTIS.git
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

- Go back into your repositories directory and clone `hcipy`:
```bash
$ cd ..
$ git clone https://github.com/ehpor/hcipy.git
```
or
```bash
$ git clone git@github.com:ehpor/hcipy.git
```

- Navigate into the cloned `hcipy` repository:  
```bash
$ cd hcipy
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
$ cd pastis
```

- Copy the file `config.ini` and name the copy `config_local.ini`.

- Open your local configfile `config_local.ini` and edit the entry `[local][local_repo_path]` to point to your local repo clone that you just created, e.g.:
```ini
[local]
...
local_repo_path = /Users/<user-name>/repos/PASTIS
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
$ python run_cases.py
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

If you want to change any of these, please refer to the section about the [configfile](#configuration-file). 

## Requirements

### Conda environment
We provide an `environement.yml` file that can be taken advantage of with the conda package management system. By creating 
a conda environment from this file, you will be set up to run all the code in this package. The only other thing you 
will need is to install `hcipy` correctly, see below.

If you don't know how to start with conda, you can [download miniconda here](https://docs.conda.io/en/latest/miniconda.html). 
After a successful installation, you can create the `pastis` conda environment by navigating with the terminal into the 
`PASTIS` repository, where the `environement.yml` is located, and run:
```bash
$ conda env create --file environment.yml
```
This will create a conda environment called `pastis` that contains all the needed packages at the correct versions 
(except hcipy). If you want to give the environment a different name while it is getting created, you can run:
```bash
$ conda env create --name <env-name> --file environment.yml
```
where you have to replace `<env-name>` with your desired package name.

### `hcipy`
PASTIS relies heavily on the `hcipy` package, most importantly for its implementation of a segmented mirror. The current
PASTIS code is built around an old version of that which is not compatible with the most recent version of `hcipy`. For 
this reason, you will need to clone the `hcipy` repository manually instead of installing the package from pip, then 
checkout the commit with the correct version and install this version into your `pastis` conda environment. To do that,
navigate to the location on disk that contains your repos and clone the `hcipy` repository per http *or* ssh:
```bash
$ git clone https://github.com/ehpor/hcipy.git
$ git clone git@github.com:ehpor/hcipy.git
```
Make sure to activate you `pastis` conda environment since this is where we want to install `hcipy` into:
```bash
$ conda activate pastis
```
Navigate into the `hcipy` repo (`$ cd hicpy`) and checkout the required commit:
```bash
$ git checkout 980f39c
```
Then install the package:
```bash
$ python setup.py install
```
This is a static installation of the `hcipy` package into the conda environment `pastis` only. If you now check out a 
different commit or branch in your local `hcipy` repository, this will not influence this environment. Note how the installation
process will create a "build" directory inside the `hcipy` repository that you are free to delete if you like.

We are currently refactoring our code to be compatible with the improved, current version of `hcipy` and will update our
readme accordingly when this change has successfully happened.

### Plotting
There are certain things that the code is not controlling that are connected to plotting settings with `matplotlib`. Initially,
the plotting should work as expected but the results might be slightly different from what is presented in the paper 
Laginja et al. (2020, in prep), for example where `matplotlib` puts the image origin. If you want to use the lower left
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
This will make it use Type 42 fonts instead.

#### On MacOS Mojave 10.14.6 - backend
The `tkagg` backend makes the machine crash and restart, so don't use that one. The default should run fine, but if you
encounter this issue you can change the default backend in the `matplotlibrc` file under `backend` at almost the very top of the file.


## Configuration file

The main configuration file is `config.ini`, which holds all of your simulation parameters. This specific file, however, 
is version controlled, and the paths to local directories will get messed up if you push or pull this file; you might 
also lose the changes you made to the parameters. This is why `config.ini` is initially supposed to be used as a **template**.

In order to make it work for you, copy `config.ini` and rename the copy to `config_local.ini`. In this **local configfile**, 
you can set all your parameters, and it will override the `config.ini` at runtime. Whichever configfile is used in the 
code, the version-controlled one or the local one, a copy of it is always saved together with the PASTIS matrix output. In the 
case you want to version control the configfile you use, we recommend that you **fork** the repository and simply use the 
`config.ini` file directly.

The first section deals with local paths. Here, you need to point the file to the local clone of your repo and the 
directory you want to have the output data saved to:
```ini
[local]
...
local_data_path = /Users/<user-name>/data_from_repos/pastis_data
local_repo_path = /Users/<user-name>/repos/PASTIS
```

In the next section, you make a selection of the telescope you want to run the analysis on. Currently, only LUVOIR-A is
supported, although there exist other telescope sections in the configfile that were used for testing and setup.
```ini
[telescope]
name = LUVOIR
```
This name has to equal the section name of the configfile that specifies the telescope parameters. In the LUVOIR case, 
we have some parameters for the telescope itself, and for the coronagraph, as well as the operating wavelength.
```ini
[LUVOIR]
; telescope
nb_subapertures = 120
diameter = 15.
gaps = 0.02
optics_path = ${local:local_repo_path}/LUVOIR_delivery_May2019/

; coronagraph
; iwa and owa from dictionaries within files. could move that to util.

; the coro size is not used automatically in the functions, it is always defined (or read from here) manually
coronagraph_size = small
lambda = 500.
```
The number of subapertures will not change, the diameter and gaps are in units of meters. The key `optics_path`  specifies
the data location of the files that define the LUVOIR telescope: aperture, Lyot stop and APLC designs. The path goes into 
the local repo path from `[local] -> [local_path]` and into the right location. There are three APLC designs available, 
with a small, medium and large FPM, and the key `coronagraph_size` lets you switch between them. Finally, `lambda` sets
the wavelength in nanometers.

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
`sampling` defines the image sampling in units of pixels per lambda/D, `im_size_lamD_hcipy` is the total image size of 
the dark hole images in units of lambda/D. `current_analysis` is *not* used in the main launcher script (`run_cases.py`),
but lets you define a matrix directory for repeating an analysis with some of the other scripts and the pastis functions.

Finally, the calibration section defines the local aberration used on each segment and the amplitude of the calibration
aberration for the generation of the PASTIS matrix.
```ini
[calibration]
;! Noll convention!  --- units are NANOMETERS
single_aberration = 1.
zernike = 1
```
`zernike` refers to the local Zernike mode used on the segments as indexed in the section `[zernikes]` (not shown in README),
`1` means piston. `single_aberration` is the amplitude of the calibration aberration of the matrix in nanometers.


## Output directory

Each time a new PASTIS matrix is generated, this will create a new data folder in the directory you specified under
`[local]` --> `[local_data_path]`. These data folders will be of the form `2020-01-13T21-34-29_luvoir-small`, capturing 
date and time of the start of the matrix generation, the telescope name, and for LUVOIR the APLC choice.

The code will copy the used configfile into this data folder, among other things. The data directory structure is as
follows:

```bash
|-- 2020-01-13T21-34-29_example
|   |-- coronagraph_floor.txt: E2E DH average contrast for unaberrated pupil
|   |--matrix_numerical
|      |-- config_local.ini: copy of the configfile used for matrix generation
|      |-- contrasts.txt: list of E2E DH average contrasts per aberrated segment pair
|      |-- OTE_images
|          |-- PDF images of each segment pair aberration in the pupil
|          |-- ...
|      |-- PASTISmatrix_num_piston_Noll1.fits: the PASTIS matrix
|      |-- psfs
|          |-- psf_cube.fits: an image cube of the PSF from each segment pair aberration
|   |--results
|      |-- all results form the PASTIS analysis
|      |-- ...
|   |-- unaberrated_dh.pdf: image of unaberrated DH from E2E simulator
```


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


<!-- MARKDOWN LINKS & IMAGES -->
[python-version-url]: https://img.shields.io/badge/Python-3.7-green.svg?style=flat