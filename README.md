This is the code for my master's project and the University of St Andrews, supervised by Rita Tojeiro.

# Dependencies
One of the Python packages used is Corrfunc, which does not support Windows, so this code must be run on a non-Windows machine or a virtual machine such as `wsl`.

Ensure that the following programs are installed, up to date, and accessible on the PATH
- `make`
- `gcc`
- `gsl`

The Python package dependencies can be found in `requirements.txt`

# Installation
- `git clone` the repository
- Enter the repository and create a virtual environment using `python -m venv`
- Activate the virtual environment using `source .venv/bin/activate`
- Install the Python packages using `pip install -r requirements.txt`

# Project Files
Here is a brief description of all the scripts and their purpose
- `catalogues`
    - `catalogue.py` - Contains functions for creating a galaxy catalogue and plotting a map of the catalogue
    - `const_number_density.py` - Creates a constant number density catalogue
    - `const_stellar_mass.py` - Creates a constant stellar mass catalogue
    - `magnitude_limited.py` - Creates a magnitude limited catalogue
- `correlation_functions`
    -  `correlation_function.py` - Contains a function for calculating the real-space and redshift-space galaxy correlation function of a catalogue
    -  `create_correlation_functions.py` - Calculates the galaxy correlation functions for all of the galaxy catalogues generated in `catalogues`
- `bias_evolution`
    - `bias_evolution.py` - Takes the measured galaxy correlation functions and fits the linear bias
    - `bias_fit.py` - Fits the bias evolution to the bias evolution models
    - `bias_models.py` - Contains the bias evolution models
- `number_density`
    - `number_density.py` - Measures the number density evolution in a galaxy catalogue
