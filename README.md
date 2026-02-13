# CIONIC Data Tools

This repository provides tools for fetching and manipulating cionic collection data.

There are currently two supported libraries: command line scripts, and jupyter notebooks.  
Both methods make use of shared python code in the [cionic](cionic) directory. 

* [scripts](scripts/README.md)  
  command line scripts for syncing collection data from the cionic servers. quick with minimal setup.
* [jupyter](jupyter/README.md)  
  jupyter notebooks (via docker) for running data analysis. requires docker but offers easier-to-use notebooks.

Testing:

* [tests](tests/README.md)
  pytest tests for the cionic data tools. helps ensure the code works as expected.

Additional documentation:

* [npz](npz.md)  
  outlines the format of cionic npz files

Dependencies:
* The Docker container uses Python 3.11.4 specifically, which is why it has its own jupyter/requirements.txt file.
* If you are not using the Docker container, you can install the dependencies in requirements.txt into a virtual environment on your local machine. See the README files in scripts and tests for instructions. These dependencies are known to work with Python 3, up to version 3.13.1.
