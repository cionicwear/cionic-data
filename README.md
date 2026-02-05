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





