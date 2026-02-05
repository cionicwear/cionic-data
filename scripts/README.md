# Scripts

The scripts directory contains useful scripts for managing cionic data

## Auth

You will first need to download an authorization token from the web portal.

1. login at https://cionic.com/a
2. click the profile menu at the top right of the screen
3. select * Download Token *
4. save `token.json` file to the root of `cionic-data`

## Setup

The scripts depend on third party python packages including numpy, pandas, scipy, matplotlib, requests, and development tools (black, flake8, isort, pre_commit, pytest).

These packages can be installed into your environment with the following commands

Use a Virtual Environment in the project root:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

Install packages:

`pip3 install -r jupyter/requirements.txt`

Set up pre-commit hooks:

`pre-commit install`

## download.py

The download script enables fetch and segmentation of npz files to the local directory.

```
./scripts/download.py [orgid] [studyid] [-n <collection numbers to download>] [-c <streams to csv>] [-o <output directory>] [-t <filepath to tokenfile>] [-l <limit>] [-f] 

Common usage examples:
./scripts/download.py -h                            (print help)
./scripts/download.py                               (interactive org and study - default limit = 20)
./scripts/download.py -f -l 5                       (interactive org and study - download last 5 collections including collection files)
./scripts/download.py -c emg fquat                  (interactive org and study - create csv files for emg and fquat streams)
./scripts/download.py cionic sample -p quad-assist  (download from cionic org, sample study and quad-assist protocol)
./scripts/download.py cionic sample -n 253 254      (download collections 253 & 254 from cionic org, sample study)

positional arguments:
  orgid               organization shortname
  studyid             study shortname

optional arguments:
  -h, --help          show this help message and exit
  -c CSVS [CSVS ...]  generate CSV for listed streams
  -f                  download additional collection files
  -l LIMIT            number of most recent collections to fetch
  -n NUMS [NUMS ...]  collection numbers to download
  -o OUTDIR           directory to store output
  -p PROTOID          protocol shortname to download
  -t TOKEN            path to auth credentials json file
```

Run with the default parameters, the user will be prompted to select the `orgid` and `studyid`
Each collection from that protcol will be downloaded to the directory  `recordings/<orgid>/<studyid>/`

A folder for each collection will be created with the raw npz `<orgid>_<studyid>_<collnum>.npz`  
and the segmented npz `<orgid>_<studyid>_<collnum>_seg.npz`

If all files from the collection are desired (including videos and notes) specify the `-f` option

For csv export specify a list of stream names to convert.  For example `-c fquat emg` will create CSV files for all quaternion and emg streams

## auth.py

```
usage: 
./scripts/auth.py [email] [org] [-a <admin collector analyst>] [-r <admin collector analyst>]
REQUIRES ORG ADMIN ROLE

positional arguments:
  email             email to grant permission
  org               organization shortname

optional arguments:
  -h, --help        show this help message and exit
  -a ADD [ADD ...]  add role flags: -a analyst collector admin
  -r REM [REM ...]  remove role flags: -d analyst collector admin
  -t TOKEN          path to auth credentials json file
```

## upload_electrode_config.py

This script allows you to upload custom electrode configurations, which will be used for mapping and calibration displays in the corresponding studies in the CIONIC App.
```
usage: 
./scripts/upload_electrode_config.py
    [org_shortname]
    [study_shortname]
    [config_name]
    [json_path]
    [--token_path <filepath to tokenfile>]
    [--new]

Common usage examples:

Print help
./scripts/upload_electrode_config.py -h

Interactive mode - prompts for all inputs
./scripts/upload_electrode_config.py

Create new config in cionic/demo-develop study
./scripts/upload_electrode_config.py cionic demo-develop EXAMPLE_CONFIG \
    ./recordings/cionic/demo-develop/EXAMPLE_CONFIG.json --new

Update existing config in cionic/demo-develop study
./scripts/upload_electrode_config.py cionic demo-develop EXAMPLE_CONFIG \
    ./recordings/cionic/demo-develop/EXAMPLE_CONFIG.json

Interactive config selection - specify only org and study
./scripts/upload_electrode_config.py cionic demo-develop

positional arguments:
  org                 organization shortname
  study               study shortname
  config_name         name of the configuration
  json_path           path to the JSON configuration file

optional arguments:
  -h, --help          show this help message and exit
  --new               create a new config
  --token-path TOKEN_PATH
                      path to the token file (default: ../token.json)
```

## download_electrode_config.py

This script allows you to download custom electrode configurations, which are used for mapping and calibration displays in the corresponding studies in the CIONIC App.

```
usage:
./scripts/download_electrode_config.py
    [org_shortname]
    [study_shortname]
    [--token_path <filepath to tokenfile>]

Common usage examples:

Print help
./scripts/download_electrode_config.py -h

Interactive mode - prompts for org and study selection
./scripts/download_electrode_config.py

Download all configs from cionic/demo-develop study
./scripts/download_electrode_config.py cionic demo-develop

Download all configs from all studies in cionic org
./scripts/download_electrode_config.py cionic

positional arguments:
  org                 organization shortname
  study               study shortname

optional arguments:

  -h, --help          show this help message and exit
  --token-path TOKEN_PATH
                      path to the token file (default: ../token.json)
```


## Committing changes with pre-commit hooks

Pushing changes requires passing formatting and linting standards integrated into pre-commit hooks. These will automatically run when you try to commit, and the commit will be blocked if any tests fail. It is convenient to check if changes will pass prior to committing with:

`pre-commit run --all-files`
