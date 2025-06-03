# Scripts

The scripts directory contains useful scripts for managing cionic data

## Auth

You will first need to download an authorization token from the web portal.

1. login at https://cionic.com/a
2. click the profile menu at the top right of the screen
3. select * Download Token *
4. save `token.json` file to the root of `cionic-data`

## Setup

The scripts depend on the following python packages:

```
numpy==1.23.4
requests==2.25.1
pandas==1.5.1
scipy==1.9.3
matplotlib==3.3.4
```

These packages can be installed into your environment with the following commands

Use a Virtual Environment:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

Install packages:

`pip3 install -r scripts/requirements.txt`

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