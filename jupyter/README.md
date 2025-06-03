# Jupyter

## Auth

You will first need to download an authorization token from the web portal.

1. login at https://cionic.com/a
2. click the profile menu at the top right of the screen
3. select * Download Token *
4. save `token.json` file to the root of `cionic-data`


## Local Setup
1. Install docker: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
2. cd into the jupyter directory: `cd <path-to-cionic-data>/jupyter`
3. Use docker to bring up jupyter: `docker compose up`
4. Once jupyter is running, you'll see a link in the docker output that looks like: `http://127.0.0.1:8888/lab?token=2492c32c2d9fd7e7330c184f276549391d911ee94b81eb2b`
5. Copy that link and paste will open jupyter in your browser

## Runner Notebook
1. Click on the folder icon in the top left of jupyter
2. Open `analysis/runner.ipynb`
3. Execute the notebook
4. Select options for org / study / notebook
5. Hit run

## Private volumes
A common pattern of use is to have notebooks/code that you do not intend to share publicly.

You can leverage [docker compose -f](https://docs.docker.com/reference/cli/docker/compose/#use--f-to-specify-the-name-and-path-of-one-or-more-compose-files)
to specify additional volumes.

For example:

* create a directory called `cool-analysis` at the same directory level as `cionic-data`
* create a docker-compose file `private-volumes.yml` with the following volume specification

```
version: "3.1"
services:
  jupyter:
    volumes:
      - ../../cool-analysis:/home/jovyan/cool-analysis
```

* start your docker container specifying the primary and additional config files

```
docker-compose -f docker-compose.yml -f private-volumes.yml up
```

private-*.yml is added to `.gitignore` to avoid accidental cehckin
