# Jupyter

## Auth

You will first need to download an authorization token from the web portal.

1. login at https://cionic.com/a
2. click the profile menu at the top right of the screen
3. select * Download Token *
4. save `token.json` file to the root of `cionic-data`


## Local Setup

### Prerequisites
1. Install Docker: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
2. Navigate to jupyter directory:
   ```bash
   cd <path-to-cionic-data>/jupyter
   ```
3. Start the Docker container:
   ```bash
   docker compose -f docker-compose.yml -f private-volumes.yml up
   ```

### Option 1: JupyterLab (Browser-based)
*Recommended for quick analysis*

1. Once Jupyter is running, look for a URL in the Docker output that looks like:
   ```
   http://127.0.0.1:8888/lab?token=2492c32c2d9fd7e7330c184f276549391d911ee94b81eb2b
   ```
2. Copy and paste this URL into your browser

### Option 2: VS Code
*Recommended for development*

1. Download VS Code: [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)

2. Install required extensions:
   - Remote - Containers (ms-vscode-remote.remote-containers)
   - Python (ms-python.python)
   - Jupyter (ms-toolsai.jupyter)

3. Open the project:
   - File → Open Folder → Select cionic-data directory
   - When prompted, click "Reopen in Container"
   - If not prompted, press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux) and select "Remote-Containers: Reopen in Container"
   - Select working directory
  

## Runner Notebook
1. (JupyterLab only) Click on the folder icon in the top left of jupyter
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
* create a docker-compose file `private-volumes.yml` in `cionic-data/jupyter/` with the following volume specification

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

private-*.yml is added to `.gitignore` to avoid accidental checkin

## Troubleshooting
- When version dependencies change, rebuild the contain with `docker compose up --build`
- If a notebook is misbehaving (e.g, when downloading an `npz`, a `FileNotFoundError` may appear on rare occassions), shut down the notebook's kernel and retry

