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
   docker compose -f docker-compose.yml up
   ```
4. (Optional) If you want to use private volumes, skip step 3 and add the private-volumes.yml file to the docker compose command instead:
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

3. Click on the folder icon in the top left of jupyter

4. Open `analysis/runner.ipynb`

5. Execute the notebook

6. Select options for org / study / notebook

7. Hit run

### Option 2: VS Code
*Recommended for development*

1. Download VS Code: [https://code.visualstudio.com/Download](https://code.visualstudio.com/Download)

2. Install required extensions by clicking the Extensions icon in the left sidebar. Search for and install:
   - Dev Containers (ms-vscode-remote.remote-containers)
   - Python (ms-python.python)
   - Jupyter (ms-toolsai.jupyter)

3. Open the project:
   - Click the blue icon in the bottom left corner of the window. If you hover over it, it should say "Open a Remote Window."
   - Click "Attach to Running Container..." in the Command Palette in the top center of the window.
   - Select the container named `cionic-data`. If it does not appear, the container may not be running.
   - Click "Open Folder" and select the working directory (`/home/jovyan/`)
  
4. Open `analysis/runner.ipynb`

5. Try to execute the notebook (you can press the "Run All" button in the top of the `runner.ipynb` tab or press `Shift+Enter` on each cell)

6. You will be prompted to select a Python environment. Select the `base` environment. Note: If no options appear, you may need to update the Python or Jupyter extensions and restart VS Code. You can do this by opening the extensions tab, searching for the extensions, and clicking the "Update" button if it appears. After updating, restart VS Code and try again.

Note: the first time running runner.ipynb, the notebook may stall, improperly render the UI or otherwise fail. In this case, restart the kernel, close the notebook tab and try again.

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

