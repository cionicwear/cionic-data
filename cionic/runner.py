import os
import pathlib
import shutil
import subprocess

import dateutil.parser
import papermill as pm
from IPython.display import HTML, Markdown, display

from cionic import api


def run_collections(
    notebook,
    orgid,
    study,
    collections,
    outdir,
    prepare_only=False,
    overwrite=False,
    limit=None,
    parameters=None,
):
    if parameters is None:
        parameters = {}
    notepath = os.path.abspath(notebook)

    collections = sorted(collections, key=lambda collection: -collection['created_ts'])
    if limit:
        collections = collections[0:limit]

    for collection in collections:

        # until there is a collection number created_ts is the most unique
        unique = collection['num']

        col_dir = f"{outdir}/{unique}"
        nbk_out = f"{col_dir}/{orgid}_{study}_{unique}_{notebook}"
        name = f"{orgid}_{study}_{unique}"

        if overwrite and os.path.exists(col_dir):
            shutil.rmtree(col_dir)

        pathlib.Path(col_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(nbk_out):
            parameters['datapath'] = f"{orgid}/collections/{collection['xid']}"
            parameters['download'] = (
                f"{orgid}/collections/{collection['xid']}/streams/npz"
            )
            parameters['npzpath'] = f"{name}.npz"
            parameters['files_url'] = f"{orgid}/collections/{collection['xid']}/files"
            parameters['files_dir'] = "files/"
            parameters['collection_num'] = collection['num']
            parameters['title'] = collection['title']

            try:
                pm.execute_notebook(
                    notepath,
                    nbk_out,
                    parameters=parameters,
                    prepare_only=prepare_only,
                    cwd=col_dir,
                    store_widget_state=True,
                )
                subprocess.call(['jupyter', 'trust', nbk_out])

            except Exception as e:
                display(
                    Markdown(f"⚠️ **Warning**: Exception in running `{nbk_out}` {e}")
                )

        dt = dateutil.parser.parse(collection['time_created'])
        day = dt.strftime("%m/%d/%Y")
        time = dt.strftime("%H:%M:%S")
        path = f"{orgid}/studies/{study}/collections/{collection['num']}"

        display(Markdown(f"[{collection['title']}]({nbk_out})"))
        display(
            HTML(
                f"""<table><tr>
        <td> {day} </td>
        <td> {time} </td>
        <td> <a href="{api.web_url(path)}">{path}</a> </td>
        </tr></table>"""
            )
        )


def run_metrics(
    metadata_list,
    output_path,
    prepare_only=True,
    overwrite=False,
    tokenpath="../../token.json",
):
    """Run metrics notebook with specified metadata list."""

    # Source metrics notebook path
    source_notebook = "metrics.ipynb"

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    # print(output_path)

    # # Copy metrics notebook to output directory
    dest_notebook = f"{output_path}/{os.path.basename(output_path)}_metrics.ipynb"
    shutil.copy2(source_notebook, dest_notebook)

    # Parameters to inject into the notebook
    parameters = {
        "tokenpath": tokenpath,
        "metadata_list": metadata_list,
    }

    try:
        # Execute the notebook with papermill, injecting the parameters
        pm.execute_notebook(
            input_path=source_notebook,
            output_path=dest_notebook,
            parameters=parameters,
            prepare_only=prepare_only,
            overwrite=overwrite,
        )

        # Trust the executed notebook
        subprocess.call(["jupyter", "trust", dest_notebook])

        display(Markdown(f"**Success**: Metrics notebook created at `{dest_notebook}`"))
        display(
            HTML(f'<a href="{dest_notebook}" target="_blank">Open Metrics Notebook</a>')
        )

        return dest_notebook

    except Exception as e:
        display(Markdown(f"⚠️ **Error**: Failed to create metrics notebook: {e}"))
        return None
