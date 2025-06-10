import os
import pathlib
import shutil
import subprocess

import dateutil.parser
import papermill as pm
from IPython.display import HTML, Markdown, display

import cionic


def run_collections(
    notebook,
    orgid,
    study,
    collections,
    outdir,
    prepare_only=False,
    overwrite=False,
    limit=None,
    parameters={},
):

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
        <td> <a href="{cionic.web_url(path)}">{path}</a> </td>
        </tr></table>"""
            )
        )
