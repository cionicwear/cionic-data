import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime

import dateutil.parser
import ipywidgets as widgets
import papermill as pm
from IPython.display import HTML, Markdown, display
from ipywidgets import Layout

from cionic import api, json2npy, plotting


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


BUTTON_WIDTH = "300px"


class MetadataListCreator:
    """Interactive widget interface for creating metadata lists for analysis."""

    def __init__(
        self,
        output_path: str = "../../recordings/metrics",
        tokenpath: str = "../../token.json",
    ):
        self.tokenpath = os.path.abspath(tokenpath)

        # Data storage
        self.organizations = []
        self.studies = {}
        self.metadata_list = []
        self.current_group = {}

        # State tracking
        self.num_groups = 0
        self.current_group_index = 0
        self.output_path = os.path.abspath(
            f"{output_path}/{self._generate_suggested_path()}"
        )
        self.org_shortname = ""

        # UI state
        self._display_rendered = False
        self._current_step = None

        # Create persistent UI components - these are NEVER recreated
        self.step_content = widgets.VBox([])
        self.json_output = widgets.Textarea(
            value='[]',
            rows=15,
            description="Metadata JSON:",
            layout=Layout(width='800px', height='400px'),
            disabled=True,
        )

        # Main container - created once and never changed
        self.main_container = widgets.VBox(
            [
                widgets.HTML("<h2>Metadata List Creator</h2>"),
                self.step_content,
                widgets.HTML("<h3>Generated Metadata:</h3>"),
                self.json_output,
            ]
        )

    def _load_organizations(self):
        """Load available organizations from API."""
        try:
            auth_data = api.auth(tokenpath=self.tokenpath)
            self.organizations = [org['shortname'] for org in auth_data]
        except Exception as e:
            print(f"Failed to load organizations: {e}")
            self.organizations = []

    def _generate_suggested_path(self) -> str:
        """Generate a suggested output path."""
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{date_str}_output_metrics"

    def _update_step_content(self, new_content):
        """Update the step content area with new widgets."""
        if self._current_step == new_content:
            return  # Prevent duplicate updates

        self._current_step = new_content
        self.step_content.children = new_content

    def _get_phase_labels(self, study_shortname, collection_num):
        npz = api.download_npz_from_metadata(
            org_shortname=self.org_shortname,
            study_shortname=study_shortname,
            collection_num=collection_num,
            tokenpath=self.tokenpath,
            segmented=True,
            include_eulers=False,
            include_gait_splits=False,
        )
        boundaries = json2npy.from_jsonl(npz["gwlabels.jsonl"].split(b"\n"))
        labels = sorted(
            list(set([b["add"]["label"] for b in boundaries])), key=str.lower
        )
        return labels

    def _show_output_path_step(self):
        """Show the output path creation step."""
        path_input = widgets.Text(
            value=self.output_path,
            description="Output Path:",
            placeholder="Enter custom path or use generated one",
            layout=Layout(width='600px'),
        )

        next_button = widgets.Button(
            description="Next: Select Organization",
            button_style='primary',
            layout=Layout(width=BUTTON_WIDTH),
        )

        def on_next_clicked(b):
            self.output_path = path_input.value.strip()
            if self.output_path:
                self._show_org_selection_step()
            else:
                print("Please enter an output path")

        next_button.on_click(on_next_clicked)

        content = [
            widgets.HTML(
                "<h3>Step 1: Create Output Path</h3>"
                "<p>Specify the output path for your analysis:</p>"
            ),
            path_input,
            next_button,
        ]

        self._update_step_content(content)

    def _show_org_selection_step(self):
        """Show organization selection step."""
        org_select = widgets.Dropdown(
            options=self.organizations,
            description="Organization:",
            layout=Layout(width='300px'),
        )

        next_button = widgets.Button(
            description="Next: Number of Groups",
            button_style='primary',
            layout=Layout(width=BUTTON_WIDTH),
        )

        def on_next_clicked(b):
            if org_select.value:
                self.org_shortname = org_select.value
                self._load_studies_for_org()
                self._show_group_count_step()
            else:
                print("Please select an organization")

        next_button.on_click(on_next_clicked)

        content = [
            widgets.HTML(
                f"<h3>Step 2: Select Organization</h3>"
                f"<p>Output path: <code>{self.output_path}</code></p>"
            ),
            org_select,
            next_button,
        ]

        self._update_step_content(content)

    def _load_studies_for_org(self):
        """Load studies for the selected organization."""
        try:
            studies_data = api.get_cionic(f'{self.org_shortname}/studies')
            self.studies[self.org_shortname] = studies_data
        except Exception as e:
            print(f"Failed to load studies: {e}")
            self.studies[self.org_shortname] = []

    def _show_group_count_step(self):
        """Show step to specify number of groups."""
        count_input = widgets.IntText(
            value=2,
            description="# Groups:",
            min=1,
            max=100,
            layout=Layout(width='200px'),
        )

        next_button = widgets.Button(
            description="Next: Create Groups",
            button_style='primary',
            layout=Layout(width=BUTTON_WIDTH),
        )

        def on_next_clicked(b):
            if count_input.value > 0:
                self.num_groups = count_input.value
                self.current_group_index = 0
                self._show_group_creation_step()
            else:
                print("Please enter a valid number of groups")

        next_button.on_click(on_next_clicked)

        content = [
            widgets.HTML(
                f"<h3>Step 3: Specify Number of Groups</h3>"
                f"<p>Output path: <code>{self.output_path}</code></p>"
                f"<p>Organization: <code>{self.org_shortname}</code></p>"
            ),
            count_input,
            next_button,
        ]

        self._update_step_content(content)

    def _show_group_creation_step(self):
        """Show group creation step for current group."""
        # Initialize current group
        self.current_group = {
            "group_name": "",
            "org_shortname": self.org_shortname,
            "group_color": plotting.COLORS[
                self.current_group_index % len(plotting.COLORS)
            ],
            "recordings": [],
        }

        group_name_input = widgets.Text(
            description="Group Name:",
            placeholder="Enter group name",
            layout=Layout(width='300px'),
        )

        color_display = widgets.Text(
            value=self.current_group["group_color"],
            description="Color Display:",
            placeholder="Enter custom color or use generated one",
            layout=Layout(width='300px'),
        )

        next_button = widgets.Button(
            description="Next: Add Recording",
            button_style='primary',
            layout=Layout(width=BUTTON_WIDTH),
        )

        def on_next_clicked(b):
            if group_name_input.value.strip():
                self.current_group["group_name"] = group_name_input.value.strip()
                self.current_group["group_color"] = color_display.value.strip()
                self._show_recording_creation_step()
            else:
                print("Please enter a group name")

        next_button.on_click(on_next_clicked)

        content = [
            widgets.HTML(
                f"<h3>Step 4: Create Group {self.current_group_index + 1} of "
                f"{self.num_groups}</h3>"
                f"<p>Output path: <code>{self.output_path}</code></p>"
                f"<p>Organization: <code>{self.org_shortname}</code></p>"
            ),
            group_name_input,
            color_display,
            next_button,
        ]

        self._update_step_content(content)

    def _show_recording_creation_step(self):
        """Show recording creation step."""
        studies = self.studies.get(self.org_shortname, [])
        study_shortnames = sorted(
            [s.get("shortname", "") for s in studies], key=str.lower
        )
        study_select = widgets.Dropdown(
            options=study_shortnames,
            description="Study:",
            layout=Layout(width='300px'),
        )

        # Initialize collection dropdown
        collection_input = widgets.Dropdown(
            options=[],
            description="Collection #:",
            layout=Layout(width='300px'),
        )

        def update_collections(change=None):
            """Update collection options when study changes."""
            if study_select.value:
                study = next(
                    (s for s in studies if s.get("shortname") == study_select.value),
                    None,
                )
                if study:
                    collection_kwargs = {"sxid": study["xid"]}
                    collections = api.get_cionic(
                        "cionic/collections", **collection_kwargs
                    )
                    collection_input.options = sorted(
                        [int(c["num"]) for c in collections], reverse=True
                    )
                else:
                    collection_input.options = []
            else:
                collection_input.options = []

        # Update collections when study changes
        study_select.observe(update_collections, names='value')

        # Initialize collections for the first selected study
        update_collections()

        # Initialize label dropdown
        label_input = widgets.Dropdown(
            options=[],
            description="Label:",
            placeholder="e.g., stim_walk",
            layout=Layout(width='300px'),
        )

        def update_labels(change=None):
            """Update label options when collection changes."""
            if study_select.value and collection_input.value is not None:
                labels = self._get_phase_labels(
                    study_shortname=study_select.value,
                    collection_num=collection_input.value,
                )
                label_input.options = labels
            else:
                label_input.options = []

        collection_input.observe(update_labels, names='value')

        add_recording_button = widgets.Button(
            description="Add Recording",
            button_style='success',
            layout=Layout(width=BUTTON_WIDTH),
        )

        # Create recordings display
        recordings_display = widgets.HTML(self._format_current_recordings())

        # Progress button area
        progress_area = widgets.VBox([])

        def update_progress_button():
            if len(self.current_group["recordings"]) > 0:
                if self.current_group_index < self.num_groups - 1:
                    progress_text = "Next Group"
                else:
                    progress_text = "Finish"

                progress_button = widgets.Button(
                    description=progress_text,
                    button_style='primary',
                    layout=Layout(width=BUTTON_WIDTH),
                )

                def on_progress_clicked(b):
                    self._finalize_current_group()

                progress_button.on_click(on_progress_clicked)
                progress_area.children = [progress_button]
            else:
                progress_area.children = []

        def on_add_recording_clicked(b):
            if (
                study_select.value
                and collection_input.value
                and label_input.value.strip()
            ):
                recording = {
                    "study_shortname": study_select.value,
                    "collection_num": collection_input.value,
                    "label": label_input.value.strip(),
                }
                self.current_group["recordings"].append(recording)

                # Clear inputs
                collection_input.value = None
                label_input.value = None

                # Update displays
                recordings_display.value = self._format_current_recordings()
                update_progress_button()
                self._update_json_output()
            else:
                print("Please fill in all recording fields")

        add_recording_button.on_click(on_add_recording_clicked)
        update_progress_button()

        content = [
            widgets.HTML(
                f"<h3>Add Recordings to Group: {self.current_group['group_name']}</h3>"
                f"<p>Output path: <code>{self.output_path}</code></p>"
                f"<p>Organization: <code>{self.org_shortname}</code></p>"
            ),
            study_select,
            collection_input,
            label_input,
            recordings_display,
            widgets.HBox([add_recording_button]),
            progress_area,
        ]

        self._update_step_content(content)

    def _format_current_recordings(self) -> str:
        """Format current recordings for display."""
        if not self.current_group["recordings"]:
            return "<p><em>No recordings added yet</em></p>"

        html = "<h4>Current Recordings:</h4><ul>"
        for rec in self.current_group["recordings"]:
            html += (
                f"<li><strong>{rec['study_shortname']}</strong> - Collection: "
                f"{rec['collection_num']} - Label: {rec['label']}</li>"
            )
        html += "</ul>"
        return html

    def _finalize_current_group(self):
        """Finalize current group and move to next or finish."""
        # Add current group to metadata list
        self.metadata_list.append(self.current_group.copy())
        self._update_json_output()

        # Move to next group or finish
        self.current_group_index += 1

        if self.current_group_index < self.num_groups:
            # Start next group
            self._show_group_creation_step()
        else:
            # Finished all groups
            self._show_completion_step()

    def _show_completion_step(self):
        """Show completion step with final results."""
        run_metrics_button = widgets.Button(
            description="Generate New Metrics Notebook",
            button_style="primary",
            layout=Layout(width=BUTTON_WIDTH),
        )

        def on_run_metrics_clicked(b):
            try:
                self._run_metrics()
                print("✅ Metrics analysis completed successfully!")
            except Exception as e:
                print(f"❌ Error running metrics analysis: {e}")

        run_metrics_button.on_click(on_run_metrics_clicked)

        summary = f"""
        <h3>Metadata List Complete!</h3>
        <p><strong>Output Path:</strong> {self.output_path}</p>
        <p><strong>Organization:</strong> {self.org_shortname}</p>
        <p><strong>Groups Created:</strong> {len(self.metadata_list)}</p>
        <p><strong>Total Recordings:</strong> {
            sum(len(group['recordings']) for group in self.metadata_list)
        }</p>
        """

        content = [widgets.HTML(summary), widgets.HBox([run_metrics_button])]

        self._update_step_content(content)

    def _update_json_output(self):
        """Update the JSON output display."""
        self.json_output.value = json.dumps(self.metadata_list, indent=2)

    def display(self):
        """Display the metadata creator interface."""

        if not self._display_rendered:
            self._load_organizations()
            display(self.main_container)
            self._display_rendered = True

        self._show_output_path_step()

    def _run_metrics(self):
        run_metrics(
            metadata_list=self.metadata_list,
            output_path=self.output_path,
            tokenpath=self.tokenpath,
        )
