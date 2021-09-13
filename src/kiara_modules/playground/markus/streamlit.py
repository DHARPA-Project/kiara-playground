# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import typing

from kiara import Kiara
from kiara.data import Value, ValueSet
from kiara.pipeline.controller.batch import BatchControllerManual
from kiara.processing import JobStatus
from kiara.workflow.kiara_workflow import KiaraWorkflow
from streamlit.uploaded_file_manager import UploadedFile


def init_session(
    st,
    module_type: str,
    module_config: typing.Optional[typing.Mapping[str, typing.Any]] = None,
):

    if "kiara" not in st.session_state:
        print("Create kiara object.")
        kiara = Kiara()
        st.session_state["kiara"] = kiara
    else:
        kiara = st.session_state["kiara"]

    if "workflow" not in st.session_state:
        print(f"Create workflow in session: {module_type}")

        controller = BatchControllerManual()
        workflow: KiaraWorkflow = kiara.create_workflow(
            module_type, module_config=module_config, controller=controller
        )
        st.session_state["workflow"] = workflow
    else:
        workflow = st.session_state["workflow"]

    return (
        kiara,
        workflow,
    )


def set_workflow_input(workflow: KiaraWorkflow, process: bool = False, **inputs):

    failed = {}
    for k, v in inputs.items():
        try:
            workflow.inputs.set_value(k, v)
        except Exception as e:
            failed[k] = e

    print(failed)

    if process:
        try:
            workflow.pipeline.controller.process_pipeline()
        except Exception as e:
            print(e)


def process_to_stage(workflow: KiaraWorkflow, stage_nr: int):

    controller: BatchControllerManual = workflow.pipeline.controller  # type: ignore
    controller.process_stage(stage_nr=stage_nr)


def get_step_output(workflow, step_id: str, output_name: str) -> Value:

    values: ValueSet = workflow.pipeline.get_step_outputs(step_id=step_id)
    value = values.get_value_obj(output_name)

    return value


def find_all_aliases_of_type(kiara: Kiara, value_type: str) -> typing.List[str]:

    result = []
    for alias in kiara.data_store.alias_names:
        v_obj = kiara.data_store.get_value_obj(alias)
        assert v_obj is not None
        if v_obj.type_name == value_type:
            result.append(alias)

    return result


def import_bytes(kiara: Kiara, uploaded_file: UploadedFile):

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        imported: ValueSet = kiara.run(  # type: ignore
            "file.import_from.local.file_path",
            inputs={"source": path},
        )
        file_item = imported.get_value_obj("value_item")
        assert file_item is not None
        file_item.save(aliases=[uploaded_file.name.replace(".", "__")])


def onboard_file(kiara: Kiara, uploaded_file):

    if uploaded_file:
        if isinstance(uploaded_file, UploadedFile):
            if uploaded_file.name not in kiara.data_store.alias_names:
                import_bytes(kiara=kiara, uploaded_file=uploaded_file)
        else:
            for x in uploaded_file:
                if x.name in kiara.data_store.alias_names:
                    continue
                import_bytes(kiara=kiara, uploaded_file=x)


def onboard_file_bundle(
    kiara: Kiara, uploaded_files, aliases: typing.Optional[typing.Iterable[str]] = None
):

    if not uploaded_files:
        return

    if aliases is None:
        aliases = []
    if isinstance(aliases, str):
        aliases = [aliases]

    bundle_aliases = [f"{x}_uploaded_files" for x in aliases]

    if isinstance(uploaded_files, UploadedFile):
        uploaded_files = [uploaded_files]

    with tempfile.TemporaryDirectory() as tmpdirname:
        for uf in uploaded_files:
            path = os.path.join(tmpdirname, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())

        inputs = {
            "path": tmpdirname,
            "aliases": aliases,
            "file_bundle_aliases": bundle_aliases,
        }
        kiara.run("table.import.from_local_folder", inputs=inputs)

    shutil.rmtree(tmpdirname, ignore_errors=True)


def check_workflow_status(workflow: KiaraWorkflow):

    status = []
    failed = {}
    for step_id in workflow.pipeline.step_ids:
        job = workflow.controller.get_job_details(step_id)
        if not job:
            continue
        if job.status == JobStatus.FAILED:
            failed[step_id] = job.error if job.error else "-- no error details --"

    print()
    if failed:
        status.append("Error: One or several workflow steps failed!\n")
        for s_id, msg in failed.items():
            status.append(f" - {s_id}: {msg}")

        return (False, status)
    else:
        status.append("No errors.")
        return (True, status)


class MultiPageApp(object):
    def __init__(self, streamlit):

        self._streamlit = streamlit

        if "kiara" not in self._streamlit.session_state:
            print("Create kiara object.")
            self._kiara = Kiara()
            self._streamlit.session_state["kiara"] = self._kiara
        else:
            self._kiara = self._streamlit.session_state["kiara"]

        self._pages = {}

    @property
    def kiara(self) -> Kiara:
        return self._kiara

    def add_page(self, title: str, func: typing.Callable):

        if title in self._pages.keys():
            raise Exception(f"Duplicate page: {title}")

        self._pages[title] = func

    def run(self):
        # Drodown to select the page to run
        page = self._streamlit.sidebar.selectbox(
            "Step navigation",
            self._pages.keys(),
        )

        # run the app function
        self._pages[page](kiara=self._kiara)
