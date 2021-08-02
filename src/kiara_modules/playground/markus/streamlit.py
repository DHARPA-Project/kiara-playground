import os
import tempfile
import typing

from streamlit.uploaded_file_manager import UploadedFile

from kiara import Kiara
from kiara.data import Value, ValueSet
from kiara.pipeline.controller.batch import BatchController, BatchControllerManual
from kiara.workflow.kiara_workflow import KiaraWorkflow


def init_session(st, module_type: str, module_config: typing.Optional[typing.Mapping[str, typing.Any]]=None):

    if "kiara" not in st.session_state:
        print("Create kiara object.")
        kiara = Kiara()
        st.session_state["kiara"] = kiara
    else:
        kiara = st.session_state["kiara"]

    if "workflow" not in st.session_state:
        print(f"Create workflow in session: {module_type}")

        controller = BatchControllerManual()
        workflow: KiaraWorkflow = kiara.create_workflow(module_type, module_config=module_config, controller=controller)
        st.session_state["workflow"] = workflow
    else:
        workflow = st.session_state["workflow"]

    return (kiara, workflow,)

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

    controller: BatchControllerManual = workflow.pipeline.controller
    controller.process_stage(stage_nr=stage_nr)

def get_step_output(workflow, step_id: str, output_name: str) -> Value:

    values: ValueSet = workflow.pipeline.get_step_outputs(step_id=step_id)
    value = values.get_value_obj(output_name)

    return value

def find_all_aliases_of_type(kiara: Kiara, value_type: str) -> typing.List[str]:

    result = []
    for alias in kiara.data_store.aliases:
        md = kiara.data_store.get_metadata_for_id(alias)
        if md.value_type == value_type:
            result.append(alias)

    return result

def import_bytes(kiara: Kiara, uploaded_file: UploadedFile):

    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, uploaded_file.name)
        with open(path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".csv"):
            kiara.run("table.import.from_local_file", inputs={"path": path, "aliases": [uploaded_file.name[0:-4]], "file_aliases": [uploaded_file.name]})
        else:
            kiara.run("import.local_file", inputs={"path": path, "aliases": [uploaded_file.name]})

def onboard_file(kiara: Kiara, st, uploaded_file):

    if uploaded_file:
        if isinstance(uploaded_file, UploadedFile):
            if uploaded_file.name not in kiara.data_store.aliases:
                import_bytes(kiara=kiara, uploaded_file=uploaded_file)
        else:
            for x in uploaded_file:
                if x.name in kiara.data_store.aliases:
                    continue
                import_bytes(kiara=kiara, uploaded_file=x)


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
        status.append(
            "Error: One or several workflow steps failed!\n"
        )
        for s_id, msg in failed.items():
            status.append(f" - {s_id}: {msg}")

        return (False, status)
    else:
        status.append(
            "No errors."
        )
        return (True, status)