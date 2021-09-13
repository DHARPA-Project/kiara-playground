# -*- coding: utf-8 -*-

# run with:
#
# streamlit run examples/streamlit/streamlit.py

import streamlit as st
from kiara import Kiara
from kiara.workflow.kiara_workflow import KiaraWorkflow
from streamlit.uploaded_file_manager import UploadedFile

st.title("Kiara/streamlit experiment - Network analysis")

if "kiara" not in st.session_state:
    print("CREATE KIARA")
    kiara = Kiara()
    st.session_state["kiara"] = kiara
else:
    kiara = st.session_state["kiara"]

if "workflow" not in st.session_state:
    print("CREATE WORKFLOW")
    workflow: KiaraWorkflow = kiara.create_workflow(
        "/home/markus/projects/dharpa/kiara-playground/examples/pipelines/network_analysis_end_to_end.json"
    )
    st.session_state["workflow"] = workflow
else:
    workflow = st.session_state["workflow"]


def add_file(f):
    if f:
        if isinstance(f, UploadedFile):
            if f.name not in files.keys():
                files[f.name] = f
        else:
            files.update({x.name: x for x in f if x and x.name not in files.keys()})


fp = st.sidebar.file_uploader("hello", accept_multiple_files=True)
if fp:
    add_file(fp)

st.write(files)
