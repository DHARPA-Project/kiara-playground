# -*- coding: utf-8 -*-
import os

import streamlit as st

import kiara_streamlit
from kiara import Kiara
from kiara.data.onboarding.batch import BatchOnboard
from kiara_streamlit.data_centric import DataCentricApp

st.set_page_config(page_title="Kiara experiment: data-centric workflows", layout="wide")

if "workflow_pages" not in st.session_state.keys():
    st.session_state.workflow_pages = {}

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})

def onboard_folder(kiara: Kiara, pipeline_folder: str, corpus_path: str, value_alias: str):

    inputs = {
        "corpus_path": corpus_path
    }
    pipeline = os.path.join(pipeline_folder, "example_onboard.yaml")

    store_config_dict = {
        "outputs": [
            {"alias_template": value_alias}
        ]
    }

    onboard_config = {
        "module_type": pipeline,
        "inputs": inputs,
        "store_config": store_config_dict
    }

    # store_config = ValueStoreConfig(**store_config_dict)
    onboarder = BatchOnboard.create(kiara=kiara, **onboard_config)
    print(f"kiara data store: {kiara.data_store.base_path}")
    with st.spinner('Onboarding example data...'):
        results = onboarder.run("test")
        aliases = set()
        for _a in results.values():
            aliases.update(_a)
        print(f"Onboarded example data, available aliases: {', '.join(aliases)}")


@st.experimental_singleton
def onboard_files(_kiara: Kiara):

    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

    aliases = _kiara.data_store.alias_names
    if aliases:
        all_match = True
        for alias in ["CI_newspaper_subcorpora"]:
            if alias not in aliases:
                all_match = False
                break
        if all_match:
            print("Data already onboarded.")
            return

    base_folder = os.path.dirname(__file__)
    data_folder = os.path.join(base_folder, "..", "tm_prototype_sab", "data")
    pipeline_folder = os.path.join(base_folder, "pipelines")
    corpus_path = os.path.join(data_folder, "CI_newspaper_subcorpora")

    onboard_folder(kiara=_kiara, pipeline_folder=pipeline_folder, corpus_path=corpus_path, value_alias="CI_newspaper_subcorpora")


    for dir in os.listdir(corpus_path):
        path = os.path.join(corpus_path, dir)
        if os.path.isdir(path):
            name = dir.replace("'", "_")
            onboard_folder(kiara=_kiara, pipeline_folder=pipeline_folder, corpus_path=path, value_alias=name)

    return True

onboarded = onboard_files(_kiara=st.kiara)

st.header("Data-centric workflows")

app = DataCentricApp.create(config={"show_pipeline_status": True})

app.run()
