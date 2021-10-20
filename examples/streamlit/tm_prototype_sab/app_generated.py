# -*- coding: utf-8 -*-
import os

import streamlit as st

import kiara_streamlit
from kiara_streamlit.pipelines import PipelineApp
from kiara_streamlit.pipelines.pages.step import StepPage

st.set_page_config(page_title="Kiara-streamlit auto-rendered pipeline", layout="wide")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
main_pipeline = os.path.join(pipelines_folder, "tm_pipeline.yaml")

kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})

app = PipelineApp.create(
    pipeline=main_pipeline
)

if not app.pages:
    for step_id in app.pipeline.step_ids:
        # inputs = app.pipeline.get_pipeline_inputs_for_step(step_id)
        # if inputs:
        page = StepPage(id=step_id)
        app.add_page(page)

app.run()
