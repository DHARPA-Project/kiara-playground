# -*- coding: utf-8 -*-

import os

import streamlit as st
from jinja2 import FileSystemLoader, Environment

import kiara_streamlit


pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})

pipeline_path = os.path.join(pipelines_folder, "tm_pipeline.yaml")
pipeline = st.kiara.create_pipeline(pipeline_path)

templates_path = os.path.join(os.path.dirname(__file__), "templates")

loader = FileSystemLoader(templates_path)
env: Environment = Environment(loader=loader)

template = env.get_template("status.j2")
rendered = template.render(kiara=st.kiara, pipeline=pipeline, step_id="text_pre_processing")

import streamlit.components.v1 as components
components.html(rendered, height=500)
