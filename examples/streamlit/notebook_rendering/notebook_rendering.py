# -*- coding: utf-8 -*-
import os
import typing

import streamlit as st

import kiara_streamlit
from kiara.defaults import KIARA_RESOURCES_FOLDER
from kiara.info.kiara import KiaraContext
from kiara.rendering import KiaraRenderer
from kiara.rendering.jinja import JinjaRenderer
from kiara_modules.playground.markus.rendering import JinjaPipelineRendererPlayground
from kiara_streamlit import KiaraStreamlit

st.set_page_config(page_title="Kiara experiment: dynamic operation", layout="centered")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})
kiara_streamlit.init()

# pipeline = os.path.join(pipelines_folder, "tm_pipeline.yaml")
#
# templates_folder = os.path.join(os.path.dirname(__file__), "templates")
# template = os.path.join(templates_folder, "notebook.j2")
#
#
#
#
# def render_notebook(module: str, template: str, input_values: typing.Mapping[str, typing.Any], post_process: bool=True):
#
#     template_renderer: KiaraRenderer = JinjaPipelineRendererPlayground(
#         kiara=st.kiara
#     )
#
#     inputs = {
#         "module": module,
#         "template": template,
#         "pipeline_inputs": input_values
#     }
#
#     augmented_inputs = template_renderer._augment_inputs(**inputs)
#     rendered = template_renderer._render_template(inputs=augmented_inputs)
#
#     post_processed = template_renderer._post_process(rendered=rendered, inputs=augmented_inputs)
#     return rendered, post_processed
#
#
# render_button = st.button("Render notebook")
# if render_button:
#     input_values = {
#         "corpus_table": "value:ci_newspaper_subcorpora"
#     }
#     rendered, post_processed = render_notebook(module=pipeline, template=template, post_process=False, input_values=input_values)
#
#     st.write(f"```\n{rendered}\n```")
#
#     with open("/tmp/test.ipynb", 'wt') as f:
#         f.write(post_processed)

ktx = KiaraContext.create(kiara=st.kiara, ignore_errors=True)
st.write(f"``` json\n{ktx.json(indent=2)}\n```")
