# -*- coding: utf-8 -*-
import json
import os
import typing

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit.delta_generator import DeltaGenerator
from streamlit_observable import observable

import kiara_streamlit
from kiara_streamlit.pipelines import PipelineApp
from kiara_streamlit.pipelines.pages import PipelinePage

st.set_page_config(page_title="Kiara-streamlit auto-rendered pipeline", layout="wide")

pipelines_folder = os.path.join(os.path.dirname(__file__), "pipelines")
kiara_streamlit.init(kiara_config={"extra_pipeline_folders": [pipelines_folder]})


# =======================================================================================
# create one class per page, the only method that needs to be implemented is 'run_page'

class GraphAssemblyPage(PipelinePage):

    def find_likely_index(self, options: typing.Iterable, keyword: str):

        for idx, alias in enumerate(options):
            if keyword.lower() in alias.lower():
                return idx

        return 0

    def get_table_column_names(self, id):

        if not id:
            return []
        md = st.kiara.data_store.get_value_obj(id)
        if not md:
            return []
        return md.metadata["table"]["metadata_item"]["column_names"]

    def run_page(self, st: DeltaGenerator):

        left, right = st.columns([3,1])
        edges_table = st.kiara.value_input_table("Select the table containing the graph edges",
                                                 add_no_value_option=True, container=left)
        nodes_table = st.kiara.value_input_table("Select the table containing node attributes",
                                                 add_no_value_option=True, container=right)

        edge_column_names = self.get_table_column_names(edges_table)
        nodes_column_names = self.get_table_column_names(nodes_table)

        default_source_name = self.find_likely_index(edge_column_names, "source")
        default_target_name = self.find_likely_index(edge_column_names, "target")
        default_weight_name = self.find_likely_index(edge_column_names, "weight")
        default_id_name = self.find_likely_index(nodes_column_names, "Id")

        input_source, input_target, input_weight, input_nodes_index = st.columns([1, 1, 1, 1])

        source_column_name = input_source.selectbox(
            "Source column name", edge_column_names, index=default_source_name
        )
        target_column_name = input_target.selectbox(
            "Target column name", edge_column_names, index=default_target_name
        )
        weight_column_name = input_weight.selectbox(
            "Weight column name", edge_column_names, index=default_weight_name
        )
        nodes_index_name = input_nodes_index.selectbox(
            "Nodes table_index", nodes_column_names, index=default_id_name
        )

        create_button = st.button(label="Create graph")

        if create_button:
            inputs = {
                "edges_table": edges_table,
                "nodes_table": nodes_table,
                "source_column_name": source_column_name,
                "target_column_name": target_column_name,
                "weight_column_name": weight_column_name,
                "index_column_name": nodes_index_name
            }
            self.pipeline.inputs.set_values(**inputs)
            self.pipeline_controller.process_pipeline()

        outputs = self.pipeline.get_step_outputs("augment_graph")
        graph_value = outputs.get_value_obj("graph")

        graph_col, info_col = st.columns([8, 2])

        md = "### Graph properties"
        if graph_value.item_is_valid():
            properties = self.pipeline.get_step_outputs("graph_properties")
            st.kiara.write_value(graph_value, container=graph_col)
            for prop, value in properties.get_all_value_data().items():
                md = f"{md}\n- {prop}: *{value}*"
        else:
            md = f"{md}\n- no graph"
        info_col.write(md)

        if graph_value.item_is_valid():
            info_col.write("### Save graph")
            alias_field = info_col.text_input("Alias")
            save_graph_button = info_col.button("Save")
            if save_graph_button:
                if not graph_value.item_is_valid():
                    info_col.write("No valid graph object.")
                if not alias_field:
                    info_col.write("No alias provided, not saving.")

                graph_value.save(aliases=[alias_field])
                info_col.write("Graph saved.")

# ===============================================================================================================
# main app

main_pipeline = os.path.join(pipelines_folder, "network_analysis_streamlit.yaml")

app = PipelineApp.create(
    pipeline=main_pipeline, config={"show_pipeline_status": True, "show_prev_and_next_buttons": True}
)

if not app.pages:
    app.add_page(GraphAssemblyPage(id="Network graph assembly"))

app.run()
